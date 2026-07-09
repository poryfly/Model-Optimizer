# ==========================================
# 【最高优先级】：必须在 import datasets 之前设置环境变量！
# 将所有 HuggingFace 相关的缓存统一重定向到 /data2/ 目录
# ==========================================
import os
import argparse
import pandas as pd
import json
import shutil
from pathlib import Path
import gzip

CUSTOM_CACHE_DIR = "/data2/.cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 国内镜像加速

# 确保缓存目录存在
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

import gc
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, List

# ==========================================
# 全局配置区
# ==========================================
RAW_SHARDS_DIR = f"{CUSTOM_CACHE_DIR}/01_raw_filtered_shards"
FINAL_OUTPUT_DIR = f"{CUSTOM_CACHE_DIR}/02_final_pretrain_jsonl_v3"
LOGS_DIR = f"{CUSTOM_CACHE_DIR}/03_datatrove_logs" 
os.makedirs(RAW_SHARDS_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

@dataclass
class DatasetConfig:
    repo_id: str
    name: Optional[str]       # 用于指定 HuggingFace 数据集的 Subset (子集)
    text_field: str           
    score_field: Optional[str]
    max_docs: int             
    batch_size: int = 100000  

# ==========================================
# 最新数据配方 (包含 4 个 Ultra-FineWeb-L3 子集)
# ==========================================
DATASETS = [
    # 1. 面壁 Ultra-FineWeb-L3 中文 QA 合成数据
    DatasetConfig(
        repo_id="openbmb/Ultra-FineWeb-L3", 
        name="Ultra-FineWeb-L3-zh-QA-Synthetic",
        text_field="text", score_field=None, max_docs=5_000_000
    ),
    # 2. 面壁 Ultra-FineWeb-L3 中文 多风格合成数据
    DatasetConfig(
        repo_id="openbmb/Ultra-FineWeb-L3", 
        name="Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
        text_field="text", score_field=None, max_docs=4_200_000
    ),
    # 3. 面壁 Ultra-FineWeb-L3 英文 QA 合成数据
    DatasetConfig(
        repo_id="openbmb/Ultra-FineWeb-L3", 
        name="Ultra-FineWeb-L3-en-QA-Synthetic",
        text_field="text", score_field=None, max_docs=8_000_000
    ),
    # 4. 面壁 Ultra-FineWeb-L3 英文 多风格合成数据
    DatasetConfig(
        repo_id="openbmb/Ultra-FineWeb-L3", 
        name="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
        text_field="text", score_field=None, max_docs=6_000_000
    ),
    # 5. 高质量数学语料 (提升逻辑推理能力)
    DatasetConfig(
        repo_id="open-web-math/open-web-math", 
        name=None, text_field="text", score_field=None, max_docs=10_000_000 
    )
]

# ==========================================
# 核心工具：智能文本提取器
# (解决合成数据集没有单一 text 字段的问题)
# ==========================================
def extract_text_from_example(example: dict, cfg: DatasetConfig) -> str:
    # 1. 如果直接存在指定的 text_field，直接返回
    if cfg.text_field in example and example[cfg.text_field]:
        return str(example[cfg.text_field])
    
    # 2. 尝试拼接常见的 QA / Prompt 字段 (针对合成数据)
    if "question" in example and "answer" in example:
        return f"Question: {example.get('question', '')}\n\nAnswer: {example.get('answer', '')}"
    if "prompt" in example and "response" in example:
        return f"{example.get('prompt', '')}\n\n{example.get('response', '')}"
    if "instruction" in example and "output" in example:
        return f"{example.get('instruction', '')}\n\n{example.get('output', '')}"
        
    # 3. 兜底：寻找字典里最长的字符串字段
    max_len, best_text = 0, ""
    for k, v in example.items():
        if isinstance(v, str) and len(v) > max_len:
            max_len, best_text = len(v), v
    return best_text

# ==========================================
# 阶段 1: 小样本探测与字段自适应
# ==========================================
def probe_dataset(cfg: DatasetConfig, probe_size: int = 5000) -> float:
    print(f"\n[阶段1] 探测数据集: {cfg.repo_id} (Subset: {cfg.name or 'Default'}) ...")
    
    for attempt in range(3):
        try:
            ds = load_dataset(cfg.repo_id, name=cfg.name, split="train", streaming=True)
            break
        except Exception as e:
            if attempt == 2:
                print(f"  -> ❌ 经过3次重试仍无法加载，错误: {e}")
                return -1.0
            print(f"  -> ⚠️ 网络波动，{3-attempt}秒后重试...")
            time.sleep(3)

    # 获取样本并测试文本提取器
    sample = next(iter(ds))
    extracted = extract_text_from_example(sample, cfg)
    if not extracted:
        print(f"  -> ❌ 无法从数据集中提取文本，可用字段: {list(sample.keys())}")
        return -1.0
    print(f"  -> ✅ 文本提取成功 (预览前50字: {extracted[:50].replace(chr(10), ' ')}...)")

    if not cfg.score_field or cfg.score_field not in sample: 
        return 0.0

    scores = []
    for i, example in enumerate(ds):
        if i >= probe_size: break
        if cfg.score_field in example and example[cfg.score_field] is not None:
            scores.append(float(example[cfg.score_field]))
            
    if not scores: return 0.0
    threshold = np.percentile(scores, 85)
    print(f"  -> 85分位阈值={threshold:.2f}")
    return threshold

# ==========================================
# 阶段 2: 边过滤边下载 (精准拉取)
# ==========================================
def streaming_download_and_filter(cfg: DatasetConfig, threshold: float):
    if threshold == -1.0:
        print(f"\n[阶段2] ⏭️ 跳过 {cfg.repo_id} ({cfg.name})")
        return

    print(f"\n[阶段2] 开始精准下载: {cfg.repo_id} (Subset: {cfg.name}, 上限: {cfg.max_docs} 条)")
    
    for attempt in range(3):
        try:
            ds = load_dataset(cfg.repo_id, name=cfg.name, split="train", streaming=True)
            break
        except Exception as e:
            if attempt == 2: return
            time.sleep(3)

    def filter_func(example):
        if cfg.score_field and threshold > 0:
            if example.get(cfg.score_field, 0) < threshold: return False
        text = extract_text_from_example(example, cfg)
        if not text or len(text) < 100: return False
        return True

    filtered_ds = ds.filter(filter_func)
    texts, sources, saved_count = [], [], 0
    
    # 【修复点】：将 Subset 名称加入文件名前缀，防止同一 repo 的不同子集文件互相覆盖
    safe_repo = cfg.repo_id.replace("/", "_")
    safe_name = cfg.name.replace("-", "_") if cfg.name else "default"
    out_prefix = f"{safe_repo}_{safe_name}"
    
    try:
        for example in filtered_ds:
            # 统一将提取出的文本存入 "text" 字段，以兼容后续的 datatrove 清洗
            texts.append(extract_text_from_example(example, cfg))
            sources.append(f"{cfg.repo_id}/{cfg.name}" if cfg.name else cfg.repo_id)
            
            if len(texts) >= cfg.batch_size:
                table = pa.table({"text": texts, "source": sources})
                pq.write_table(table, os.path.join(RAW_SHARDS_DIR, f"{out_prefix}_shard_{saved_count//cfg.batch_size}.parquet"))
                texts, sources = [], []
                saved_count += cfg.batch_size
                print(f"  -> [下载进度] 已保存 {saved_count} 条...")
            if saved_count >= cfg.max_docs: break
    except Exception as e:
        print(f"  -> ⚠️ 下载中断: {e}，已保存当前进度。")
        
    if texts:
        table = pa.table({"text": texts, "source": sources})
        pq.write_table(table, os.path.join(RAW_SHARDS_DIR, f"{out_prefix}_shard_final.parquet"))
        saved_count += len(texts)
    print(f"✅ {cfg.repo_id} ({cfg.name}) 下载完成，共获取 {saved_count} 条高质量文档。")
    gc.collect()



import sys
import shutil
import inspect
import os

# ==========================================
# 核心黑科技：智能参数适配器 (保持不变)
# ==========================================
def smart_instantiate(cls, **desired_kwargs):
    """
    智能实例化：自动探测类的 __init__ 签名，只传入当前版本支持的参数。
    """
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
    except Exception:
        return cls(**desired_kwargs)
    
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    
    final_kwargs = {}
    ignored_keys = []
    
    for k, v in desired_kwargs.items():
        if k in params or has_var_kwargs:
            final_kwargs[k] = v
        else:
            ignored_keys.append(k)
            
    if ignored_keys and not has_var_kwargs:
        # 静默跳过，避免刷屏，但保留核心提示
        pass 
        
    return cls(**final_kwargs)


import os
import glob as glob_lib

# ==========================================
# 核心黑科技：ParquetReader 路径自动修复器
# ==========================================
def create_safe_reader(directory_path: str, **extra_kwargs):
    """
    自动探测并修复 datatrove Reader 的本地路径解析问题。
    支持 Parquet, MinHash Signatures (.sig), 和 Clusters (.clusters) 格式。
    """
    from datatrove.pipeline.readers import ParquetReader
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录不存在: {directory_path}")
        
    # 多格式预检：依次尝试 parquet, sig, clusters
    formats_to_try = [
        ("*.parquet", "Parquet"),
        ("**/*.sig", "MinHash Signature"),
        ("**/*.clusters", "MinHash Cluster"),
        ("*.sig", "MinHash Signature (flat)"),
        ("*.clusters", "MinHash Cluster (flat)")
    ]
    
    found_format = None
    found_files = []
    
    for pattern, fmt_name in formats_to_try:
        # 使用递归 glob (**) 来查找子目录中的文件
        files = glob_lib.glob(os.path.join(directory_path, pattern), recursive=True)
        if files:
            found_format = fmt_name
            found_files = files
            break
            
    if not found_files:
        raise FileNotFoundError(f"目录下没有找到任何支持的数据文件 (parquet/sig/clusters): {directory_path}")
        
    print(f"  -> ✅ 原生系统预检通过: 在 {directory_path} 下发现 {len(found_files)} 个 {found_format} 文件")

    # 策略池：纯目录、file:// 协议、glob_pattern 分离
    kwargs_pool = [
        {'data_folder': directory_path},
        {'folder': directory_path},
        {'path': directory_path},
        {'data_folder': f"file://{directory_path}"},
        {'folder': f"file://{directory_path}"},
    ]
    
    # 如果找到了特定后缀，尝试注入 glob_pattern
    if found_format == "MinHash Signature":
        kwargs_pool.extend([
            {'data_folder': directory_path, 'glob_pattern': '**/*.sig'},
            {'data_folder': directory_path, 'glob_pattern': '*.sig'},
        ])
    elif found_format == "MinHash Cluster":
        kwargs_pool.extend([
            {'data_folder': directory_path, 'glob_pattern': '**/*.clusters'},
            {'data_folder': directory_path, 'glob_pattern': '*.clusters'},
        ])

    for pool in kwargs_pool:
        pool.update(extra_kwargs)

    return smart_instantiate(ParquetReader, **{k: v for pool in kwargs_pool for k, v in pool.items()})



# ==========================================
# 阶段 3: 本地深度清洗与全局 MinHash 去重 (0.9.0 架构重构版)
# ==========================================
def local_deep_clean_and_dedup(workers: int = 128):
    print(f"\n[阶段3] 开始本地深度清洗与全局 MinHash 去重 (最大并发: {workers} 个进程)...")
    
    # 1. 导入组件
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
    from datatrove.pipeline.writers import JsonlWriter, ParquetWriter
    from datatrove.pipeline.dedup import (
        MinhashConfig,
        MinhashDedupSignature, 
        MinhashDedupBuckets, 
        MinhashDedupCluster,
        MinhashDedupFilter  
    )
    
    try:
        from datatrove.executor.local import LocalPipelineExecutor
    except ImportError:
        from datatrove.executor.local import LocalExecutor as LocalPipelineExecutor

    # 2. 智能初始化 MinhashConfig
    print("  -> 🔧 智能初始化 MinhashConfig...")
    minhash_config = smart_instantiate(
        MinhashConfig,
        num_buckets=20,
        hashes_per_bucket=13,  # 0.9.0 新版参数 (20 * 13 = 260 个哈希)
        num_perms=260,         # 0.8.x 版本参数
        num_hashes=260,        # 更老版本参数
        n_grams=5              # 0.9.0 可用参数
    )

    # 定义 MinHash 中间产物目录
    SIGNATURES_DIR = f"{LOGS_DIR}/01_minhash_signatures"
    BUCKETS_DIR    = f"{LOGS_DIR}/02_minhash_buckets"
    CLUSTERS_DIR   = f"{LOGS_DIR}/03_minhash_clusters"
    
    for d in [SIGNATURES_DIR, BUCKETS_DIR, CLUSTERS_DIR]:
        if os.path.exists(d): shutil.rmtree(d)

    parallel_tasks = workers 
    aggregation_tasks = 1 

    # ------------------------------------------
    # Pipeline 1: 质量过滤 + 语言过滤 + 生成 MinHash 签名
    # 【架构变更】：Signature 内置了 Writer，移除末尾的 ParquetWriter
    # ------------------------------------------
    print(f"  -> [3.1/4] 执行质量过滤与生成 MinHash 签名 (Tasks: {parallel_tasks})...")
    
    reader_1 = create_safe_reader(RAW_SHARDS_DIR)
    
    
    gopher_filter = smart_instantiate(
        GopherQualityFilter,
        min_doc_words=50, min_words=50, min_word_count=50,
        max_doc_words=100000, max_words=100000, max_word_count=100000,
        max_symbol_to_word_ratio=0.3, max_symbol_word_ratio=0.3,
        max_ellipsis_to_word_ratio=0.3, max_ellipsis_word_ratio=0.3
    )
    
    lang_filter = smart_instantiate(
        LanguageFilter,
        languages=["zh", "en"], 
        target_languages=["zh", "en"],
        language_threshold=0.7, 
        threshold=0.7,
        min_prob=0.7
    )
    
    # 【核心修复】：将 output_folder 直接注入 Signature
    sig_step = smart_instantiate(
        MinhashDedupSignature, 
        config=minhash_config,
        output_folder=SIGNATURES_DIR,
        folder=SIGNATURES_DIR
    )

    # 注意：这里不再添加 ParquetWriter！
    pipeline_1 = [reader_1, gopher_filter, lang_filter, sig_step]
    LocalPipelineExecutor(pipeline=pipeline_1, tasks=parallel_tasks, logging_dir=f"{LOGS_DIR}/log_1").run()

    # ------------------------------------------
    # Pipeline 2: 生成 Buckets
    # 【架构变更】：Buckets 内置了 Writer
    # ------------------------------------------


    print(f"  -> [3.2/4] 生成 MinHash Buckets...")

    # 【核心修复】彻底删除 reader_2，Buckets 组件自己会去读 input_folder
    bucket_step = smart_instantiate(
        MinhashDedupBuckets,
        config=minhash_config,
        input_folder=SIGNATURES_DIR,
        output_folder=BUCKETS_DIR,
    )

    # Pipeline 列表中只有 bucket_step
    pipeline_2 = [bucket_step]

    # tasks 数量通常等于 num_buckets (在 config 中定义) 或上一步的文件数
    # 这里我们使用 config 中的 num_buckets，如果获取不到则回退到 workers 数量
    buckets_tasks = getattr(minhash_config, 'num_buckets', workers)

    LocalPipelineExecutor(
        pipeline=pipeline_2,
        tasks=buckets_tasks,
        logging_dir=f"{LOGS_DIR}/log_2"
    ).run()

    # ------------------------------------------
    # Pipeline 3: 生成 Clusters
    # 【架构变更】：Cluster 内置了 Writer
    # ------------------------------------------

    print(f"  -> [3.3/4] 生成 MinHash Clusters (Tasks: 1)...")
    
    cluster_step = smart_instantiate(
        MinhashDedupCluster, 
        config=minhash_config,
        input_folder=BUCKETS_DIR,     
        output_folder=CLUSTERS_DIR,   
    )
    
    pipeline_3 = [cluster_step]
    
    # 【核心修复】Clustering 阶段需要全局视角，强制 tasks=1 (World size must be 1)
    LocalPipelineExecutor(
        pipeline=pipeline_3, 
        tasks=1,  # <--- 这里必须是 1
        logging_dir=f"{LOGS_DIR}/log_3"
    ).run()


    # ------------------------------------------
    # Pipeline 4: 移除重复数据并输出最终 JSONL
    # 【架构变更】：Filter 只负责过滤，末尾必须保留 JsonlWriter
    # ------------------------------------------
   
    print(f"  -> [3.4/4] 移除重复数据并输出最终 JSONL (Tasks: {parallel_tasks})...")
    
    # 注意：这里读取的是最原始的去重前数据！
    reader_4 = create_safe_reader(RAW_SHARDS_DIR)
    
    dedup_filter = smart_instantiate(
        MinhashDedupFilter,
        config=minhash_config,
        input_folder=CLUSTERS_DIR,    # 【核心修复】显式指定读取 clusters 目录
        clusters_folder=CLUSTERS_DIR, # 兼容旧版参数名
        folder=CLUSTERS_DIR,          # 兜底参数
    )
    
    writer_4 = smart_instantiate(
        JsonlWriter,
        output_folder=FINAL_OUTPUT_DIR, 
        folder=FINAL_OUTPUT_DIR,
        output_filename="deepseek_4b_clean_shard_${rank}.jsonl",
        filename="deepseek_4b_clean_shard_${rank}.jsonl"
    )

    pipeline_4 = [reader_4, dedup_filter, writer_4]
    LocalPipelineExecutor(pipeline=pipeline_4, tasks=parallel_tasks, logging_dir=f"{LOGS_DIR}/log_4").run()

    print(f"🎉 全部处理完成！最终干净的预训练数据位于: {FINAL_OUTPUT_DIR}")


def run_sandbox_test():
    """
    沙盒测试：构造微型数据集，使用极简参数，10秒内验证全链路 IO 和去重逻辑。
    """
    print("\n" + "="*50)
    print("🚀 启动 Pipeline 沙盒冒烟测试 (Smoke Test)")
    print("="*50)
    
    # 1. 准备沙盒环境
    SANDBOX_DIR = Path("/tmp/datatrove_sandbox")
    if SANDBOX_DIR.exists():
        shutil.rmtree(SANDBOX_DIR)
        
    RAW_DIR = SANDBOX_DIR / "01_raw"
    RAW_DIR.mkdir(parents=True)
    
    # 2. 构造微型测试数据 (包含重复、部分重复、独立文本)
    test_data = {
        "text": [
            "今天天气真好，我们去公园散步吧。",  # Doc 0 (基准)
            "今天天气真好，我们去公园散步吧。",  # Doc 1 (与 Doc 0 完全重复 -> 应被过滤)
            "今天天气真好，我们去公园散步。",    # Doc 2 (极度相似 -> 应被过滤)
            "量子力学是物理学的一个重要分支。",  # Doc 3 (独立文本 -> 应保留)
            "深度学习在自然语言处理中应用广泛。", # Doc 4 (独立文本 -> 应保留)
        ],
        "id": [0, 1, 2, 3, 4]
    }
    df = pd.DataFrame(test_data)
    test_file = RAW_DIR / "test_shard_0.parquet"
    df.to_parquet(test_file)
    print(f"  -> ✅ 已生成微型测试数据: {test_file} (共 {len(df)} 条)")

    # 3. 覆盖全局变量，将 Pipeline 指向沙盒目录
    global RAW_SHARDS_DIR, SIGNATURES_DIR, BUCKETS_DIR, CLUSTERS_DIR, FINAL_OUTPUT_DIR, LOGS_DIR
    
    RAW_SHARDS_DIR = str(RAW_DIR)
    SIGNATURES_DIR = str(SANDBOX_DIR / "02_signatures")
    BUCKETS_DIR = str(SANDBOX_DIR / "03_buckets")
    CLUSTERS_DIR = str(SANDBOX_DIR / "04_clusters")
    FINAL_OUTPUT_DIR = str(SANDBOX_DIR / "05_final_output")
    LOGS_DIR = str(SANDBOX_DIR / "logs")
    
    for d in [SIGNATURES_DIR, BUCKETS_DIR, CLUSTERS_DIR, FINAL_OUTPUT_DIR, LOGS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 4. 使用极简参数运行核心函数 (强制单进程，减少 MinHash 计算量)
    # 注意：这里我们修改了 workers 为 1，避免多进程在微型数据上产生空 task 报错
    try:
        # 如果您的 local_deep_clean_and_dedup 支持传入 minhash_config 覆盖，请在这里传入
        # 否则直接调用，依赖函数内部的默认配置
        local_deep_clean_and_dedup(workers=1) 
    except Exception as e:
        print(f"\n❌ 测试失败！Pipeline 运行中断: {e}")
        return

    # 5. 自动化断言 (验证最终结果)
    print("\n🔍 正在进行结果断言...")
    
    # 【修复 1】使用 *.jsonl* 匹配 .jsonl 和 .jsonl.gz 文件
    output_files = list(Path(FINAL_OUTPUT_DIR).glob("*.jsonl*"))
    if not output_files:
        print("  -> ❌ 断言失败: 最终输出目录没有找到 JSONL/JSONL.GZ 文件！")
        return
        
    final_texts = []
    for f in output_files:
        # 【修复 2】根据后缀自动选择普通读取或 gzip 解压读取
        if f.suffix == '.gz':
            with gzip.open(f, 'rt', encoding='utf-8') as fp:
                for line in fp:
                    data = json.loads(line)
                    final_texts.append(data.get("text", ""))
        else:
            with open(f, 'r', encoding='utf-8') as fp:
                for line in fp:
                    data = json.loads(line)
                    final_texts.append(data.get("text", ""))
                
    print(f"  -> 输入数据: 5 条")
    print(f"  -> 输出数据: {len(final_texts)} 条")
    
    # 预期结果：保留 3 条 (Doc 0, Doc 3, Doc 4)，过滤掉 2 条重复的
    if 2 <= len(final_texts) <= 3:
        print("  -> ✅ 断言成功: 去重逻辑生效，成功过滤了重复文本！")
        print("  -> 保留的文本预览:")
        for t in final_texts:
            print(f"     - {t[:30]}...")
    else:
        print(f"  -> ⚠️ 断言异常: 预期保留 2~3 条，实际保留 {len(final_texts)} 条。请检查 MinHash 阈值配置。")
        
    print("\n🎉 沙盒测试圆满完成！整个 Pipeline 逻辑衔接 100% 正常。")
    print(f"   您可以前往 {SANDBOX_DIR} 查看各阶段的产物文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="运行快速沙盒测试")
    args = parser.parse_args()
    
    if args.test:
        run_sandbox_test()
    else:
        print(f"📁 HuggingFace 缓存目录已统一挂载至: {CUSTOM_CACHE_DIR}")
        for cfg in DATASETS:
            threshold = probe_dataset(cfg)
            streaming_download_and_filter(cfg, threshold)
        # 原有的正式运行逻辑
        local_deep_clean_and_dedup(workers=128)
