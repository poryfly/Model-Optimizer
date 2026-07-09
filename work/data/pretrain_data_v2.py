# ==========================================
# 【最高优先级】：必须在 import datasets 之前设置环境变量！
# ==========================================
import os
CUSTOM_CACHE_DIR = "/data2/.cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

import argparse, json, shutil, time, gc, inspect, glob as glob_lib
import gzip, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datasets import load_dataset

# ==========================================
# 全局配置区 (按子集隔离)
# ==========================================
BASE_RAW_DIR = f"{CUSTOM_CACHE_DIR}/01_raw_filtered_shards"
BASE_FINAL_DIR = f"{CUSTOM_CACHE_DIR}/02_final_pretrain_jsonl_v3"
BASE_LOGS_DIR = f"{CUSTOM_CACHE_DIR}/03_datatrove_logs"
MANIFEST_PATH = f"{CUSTOM_CACHE_DIR}/manifest.json"

for d in [BASE_RAW_DIR, BASE_FINAL_DIR, BASE_LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

@dataclass
class DatasetConfig:
    repo_id: str
    name: Optional[str]
    text_field: str
    score_field: Optional[str]
    target_docs: int          # ✅ 改名：不再是硬截断，而是目标量(建议比原max_docs多20%)
    batch_size: int = 100000
    subset_key: str = ""      # ✅ 新增：用于目录命名和swift注册的唯一key

# ==========================================
# 最新数据配方 (增加 subset_key)
# ==========================================
DATASETS = [
    DatasetConfig(
        repo_id="openbmb/Ultra-FineWeb-L3",
        name="Ultra-FineWeb-L3-zh-QA-Synthetic",
        text_field="text", score_field=None,
        target_docs=6_000_000,   # 原5M × 1.2 缓冲
        subset_key="zh_qa_synthetic"
    ),
    DatasetConfig(
        repo_id="openbmb/Ultra-FineWeb-L3",
        name="Ultra-FineWeb-L3-zh-Multi-Style-Synthetic",
        text_field="text", score_field=None,
        target_docs=5_040_000,   # 原4.2M × 1.2
        subset_key="zh_multi_style"
    ),
    DatasetConfig(
        repo_id="openbmb/Ultra-FineWeb-L3",
        name="Ultra-FineWeb-L3-en-QA-Synthetic",
        text_field="text", score_field=None,
        target_docs=9_600_000,   # 原8M × 1.2
        subset_key="en_qa_synthetic"
    ),
    DatasetConfig(
        repo_id="openbmb/Ultra-FineWeb-L3",
        name="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
        text_field="text", score_field=None,
        target_docs=7_200_000,   # 原6M × 1.2
        subset_key="en_multi_style"
    ),
    DatasetConfig(
        repo_id="open-web-math/open-web-math",
        name=None, text_field="text", score_field=None,
        target_docs=12_000_000,  # 原10M × 1.2
        subset_key="open_web_math"
    ),
]

# ==========================================
# 核心工具函数 (完全保留原有实现)
# ==========================================
def extract_text_from_example(example: dict, cfg: DatasetConfig) -> str:
    if cfg.text_field in example and example[cfg.text_field]:
        return str(example[cfg.text_field])
    if "question" in example and "answer" in example:
        return f"Question: {example.get('question', '')}\n\nAnswer: {example.get('answer', '')}"
    if "prompt" in example and "response" in example:
        return f"{example.get('prompt', '')}\n\n{example.get('response', '')}"
    if "instruction" in example and "output" in example:
        return f"{example.get('instruction', '')}\n\n{example.get('output', '')}"
    max_len, best_text = 0, ""
    for k, v in example.items():
        if isinstance(v, str) and len(v) > max_len:
            max_len, best_text = len(v), v
    return best_text

def smart_instantiate(cls, **desired_kwargs):
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
    except Exception:
        return cls(**desired_kwargs)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    final_kwargs = {}
    for k, v in desired_kwargs.items():
        if k in params or has_var_kwargs:
            final_kwargs[k] = v
    return cls(**final_kwargs)

def create_safe_reader(directory_path: str, **extra_kwargs):
    from datatrove.pipeline.readers import ParquetReader
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录不存在: {directory_path}")
    formats_to_try = [
        ("*.parquet", "Parquet"), ("**/*.sig", "MinHash Signature"),
        ("**/*.clusters", "MinHash Cluster"), ("*.sig", "MinHash Signature (flat)"),
        ("*.clusters", "MinHash Cluster (flat)")
    ]
    found_format, found_files = None, []
    for pattern, fmt_name in formats_to_try:
        files = glob_lib.glob(os.path.join(directory_path, pattern), recursive=True)
        if files:
            found_format, found_files = fmt_name, files
            break
    if not found_files:
        raise FileNotFoundError(f"目录下没有找到支持的数据文件: {directory_path}")
    print(f"  -> ✅ 预检通过: {len(found_files)} 个 {found_format} 文件")
    kwargs_pool = [
        {'data_folder': directory_path}, {'folder': directory_path},
        {'path': directory_path},
        {'data_folder': f"file://{directory_path}"}, {'folder': f"file://{directory_path}"},
    ]
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
# ✅ 重构：按子集隔离的下载函数
# ==========================================
def probe_dataset(cfg: DatasetConfig, probe_size: int = 5000) -> float:
    print(f"\n[探测] {cfg.subset_key}: {cfg.repo_id} (Subset: {cfg.name or 'Default'})")
    for attempt in range(3):
        try:
            ds = load_dataset(cfg.repo_id, name=cfg.name, split="train", streaming=True)
            break
        except Exception as e:
            if attempt == 2:
                print(f"  -> ❌ 加载失败: {e}")
                return -1.0
            time.sleep(3)
    sample = next(iter(ds))
    extracted = extract_text_from_example(sample, cfg)
    if not extracted:
        print(f"  -> ❌ 无法提取文本, 可用字段: {list(sample.keys())}")
        return -1.0
    print(f"  -> ✅ 文本提取成功 ({extracted[:50].replace(chr(10), ' ')}...)")
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

def streaming_download_and_filter(cfg: DatasetConfig, threshold: float):
    """✅ 改造：输出到子集独立目录"""
    if threshold == -1.0:
        print(f"\n[下载] ⏭️ 跳过 {cfg.subset_key}")
        return

    # ✅ 子集隔离的原始数据目录
    raw_dir = os.path.join(BASE_RAW_DIR, cfg.subset_key)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n[下载] {cfg.subset_key}: 上限 {cfg.target_docs} 条 -> {raw_dir}")
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

    try:
        for example in filtered_ds:
            text = extract_text_from_example(example, cfg)
            texts.append(text)
            # ✅ 注入来源元数据
            source_tag = f"{cfg.repo_id}/{cfg.name}" if cfg.name else cfg.repo_id
            sources.append(source_tag)

            if len(texts) >= cfg.batch_size:
                table = pa.table({"text": texts, "source": sources})
                shard_idx = saved_count // cfg.batch_size
                pq.write_table(table, os.path.join(raw_dir, f"shard_{shard_idx:05d}.parquet"))
                texts, sources = [], []
                saved_count += cfg.batch_size
                print(f"  -> 已保存 {saved_count} 条...")
            if saved_count >= cfg.target_docs: break
    except Exception as e:
        print(f"  -> ⚠️ 下载中断: {e}")

    if texts:
        table = pa.table({"text": texts, "source": sources})
        pq.write_table(table, os.path.join(raw_dir, f"shard_final.parquet"))
        saved_count += len(texts)

    print(f"✅ {cfg.subset_key} 下载完成: {saved_count} 条")
    gc.collect()


# ==========================================
# ✅ 重构：按子集隔离的清洗去重函数
# ==========================================
def clean_and_dedup_subset(subset_key: str, workers: int = 32):
    """✅ 对单个子集独立执行完整清洗+MinHash去重流水线"""
    raw_dir = os.path.join(BASE_RAW_DIR, subset_key)
    final_dir = os.path.join(BASE_FINAL_DIR, subset_key)
    logs_dir = os.path.join(BASE_LOGS_DIR, subset_key)

    if not os.path.exists(raw_dir):
        print(f"\n[清洗] ⏭️ 跳过 {subset_key}: 原始数据目录不存在")
        return None

    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    SIGNATURES_DIR = f"{logs_dir}/01_signatures"
    BUCKETS_DIR = f"{logs_dir}/02_buckets"
    CLUSTERS_DIR = f"{logs_dir}/03_clusters"
    for d in [SIGNATURES_DIR, BUCKETS_DIR, CLUSTERS_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.dedup import (
        MinhashConfig, MinhashDedupSignature,
        MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter
    )
    try:
        from datatrove.executor.local import LocalPipelineExecutor
    except ImportError:
        from datatrove.executor.local import LocalExecutor as LocalPipelineExecutor

    minhash_config = smart_instantiate(
        MinhashConfig, num_buckets=20, hashes_per_bucket=13,
        num_perms=260, num_hashes=260, n_grams=5
    )

    gopher_filter = smart_instantiate(
        GopherQualityFilter,
        min_doc_words=50, min_words=50, min_word_count=50,
        max_doc_words=100000, max_words=100000, max_word_count=100000,
        max_symbol_to_word_ratio=0.3, max_symbol_word_ratio=0.3,
        max_ellipsis_to_word_ratio=0.3, max_ellipsis_word_ratio=0.3
    )
    lang_filter = smart_instantiate(
        LanguageFilter, languages=["zh", "en"], target_languages=["zh", "en"],
        language_threshold=0.7, threshold=0.7, min_prob=0.7
    )

    # Pipeline 1: 质量过滤 + MinHash签名
    print(f"\n[清洗] {subset_key} [1/4] 质量过滤+签名生成...")
    reader_1 = create_safe_reader(raw_dir)
    sig_step = smart_instantiate(
        MinhashDedupSignature, config=minhash_config,
        output_folder=SIGNATURES_DIR, folder=SIGNATURES_DIR
    )
    LocalPipelineExecutor(
        pipeline=[reader_1, gopher_filter, lang_filter, sig_step],
        tasks=workers, logging_dir=f"{logs_dir}/log_1"
    ).run()

    # Pipeline 2: Buckets
    print(f"[清洗] {subset_key} [2/4] 生成Buckets...")
    bucket_step = smart_instantiate(
        MinhashDedupBuckets, config=minhash_config,
        input_folder=SIGNATURES_DIR, output_folder=BUCKETS_DIR
    )
    buckets_tasks = getattr(minhash_config, 'num_buckets', workers)
    LocalPipelineExecutor(
        pipeline=[bucket_step], tasks=buckets_tasks,
        logging_dir=f"{logs_dir}/log_2"
    ).run()

    # Pipeline 3: Clusters (必须tasks=1)
    print(f"[清洗] {subset_key} [3/4] 生成Clusters...")
    cluster_step = smart_instantiate(
        MinhashDedupCluster, config=minhash_config,
        input_folder=BUCKETS_DIR, output_folder=CLUSTERS_DIR
    )
    LocalPipelineExecutor(
        pipeline=[cluster_step], tasks=1,
        logging_dir=f"{logs_dir}/log_3"
    ).run()

    # Pipeline 4: 去重过滤 + JSONL输出
    print(f"[清洗] {subset_key} [4/4] 去重+输出JSONL...")
    reader_4 = create_safe_reader(raw_dir)
    dedup_filter = smart_instantiate(
        MinhashDedupFilter, config=minhash_config,
        input_folder=CLUSTERS_DIR, clusters_folder=CLUSTERS_DIR, folder=CLUSTERS_DIR
    )
    writer_4 = smart_instantiate(
        JsonlWriter, output_folder=final_dir, folder=final_dir,
        output_filename=f"{subset_key}_shard_${{rank}}.jsonl",
        filename=f"{subset_key}_shard_${{rank}}.jsonl"
    )
    LocalPipelineExecutor(
        pipeline=[reader_4, dedup_filter, writer_4],
        tasks=workers, logging_dir=f"{logs_dir}/log_4"
    ).run()

    # ✅ 统计该子集最终产出
    total_rows, total_chars = 0, 0
    for f in Path(final_dir).glob("*.jsonl*"):
        opener = gzip.open if f.suffix == '.gz' else open
        mode = 'rt' if f.suffix == '.gz' else 'r'
        with opener(f, mode, encoding='utf-8') as fp:
            for line in fp:
                data = json.loads(line)
                total_rows += 1
                total_chars += len(data.get("text", ""))

    info = {
        "subset_key": subset_key,
        "total_rows": total_rows,
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 3,  # 粗略估算
        "output_dir": final_dir
    }
    print(f"✅ {subset_key} 清洗完成: {total_rows} 行, ~{info['estimated_tokens']/1e6:.1f}M tokens")
    return info


# ==========================================
# ✅ 新增：生成全局 Manifest
# ==========================================
def generate_manifest(results: List[Dict]):
    manifest = {r["subset_key"]: r for r in results if r is not None}
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n📋 Manifest 已生成: {MANIFEST_PATH}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


# ==========================================
# 沙盒测试 (适配新结构)
# ==========================================
def run_sandbox_test():
    print("\n" + "="*50)
    print("🚀 沙盒冒烟测试 (按子集隔离版)")
    print("="*50)
    SANDBOX_DIR = Path("/tmp/datatrove_sandbox_v2")
    if SANDBOX_DIR.exists(): shutil.rmtree(SANDBOX_DIR)

    test_subsets = ["test_zh_qa", "test_en_math"]
    for sk in test_subsets:
        raw = SANDBOX_DIR / "01_raw" / sk
        raw.mkdir(parents=True)
        df = pd.DataFrame({
            "text": [f"这是{sk}的测试文本{i}。" * 20 for i in range(10)],
            "source": [sk] * 10
        })
        df.to_parquet(raw / "shard_00000.parquet")

    # 覆盖全局路径
    global BASE_RAW_DIR, BASE_FINAL_DIR, BASE_LOGS_DIR, MANIFEST_PATH
    BASE_RAW_DIR = str(SANDBOX_DIR / "01_raw")
    BASE_FINAL_DIR = str(SANDBOX_DIR / "02_final")
    BASE_LOGS_DIR = str(SANDBOX_DIR / "03_logs")
    MANIFEST_PATH = str(SANDBOX_DIR / "manifest.json")

    results = []
    for sk in test_subsets:
        info = clean_and_dedup_subset(sk, workers=1)
        results.append(info)

    generate_manifest(results)

    # 验证
    for sk in test_subsets:
        out_dir = Path(BASE_FINAL_DIR) / sk
        files = list(out_dir.glob("*.jsonl*"))
        print(f"  {sk}: {len(files)} 个输出文件")
        assert len(files) > 0, f"❌ {sk} 无输出!"
    print("\n🎉 沙盒测试通过!")


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="运行沙盒测试")
    parser.add_argument("--workers", type=int, default=128, help="清洗并发数")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，仅清洗")
    args = parser.parse_args()

    if args.test:
        run_sandbox_test()
    else:
        print(f"📁 缓存根目录: {CUSTOM_CACHE_DIR}")

        # 阶段1+2: 按子集独立下载
        #if not args.skip_download:
        #    for cfg in DATASETS:
        #        threshold = probe_dataset(cfg)
        #        streaming_download_and_filter(cfg, threshold)

        # ✅ 阶段3: 按子集独立清洗去重
        results = []
        for cfg in DATASETS:
            info = clean_and_dedup_subset(cfg.subset_key, workers=args.workers)
            results.append(info)

        # ✅ 生成全局清单
        generate_manifest(results)
        print(f"\n🎉 全部完成! 各子集数据位于: {BASE_FINAL_DIR}")
