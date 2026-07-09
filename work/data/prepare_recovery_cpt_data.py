#!/usr/bin/env python3
"""
多阶段治愈CPT数据构建脚本 v2
修复: .jsonl.gz 支持 + 内存友好的流式处理（不加载全量数据到内存）
原理: 解压→构建行偏移索引→预生成阶段索引序列→流式写出合并数据
"""
import gzip
import json
import os
import sys
import glob
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================================================================
# 用户配置（根据实际情况修改）
# ================================================================
DATA_SOURCES = {
    "zh_multi_style": {
        "dir": "/data2/.cache/huggingface/02_final_pretrain_jsonl_v3/zh_multi_style",
        "total_tokens": 1_262_845_335,
    },
    "zh_qa_synthetic": {
        "dir": "/data2/.cache/huggingface/02_final_pretrain_jsonl_v3/zh_qa_synthetic",
        "total_tokens": 2_717_365_395,
    },
    "en_multi_style": {
        "dir": "/data2/.cache/huggingface/02_final_pretrain_jsonl_v3/en_multi_style",
        "total_tokens": 6_131_516_938,
    },
    "en_qa_synthetic": {
        "dir": "/data2/.cache/huggingface/02_final_pretrain_jsonl_v3/en_qa_synthetic",
        "total_tokens": 12_855_220_351,
    },
    "open_web_math": {
        "dir": "/data2/.cache/huggingface/02_final_pretrain_jsonl_v3/open_web_math",
        "total_tokens": 16_191_103_130,
    },
}

PHASE_SCHEDULE = [
    {
        "name": "Phase-1a-路由稳定期",
        "step_ratio": 0.30,
        "ratios": {
            "zh_multi_style":  0.10,
            "zh_qa_synthetic": 0.05,
            "en_multi_style":  0.38,
            "en_qa_synthetic": 0.27,
            "open_web_math":   0.20,
        },
        "description": "多样自然文本为主，稳定MoE路由器",
    },
    {
        "name": "Phase-1b-全面恢复期",
        "step_ratio": 0.40,
        "ratios": {
            "zh_multi_style":  0.06,
            "zh_qa_synthetic": 0.08,
            "en_multi_style":  0.26,
            "en_qa_synthetic": 0.27,
            "open_web_math":   0.33,
        },
        "description": "均衡恢复全部能力，数学回升",
    },
    {
        "name": "Phase-1c-能力增强期",
        "step_ratio": 0.30,
        "ratios": {
            "zh_multi_style":  0.03,
            "zh_qa_synthetic": 0.08,
            "en_multi_style":  0.20,
            "en_qa_synthetic": 0.30,
            "open_web_math":   0.39,
        },
        "description": "强化推理和QA能力",
    },
]

TRAINING_CONFIG = {
    "seq_length": 8192,
    "batch_size_per_gpu": 2,
    "num_gpus": 8,
    "gradient_accumulation_steps": 128,
    "seed": 42,
}

TEMP_DIR = "/data2/tmp/cpt_preprocess_v2"
OUTPUT_PATH = "/data2/cpt/recovery_cpt_merged.jsonl"

# ================================================================
# 核心函数
# ================================================================

def find_gz_files(directory):
    """查找目录下所有 .jsonl.gz 文件（支持子目录递归）"""
    files = sorted(glob.glob(os.path.join(directory, "*.jsonl.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(directory, "**/*.jsonl.gz"), recursive=True))
    return files


def decompress_to_single(gz_files, output_path):
    """将多个 .jsonl.gz 解压并合并为单个 .jsonl 文件"""
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for gz_path in gz_files:
            with gzip.open(gz_path, 'rt', encoding='utf-8') as in_f:
                shutil.copyfileobj(in_f, out_f)


def build_offset_index(file_path):
    """
    构建行级字节偏移索引。
    返回 numpy array，offsets[i] = 第 i 行在文件中的起始字节位置。
    内存开销: 约 8 bytes/行 (int64)，6M行 ≈ 48MB
    """
    offsets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            offsets.append(f.tell())
            if not f.readline():
                break
    offsets.pop()  # 最后一个是 EOF 位置，移除
    return np.array(offsets, dtype=np.int64)


def extract_text(line_str):
    """从 jsonl 行中提取文本，兼容多种字段名"""
    try:
        data = json.loads(line_str)
        # 兼容常见字段名
        for key in ("text", "content", "response", "output", "completion"):
            if key in data and isinstance(data[key], str):
                return data[key]
        # 兜底：直接返回整行
        return line_str.strip()
    except json.JSONDecodeError:
        return line_str.strip()


# ================================================================
# 主流程
# ================================================================

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # ── Step 1: 解压 .jsonl.gz → 临时 .jsonl ──
    print("=" * 65)
    print("Step 1/5: 解压 .jsonl.gz 文件到临时目录")
    print("=" * 65)

    decomp_paths = {}   # {source_name: decompressed_file_path}
    for name, info in DATA_SOURCES.items():
        gz_files = find_gz_files(info["dir"])
        if not gz_files:
            print(f"  [ERROR] 未找到 .jsonl.gz: {info['dir']}")
            sys.exit(1)

        out_path = os.path.join(TEMP_DIR, f"{name}.jsonl")
        decompress_to_single(gz_files, out_path)

        size_gb = os.path.getsize(out_path) / (1024 ** 3)
        decomp_paths[name] = out_path
        print(f"  [{name}] {len(gz_files)} 个 .jsonl.gz → {size_gb:.1f} GB")

    # ── Step 2: 构建行偏移索引 ──
    print(f"\n{'=' * 65}")
    print("Step 2/5: 构建行偏移索引（支持 O(1) 随机行访问）")
    print("=" * 65)

    offset_indices = {}  # {name: np.array of byte offsets}
    line_counts = {}     # {name: total_line_count}

    for name, path in decomp_paths.items():
        offsets = build_offset_index(path)
        offset_indices[name] = offsets
        line_counts[name] = len(offsets)
        idx_mb = offsets.nbytes / (1024 ** 2)
        print(f"  [{name}] {line_counts[name]:>10,} 行  |  索引 {idx_mb:.0f} MB")

    total_lines = sum(line_counts.values())
    print(f"  {'合计':>10s} {total_lines:>10,} 行")

    # ── Step 3: 计算训练计划 ──
    cfg = TRAINING_CONFIG
    eff_batch = (cfg["batch_size_per_gpu"]
                 * cfg["num_gpus"]
                 * cfg["gradient_accumulation_steps"])
    tokens_per_step = eff_batch * cfg["seq_length"]
    total_tokens = sum(ds["total_tokens"] for ds in DATA_SOURCES.values())
    total_steps = max(1, round(total_tokens / tokens_per_step))
    total_samples = total_steps * eff_batch

    print(f"\n{'=' * 65}")
    print("Step 3/5: 训练计划计算")
    print("=" * 65)
    print(f"  总可用 Token:     {total_tokens / 1e9:.2f} B")
    print(f"  序列长度:         {cfg['seq_length']}")
    print(f"  有效批大小:       {eff_batch}  "
          f"({cfg['batch_size_per_gpu']}×{cfg['num_gpus']} GPUs×"
          f"{cfg['gradient_accumulation_steps']} grad_accum)")
    print(f"  每步 Token:       {tokens_per_step / 1e6:.2f} M")
    print(f"  总训练步数:       {total_steps:,}")
    print(f"  总样本数:         {total_samples:,}")

    # ── Step 4: 生成多阶段索引序列 ──
    print(f"\n{'=' * 65}")
    print("Step 4/5: 生成多阶段索引序列")
    print("=" * 65)

    source_names = list(DATA_SOURCES.keys())
    rng = np.random.RandomState(cfg["seed"])

    # 预分配索引数组（内存约 100MB for 34M samples）
    all_src_ids = np.empty(total_samples, dtype=np.int8)    # 5 个源, int8 足够
    all_line_ids = np.empty(total_samples, dtype=np.int64)

    pos = 0
    phase_boundaries = []

    for phase in PHASE_SCHEDULE:
        phase_steps = round(total_steps * phase["step_ratio"])
        phase_samples = phase_steps * eff_batch

        names = list(phase["ratios"].keys())
        weights = np.array([phase["ratios"][n] for n in names], dtype=np.float64)
        weights /= weights.sum()

        # 批量采样提升速度
        choices = rng.choice(len(names), size=phase_samples, p=weights)

        for i in range(phase_samples):
            chosen_name = names[choices[i]]
            all_src_ids[pos] = source_names.index(chosen_name)
            all_line_ids[pos] = rng.randint(0, line_counts[chosen_name])
            pos += 1

        start_step = (pos - phase_samples) // eff_batch
        end_step = pos // eff_batch

        phase_boundaries.append({
            "name": phase["name"],
            "start_step": start_step,
            "end_step": end_step,
            "ratios": {k: float(v) for k, v in phase["ratios"].items()},
            "description": phase.get("description", ""),
        })

        ratio_str = " / ".join(f"{k}={v:.0%}" for k, v in phase["ratios"].items())
        print(f"  [{phase['name']}]")
        print(f"    Step {start_step:,} ~ {end_step:,}  |  {phase_samples:,} 样本")
        print(f"    配比: {ratio_str}")

    # 局部窗口 shuffle（窗口 = 有效批大小，保持阶段内充分混合）
    print(f"\n  局部 shuffle（窗口={eff_batch}）...")
    window = eff_batch
    for start in range(0, total_samples, window):
        end = min(start + window, total_samples)
        perm = rng.permutation(end - start)
        all_src_ids[start:end] = all_src_ids[start:end][perm]
        all_line_ids[start:end] = all_line_ids[start:end][perm]

    idx_mem = (all_src_ids.nbytes + all_line_ids.nbytes) / (1024 ** 2)
    print(f"  索引数组内存: {idx_mem:.0f} MB")

    # ── Step 5: 流式写出合并数据 ──
    print(f"\n{'=' * 65}")
    print("Step 5/5: 流式写出合并数据")
    print("=" * 65)
    print(f"  输出文件: {OUTPUT_PATH}")

    # 打开所有解压后的文件句柄
    file_handles = {
        name: open(path, 'r', encoding='utf-8')
        for name, path in decomp_paths.items()
    }

    written = 0
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for i in tqdm(range(total_samples), desc="写出样本",
                       unit="样本", unit_scale=True):
            src_name = source_names[int(all_src_ids[i])]
            line_idx = int(all_line_ids[i])

            # 通过偏移索引 O(1) 读取指定行
            offset = offset_indices[src_name][line_idx]
            fh = file_handles[src_name]
            fh.seek(offset)
            line_str = fh.readline()

            text = extract_text(line_str)
            out_f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
            written += 1

    # 关闭所有文件句柄
    for fh in file_handles.values():
        fh.close()

    # ── 保存元信息 ──
    meta = {
        "total_samples": int(total_samples),
        "total_steps": int(total_steps),
        "effective_batch_size": int(eff_batch),
        "tokens_per_step": int(tokens_per_step),
        "total_tokens": int(total_steps * tokens_per_step),
        "phase_boundaries": phase_boundaries,
        "source_names": source_names,
        "line_counts": line_counts,
        "training_config": TRAINING_CONFIG,
        "seed": cfg["seed"],
        "output_path": OUTPUT_PATH,
    }
    meta_path = OUTPUT_PATH + ".meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ── 打印总结 ──
    out_size_gb = os.path.getsize(OUTPUT_PATH) / (1024 ** 3)
    print(f"\n{'=' * 65}")
    print("完成!")
    print(f"{'=' * 65}")
    print(f"  训练数据: {OUTPUT_PATH}  ({out_size_gb:.1f} GB, {written:,} 样本)")
    print(f"  元信息:   {meta_path}")
    print(f"  临时文件: {TEMP_DIR}/  (可手动清理: rm -rf {TEMP_DIR})")
    print(f"\n  阶段时间线:")
    for pb in phase_boundaries:
        ratio_str = " / ".join(f"{k}={v:.0%}" for k, v in pb["ratios"].items())
        print(f"    Step {pb['start_step']:>5,} ~ {pb['end_step']:>5,}  "
              f"| {pb['name']}")
        print(f"      {ratio_str}")
        print(f"      {pb['description']}")


if __name__ == "__main__":
    main()
