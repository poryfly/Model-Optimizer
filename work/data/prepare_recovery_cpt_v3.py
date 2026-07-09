#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recovery CPT 数据预处理脚本 v3
==============================
针对 DeepSeek-V4-Flash 4B-A1.5B MoE 裁剪模型的治愈预训练数据准备。

核心特性：
  1. 支持 manifest.json 中 output_dir 指向目录（内含多个 .jsonl.gz 分片文件）
  2. 字节偏移索引实现 O(1) 随机行读取，峰值内存 ~500MB（非全量加载）
  3. 三阶段配比数据预构建，单次训练自动经历多阶段
  4. 流式写出，避免内存溢出

用法：
  python prepare_recovery_cpt_v3.py \
    --manifest /path/to/manifest.json \
    --output_dir /data/cpt/recovery_cpt \
    --total_steps 100000 \
    --batch_size_per_step 256 \
    --seed 42

输出：
  {output_dir}/recovery_cpt_merged.jsonl       # 合并后的训练数据
  {output_dir}/recovery_cpt_merged.meta.json   # 阶段边界元信息
"""

import argparse
import bisect
import gzip
import json
import os
import random
import sys
import tempfile
from collections import OrderedDict
from glob import glob as glob_fn
from typing import Dict, List, Optional, Tuple

# ============================================================
# 第一部分：多文件数据源索引（核心修复）
# ============================================================

class MultiFileSourceIndex:
    """
    管理一个数据源（可包含多个 .jsonl 文件）的全局行索引。
    
    工作原理：
      - 维护 N 个文件的各自偏移索引
      - 通过 file_boundaries（累积行数）实现 global_line_id → (file_idx, local_offset) 映射
      - 读取时先二分查找文件，再 seek 读取，O(log N_files + O(1))
    """
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        # 每个文件：(jsonl_path, offset_list)
        self.files: List[Tuple[str, List[int]]] = []
        # 累积行数边界，长度 = len(files) + 1
        # file_boundaries[i] 表示第 i 个文件的起始 global line id
        self.file_boundaries: List[int] = [0]
        self.total_lines: int = 0
    
    def add_file(self, jsonl_path: str, offsets: List[int]):
        """添加一个文件的偏移索引"""
        self.files.append((jsonl_path, offsets))
        self.total_lines += len(offsets)
        self.file_boundaries.append(self.total_lines)
    
    def get_line(self, global_line_id: int) -> str:
        """
        根据 global line id 读取一行。
        使用 bisect 在 file_boundaries 中二分查找对应文件，然后 O(1) seek 读取。
        """
        if global_line_id < 0 or global_line_id >= self.total_lines:
            raise IndexError(
                f"{self.source_name}: global_line_id={global_line_id} "
                f"out of range [0, {self.total_lines})"
            )
        
        # 二分查找：找到 global_line_id 落在哪个文件
        file_idx = bisect.bisect_right(self.file_boundaries, global_line_id) - 1
        local_line_id = global_line_id - self.file_boundaries[file_idx]
        
        jsonl_path, offsets = self.files[file_idx]
        offset = offsets[local_line_id]
        
        with open(jsonl_path, 'rb') as f:
            f.seek(offset)
            line = f.readline()
        
        return line.decode('utf-8', errors='replace').strip()
    
    def get_indices_list(self) -> List[int]:
        """返回 0..total_lines-1 的索引列表"""
        return list(range(self.total_lines))
    
    def get_file_stats(self) -> List[Tuple[str, int]]:
        """返回每个文件的路径和行数"""
        stats = []
        for i, (path, offsets) in enumerate(self.files):
            stats.append((path, len(offsets)))
        return stats


def decompress_gz_file(gz_path: str, tmp_dir: str) -> str:
    """
    将单个 .jsonl.gz 解压到临时目录。
    解压后的文件名 = 源文件名去掉 .gz 后缀。
    如果已解压且非空则跳过。
    """
    gz_path = os.path.abspath(gz_path)
    # 保留目录结构避免重名
    rel_name = os.path.basename(gz_path).replace('.gz', '')
    out_path = os.path.join(tmp_dir, rel_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    print(f"    解压: {os.path.basename(gz_path)} -> {out_path}")
    bytes_written = 0
    with gzip.open(gz_path, 'rb') as fin, open(out_path, 'wb') as fout:
        while True:
            chunk = fin.read(64 * 1024 * 1024)  # 64MB chunks
            if not chunk:
                break
            fout.write(chunk)
            bytes_written += len(chunk)

    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"    完成: {out_path} ({size_gb:.2f} GB)")
    return out_path


def build_offset_index(jsonl_path: str) -> List[int]:
    """
    构建单文件的字节偏移索引。
    扫描文件，记录每一行的起始字节位置。
    
    内存：每行 8 bytes (int64)。例如 5000 万行 ≈ 400MB。
    """
    offsets = []
    with open(jsonl_path, 'rb') as f:
        offsets.append(f.tell())  # 第一行从 0 开始
        while True:
            line = f.readline()
            if not line:
                break
            if line.endswith(b'\n'):
                offsets.append(f.tell())

    # 最后一个元素可能是 EOF 位置（非行起始），需剔除
    if len(offsets) > 0:
        file_size = os.path.getsize(jsonl_path)
        if offsets[-1] == file_size:
            offsets.pop()

    return offsets


def discover_gz_files(directory: str) -> List[str]:
    """
    扫描目录下所有 .jsonl.gz 文件，按文件名排序返回。
    """
    if not os.path.isdir(directory):
        return []
    
    pattern = os.path.join(directory, '*.jsonl.gz')
    files = sorted(glob_fn(pattern))
    return files


# ============================================================
# 第二部分：数据源配置与训练计划
# ============================================================

DATA_SOURCE_CONFIG = OrderedDict({
    'zh_multi_style': {
        'manifest_key': 'zh_multi_style',
        'total_tokens': 1.26e9,
        'desc': '中文多风格语料',
    },
    'zh_qa_synthetic': {
        'manifest_key': 'zh_qa_synthetic',
        'total_tokens': 2.72e9,
        'desc': '中文QA合成语料',
    },
    'en_multi_style': {
        'manifest_key': 'en_multi_style',
        'total_tokens': 6.13e9,
        'desc': '英文多风格语料',
    },
    'en_qa_synthetic': {
        'manifest_key': 'en_qa_synthetic',
        'total_tokens': 12.86e9,
        'desc': '英文QA合成语料',
    },
    'open_web_math': {
        'manifest_key': 'open_web_math',
        'total_tokens': 16.19e9,
        'desc': '数学推理语料',
    },
})


def compute_phase_plan(total_steps: int, samples_per_step: int) -> Dict:
    """
    计算三阶段训练计划。

    三阶段设计：
      Phase-1a (0%  ~ 30%): 路由稳定期 — math 压至 20%，multi_style 放大至 48%
      Phase-1b (30% ~ 70%): 全面恢复期 — math 回升至 33%，全品类均衡
      Phase-1c (70% ~100%): 能力增强期 — math 提升至 39%，QA 增强至 38%
    """
    phase_a_end = int(total_steps * 0.30)
    phase_b_end = int(total_steps * 0.70)

    phase_ratios = {
        'phase_1a': {
            'zh_multi_style': 0.13,
            'zh_qa_synthetic': 0.13,
            'en_multi_style': 0.35,
            'en_qa_synthetic': 0.19,
            'open_web_math': 0.20,
        },
        'phase_1b': {
            'zh_multi_style': 0.10,
            'zh_qa_synthetic': 0.15,
            'en_multi_style': 0.22,
            'en_qa_synthetic': 0.20,
            'open_web_math': 0.33,
        },
        'phase_1c': {
            'zh_multi_style': 0.06,
            'zh_qa_synthetic': 0.07,
            'en_multi_style': 0.10,
            'en_qa_synthetic': 0.38,
            'open_web_math': 0.39,
        },
    }

    plan = {
        'total_steps': total_steps,
        'samples_per_step': samples_per_step,
        'total_samples': total_steps * samples_per_step,
        'phases': OrderedDict(),
    }

    phases = [
        ('phase_1a', 0, phase_a_end),
        ('phase_1b', phase_a_end, phase_b_end),
        ('phase_1c', phase_b_end, total_steps),
    ]

    for phase_name, step_start, step_end in phases:
        num_steps = step_end - step_start
        if num_steps <= 0:
            continue
        num_samples = num_steps * samples_per_step
        ratios = phase_ratios[phase_name]

        plan['phases'][phase_name] = {
            'step_start': step_start,
            'step_end': step_end,
            'num_steps': num_steps,
            'num_samples': num_samples,
            'ratios': ratios,
        }

        ratio_sum = sum(ratios.values())
        assert abs(ratio_sum - 1.0) < 1e-6, \
            f"{phase_name} ratios sum to {ratio_sum}, expected 1.0"

    return plan


# ============================================================
# 第三部分：多阶段索引序列生成
# ============================================================

def generate_global_index_sequence(
    source_indices: Dict[str, List[int]],
    plan: Dict,
    rng: random.Random,
) -> Tuple[List[str], List[int]]:
    """
    预生成全局样本序列。
    
    返回：
      all_src_ids:  每个样本对应的数据源名称
      all_line_ids: 每个样本在对应数据源中的 global line id
    
    原理：将"阶段切换"从训练时决策转移到数据构建时决策。
    jsonl 文件的前 30% 对应 Phase-1a 配比，
    中间 40% 对应 Phase-1b，
    后 30% 对应 Phase-1c。
    """
    all_src_ids = []
    all_line_ids = []

    for phase_name, phase_info in plan['phases'].items():
        num_samples = phase_info['num_samples']
        ratios = phase_info['ratios']
        print(f"  生成 {phase_name}: {num_samples:,} 样本")

        source_samples = {}
        for src_name, ratio in ratios.items():
            n = int(num_samples * ratio)
            source_samples[src_name] = n

        # 修正余数
        diff = num_samples - sum(source_samples.values())
        if diff != 0:
            max_src = max(ratios, key=ratios.get)
            source_samples[max_src] += diff

        for src_name, n_samples in source_samples.items():
            if n_samples <= 0:
                continue

            available = source_indices[src_name]
            n_available = len(available)

            if n_samples <= n_available:
                sampled = rng.sample(available, n_samples)
            else:
                sampled = list(available)
                rng.shuffle(sampled)
                remaining = n_samples - n_available
                extra = rng.choices(available, k=remaining)
                sampled.extend(extra)

            all_src_ids.extend([src_name] * n_samples)
            all_line_ids.extend(sampled)

    # 阶段内 shuffle（保持阶段边界不变）
    offset = 0
    for phase_name, phase_info in plan['phases'].items():
        n = phase_info['num_samples']
        phase_pairs = list(zip(
            all_src_ids[offset:offset + n],
            all_line_ids[offset:offset + n],
        ))
        rng.shuffle(phase_pairs)
        all_src_ids[offset:offset + n] = [p[0] for p in phase_pairs]
        all_line_ids[offset:offset + n] = [p[1] for p in phase_pairs]
        offset += n

    assert len(all_src_ids) == len(all_line_ids)
    print(f"  全局索引序列: {len(all_src_ids):,} 样本")

    return all_src_ids, all_line_ids


# ============================================================
# 第四部分：流式写出
# ============================================================

def stream_write_merged(
    output_path: str,
    source_indexes: Dict[str, MultiFileSourceIndex],
    all_src_ids: List[str],
    all_line_ids: List[int],
    log_interval: int = 100000,
):
    """
    流式写出合并 jsonl。通过 MultiFileSourceIndex.get_line() O(1) 读取。
    """
    total = len(all_src_ids)
    print(f"  流式写出 {total:,} 样本 -> {output_path}")

    written = 0
    skipped = 0
    with open(output_path, 'w', encoding='utf-8') as fout:
        for i in range(total):
            src_name = all_src_ids[i]
            global_line_id = all_line_ids[i]

            try:
                line = source_indexes[src_name].get_line(global_line_id)
            except Exception as e:
                print(f"  [警告] 读取失败: src={src_name}, "
                      f"line_id={global_line_id}, err={e}")
                skipped += 1
                continue

            if not line:
                skipped += 1
                continue

            fout.write(line + '\n')
            written += 1

            if (i + 1) % log_interval == 0:
                pct = (i + 1) / total * 100
                print(f"    进度: {i + 1:,}/{total:,} ({pct:.1f}%) "
                      f"写出: {written:,} 跳过: {skipped:,}")

    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"  写出完成: {output_path} ({size_gb:.2f} GB), "
          f"写出 {written:,} 行, 跳过 {skipped:,} 行")


# ============================================================
# 第五部分：元信息文件
# ============================================================

def write_meta_json(
    output_path: str,
    plan: Dict,
    source_indexes: Dict[str, MultiFileSourceIndex],
):
    """写出 .meta.json"""
    meta = {
        'version': 'v3',
        'description': 'Recovery CPT multi-phase training data (multi-file source)',
        'total_sources': len(source_indexes),
        'sources': {},
        'phases': {},
        'total_samples': plan['total_samples'],
        'total_steps': plan['total_steps'],
        'samples_per_step': plan['samples_per_step'],
    }

    for src_name, idx in source_indexes.items():
        meta['sources'][src_name] = {
            'total_lines': idx.total_lines,
            'num_files': len(idx.files),
            'file_stats': [
                {'path': os.path.basename(p), 'lines': n}
                for p, n in idx.get_file_stats()
            ],
            'total_tokens': DATA_SOURCE_CONFIG[src_name]['total_tokens'],
            'desc': DATA_SOURCE_CONFIG[src_name]['desc'],
        }

    cumulative = 0
    for phase_name, phase_info in plan['phases'].items():
        meta['phases'][phase_name] = {
            'sample_start': cumulative,
            'sample_end': cumulative + phase_info['num_samples'],
            'step_start': phase_info['step_start'],
            'step_end': phase_info['step_end'],
            'num_samples': phase_info['num_samples'],
            'ratios': phase_info['ratios'],
        }
        cumulative += phase_info['num_samples']

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  元信息: {output_path}")


# ============================================================
# 第六部分：Manifest 解析（修复核心）
# ============================================================

def resolve_source_directory(
    manifest_entry: dict,
    manifest_key: str,
    data_root: Optional[str],
) -> str:
    """
    从 manifest entry 中解析出数据源目录路径。
    
    支持多种 manifest 结构：
    1. {"output_dir": "/data/xxx/"} — 直接给出目录
    2. {"files": ["a.jsonl.gz", "b.jsonl.gz"]} — 文件列表
    3. {"path": "/data/xxx/", ...} — path 字段
    4. 兜底：遍历所有字符串值，找到指向目录的字段
    """
    # 优先检查 output_dir
    for key in ['output_dir', 'data_dir', 'dir', 'directory', 'path']:
        if key in manifest_entry:
            val = manifest_entry[key]
            if isinstance(val, str):
                # 处理相对路径
                if data_root and not os.path.isabs(val):
                    val = os.path.join(data_root, val)
                if os.path.isdir(val):
                    return val
    
    # 检查 files 列表 —— 取第一个文件的目录
    if 'files' in manifest_entry and isinstance(manifest_entry['files'], list):
        first_file = manifest_entry['files'][0]
        if isinstance(first_file, str):
            if data_root and not os.path.isabs(first_file):
                first_file = os.path.join(data_root, first_file)
            parent = os.path.dirname(first_file)
            if os.path.isdir(parent):
                return parent
    
    # 兜底：遍历所有值，找第一个指向已存在目录的字符串
    for v in manifest_entry.values():
        if isinstance(v, str):
            if data_root and not os.path.isabs(v):
                v = os.path.join(data_root, v)
            if os.path.isdir(v):
                return v
    
    # 最终兜底：遍历所有值，找包含 .jsonl.gz 的目录
    for v in manifest_entry.values():
        if isinstance(v, str):
            if '.jsonl.gz' in v or '.jsonl' in v:
                # 提取目录部分
                if data_root and not os.path.isabs(v):
                    v = os.path.join(data_root, v)
                # 如果 v 本身是文件，取其目录
                candidate = v if os.path.isdir(v) else os.path.dirname(v)
                if os.path.isdir(candidate):
                    return candidate
    
    return None


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Recovery CPT 数据预处理 v3')
    parser.add_argument('--manifest', type=str, required=True,
                        help='manifest.json 路径')
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据文件根目录（manifest 中相对路径的前缀）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--total_steps', type=int, default=100000,
                        help='总训练步数')
    parser.add_argument('--batch_size_per_step', type=int, default=256,
                        help='每步样本数 = batch_size * gradient_accumulation_steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--tmp_dir', type=str, default=None,
                        help='临时解压目录（默认自动创建）')
    parser.add_argument('--keep_temp', action='store_true',
                        help='保留临时解压文件（调试用）')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, 'recovery_cpt_merged.jsonl')
    output_meta = os.path.join(args.output_dir, 'recovery_cpt_merged.meta.json')

    # 临时目录：为每个数据源创建子目录，避免解压后文件重名
    if args.tmp_dir:
        tmp_base = args.tmp_dir
        os.makedirs(tmp_base, exist_ok=True)
        auto_cleanup = False
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix='recovery_cpt_')
        tmp_base = tmp_obj.name
        auto_cleanup = True

    try:
        # ================================================================
        # Step 1: 读取 manifest，解析每个数据源的目录
        # ================================================================
        print("=" * 60)
        print("Step 1: 读取 manifest 并定位数据源目录")
        print("=" * 60)

        with open(args.manifest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        if isinstance(manifest, list):
            manifest_dict = {
                item.get('name', item.get('key', str(i))): item
                for i, item in enumerate(manifest)
            }
        else:
            manifest_dict = manifest

        # 解析每个数据源的数据目录
        source_dirs = {}  # src_name -> directory path
        for src_name, src_config in DATA_SOURCE_CONFIG.items():
            manifest_key = src_config['manifest_key']
            if manifest_key not in manifest_dict:
                print(f"  [错误] manifest 中未找到 key: {manifest_key}")
                print(f"  可用 key: {list(manifest_dict.keys())}")
                sys.exit(1)

            entry = manifest_dict[manifest_key]
            data_dir = resolve_source_directory(entry, manifest_key, args.data_root)

            if data_dir is None:
                print(f"  [错误] 无法解析 {manifest_key} 的数据目录")
                print(f"  entry 内容:")
                print(f"    {json.dumps(entry, indent=2, ensure_ascii=False)[:1000]}")
                sys.exit(1)

            if not os.path.isdir(data_dir):
                print(f"  [错误] 目录不存在: {data_dir}")
                sys.exit(1)

            source_dirs[src_name] = data_dir

            # 扫描目录下的 .jsonl.gz 文件
            gz_files = discover_gz_files(data_dir)
            print(f"  {src_name}: {data_dir}")
            print(f"    -> 发现 {len(gz_files)} 个 .jsonl.gz 文件")

            if len(gz_files) == 0:
                print(f"  [错误] 目录下无 .jsonl.gz 文件: {data_dir}")
                sys.exit(1)

            # 列出前 5 个和最后 1 个文件（避免刷屏）
            for f in gz_files[:5]:
                size_mb = os.path.getsize(f) / (1024**2)
                print(f"    - {os.path.basename(f)} ({size_mb:.1f} MB)")
            if len(gz_files) > 6:
                print(f"    ... 省略 {len(gz_files) - 6} 个文件 ...")
                f = gz_files[-1]
                size_mb = os.path.getsize(f) / (1024**2)
                print(f"    - {os.path.basename(f)} ({size_mb:.1f} MB)")

        # ================================================================
        # Step 2: 解压所有文件并构建多文件偏移索引
        # ================================================================
        print()
        print("=" * 60)
        print("Step 2: 解压源文件并构建多文件偏移索引")
        print("=" * 60)

        source_indexes = {}  # src_name -> MultiFileSourceIndex

        for src_name, data_dir in source_dirs.items():
            print(f"\n  处理数据源: {src_name}")

            gz_files = discover_gz_files(data_dir)
            src_index = MultiFileSourceIndex(src_name)

            # 每个数据源用自己的子目录解压，避免文件重名
            src_tmp_dir = os.path.join(tmp_base, src_name)
            os.makedirs(src_tmp_dir, exist_ok=True)

            for fi, gz_path in enumerate(gz_files):
                # 2a: 解压
                jsonl_path = decompress_gz_file(gz_path, src_tmp_dir)

                # 2b: 构建偏移索引
                print(f"    构建索引 [{fi+1}/{len(gz_files)}]: "
                      f"{os.path.basename(gz_path)} ...")
                offsets = build_offset_index(jsonl_path)
                
                index_mb = len(offsets) * 8 / (1024**2)
                print(f"    索引完成: {len(offsets):,} 行, "
                      f"索引大小 {index_mb:.1f} MB")

                # 2c: 注册到 MultiFileSourceIndex
                src_index.add_file(jsonl_path, offsets)

            source_indexes[src_name] = src_index
            print(f"    {src_name} 汇总: "
                  f"{len(gz_files)} 个文件, "
                  f"{src_index.total_lines:,} 行")

        total_lines = sum(idx.total_lines for idx in source_indexes.values())
        total_index_mb = sum(
            sum(len(offsets) for _, offsets in idx.files)
            for idx in source_indexes.values()
        ) * 8 / (1024**2)
        print(f"\n  所有数据源汇总:")
        print(f"    总文件数: {sum(len(idx.files) for idx in source_indexes.values())}")
        print(f"    总行数:   {total_lines:,}")
        print(f"    索引内存: {total_index_mb:.1f} MB")

        # ================================================================
        # Step 3: 计算训练计划
        # ================================================================
        print()
        print("=" * 60)
        print("Step 3: 计算三阶段训练计划")
        print("=" * 60)

        plan = compute_phase_plan(args.total_steps, args.batch_size_per_step)
        print(f"  总步数:     {plan['total_steps']:,}")
        print(f"  每步样本数: {plan['samples_per_step']}")
        print(f"  总样本数:   {plan['total_samples']:,}")

        for phase_name, phase_info in plan['phases'].items():
            ratios_str = ', '.join(
                f"{k}={v:.0%}" for k, v in phase_info['ratios'].items()
            )
            print(f"\n  {phase_name} "
                  f"(step {phase_info['step_start']}-{phase_info['step_end']}, "
                  f"{phase_info['num_samples']:,} 样本):")
            print(f"    配比: {ratios_str}")

        # ================================================================
        # Step 4: 生成多阶段全局索引序列
        # ================================================================
        print()
        print("=" * 60)
        print("Step 4: 生成多阶段全局索引序列")
        print("=" * 60)

        source_indices = {}
        for src_name in DATA_SOURCE_CONFIG:
            indices = source_indexes[src_name].get_indices_list()
            rng.shuffle(indices)
            source_indices[src_name] = indices
            print(f"  {src_name}: {len(indices):,} 行 (已打散)")

        all_src_ids, all_line_ids = generate_global_index_sequence(
            source_indices, plan, rng
        )

        # 验证各阶段分布
        print()
        print("  [验证] 各阶段数据源实际分布:")
        offset = 0
        for phase_name, phase_info in plan['phases'].items():
            n = phase_info['num_samples']
            phase_srcs = all_src_ids[offset:offset + n]
            dist = {}
            for s in phase_srcs:
                dist[s] = dist.get(s, 0) + 1
            dist_str = ', '.join(
                f"{k}={v/n:.1%}" for k, v in sorted(dist.items())
            )
            print(f"    {phase_name}: {dist_str}")
            offset += n

        # ================================================================
        # Step 5: 流式写出
        # ================================================================
        print()
        print("=" * 60)
        print("Step 5: 流式写出合并 jsonl")
        print("=" * 60)

        stream_write_merged(
            output_path=output_jsonl,
            source_indexes=source_indexes,
            all_src_ids=all_src_ids,
            all_line_ids=all_line_ids,
            log_interval=100000,
        )

        # ================================================================
        # Step 6: 元信息
        # ================================================================
        print()
        print("=" * 60)
        print("Step 6: 写出元信息")
        print("=" * 60)

        write_meta_json(output_meta, plan, source_indexes)

        # ================================================================
        # 完成
        # ================================================================
        print()
        print("=" * 60)
        print("全部完成!")
        print("=" * 60)
        output_size_gb = os.path.getsize(output_jsonl) / (1024**3)
        print(f"  输出文件: {output_jsonl} ({output_size_gb:.2f} GB)")
        print(f"  元信息:   {output_meta}")
        print()
        print("  启动训练命令:")
        print("  " + "\n  ".join([
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\",
            "swift sft \\",
            "  --model /path/to/pruned_deepseek_v4_flash_4b_a1.5b \\",
            "  --model_type deepseek-v4 \\",
            "  --train_type full \\",
            f"  --dataset {output_jsonl} \\",
            "  --template_type completion \\",
            "  --max_length 8192 \\",
            "  --learning_rate 1.8e-4 \\",
            "  --warmup_ratio 0.04 \\",
            "  --lr_scheduler_type cosine \\",
            "  --min_lr_ratio 0.04 \\",
            "  --optimizer adamw_torch \\",
            "  --weight_decay 0.1 \\",
            "  --adam_beta2 0.95 \\",
            "  --max_grad_norm 1.0 \\",
            "  --batch_size 2 \\",
            "  --gradient_accumulation_steps 128 \\",
            "  --num_train_epochs 1 \\",
            "  --deepspeed zero3 \\",
            "  --bf16 true \\",
            "  --gradient_checkpointing true \\",
            "  --output_dir /path/to/output/recovery_cpt_v1 \\",
            "  --save_steps 200 \\",
            "  --save_total_limit 5 \\",
            "  --logging_steps 10 \\",
            "  --seed 42",
        ]))

    finally:
        if not args.keep_temp and auto_cleanup:
            print(f"\n  清理临时目录: {tmp_base}")
            tmp_obj.cleanup()
        elif args.keep_temp:
            print(f"\n  保留临时目录 (--keep_temp): {tmp_base}")


if __name__ == '__main__':
    main()