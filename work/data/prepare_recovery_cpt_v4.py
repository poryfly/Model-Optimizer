#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recovery CPT 数据预处理脚本 v4
==============================
针对 DeepSeek-V4-Flash 4B-A1.5B MoE 裁剪模型的治愈预训练数据准备。
在 v3 基础上新增 seq_len 课程学习支持。

核心特性：
  1. 支持 manifest.json 中 output_dir 指向目录（内含多个 .jsonl.gz 分片文件）
  2. 字节偏移索引实现 O(1) 随机行读取，峰值内存 ~500MB（非全量加载）
  3. 三阶段配比数据预构建，单次训练自动经历多阶段
  4. 流式写出，避免内存溢出
  5. [NEW] seq_len 课程学习：按字符数分桶，生成 5 份对应 2k/4k/8k/16k/32k 的独立数据集
  6. [NEW] 样本不足时按最近邻桶填充（训练时截断）

seq_len 课程学习原理：
  MoE 模型（如 DeepSeek-V4）稀疏注意力收敛策略：
  2k → 4k → 8k → 16k → 32k，先让路由器在短序列下稳定，再逐步扩展上下文。
  每个 seq_len 阶段内部保持三阶段数据配比（Phase-1a/1b/1c），确保数学/语言能力均衡恢复。
  各 seq_len 文件步数均等（total_steps // 5），训练脚本按顺序加载。

用法：
  python prepare_recovery_cpt_v4.py \\
    --manifest /path/to/manifest.json \\
    --data_root /path/to/data_root/ \\
    --output_dir /data/cpt/recovery_cpt_v4 \\
    --total_steps 100000 \\
    --batch_size_per_step 256 \\
    --seed 42

输出：
  {output_dir}/seq2k/recovery_cpt_seq2k.jsonl
  {output_dir}/seq2k/recovery_cpt_seq2k.meta.json
  {output_dir}/seq4k/recovery_cpt_seq4k.jsonl
  {output_dir}/seq4k/recovery_cpt_seq4k.meta.json
  {output_dir}/seq8k/recovery_cpt_seq8k.jsonl
  {output_dir}/seq8k/recovery_cpt_seq8k.meta.json
  {output_dir}/seq16k/recovery_cpt_seq16k.jsonl
  {output_dir}/seq16k/recovery_cpt_seq16k.meta.json
  {output_dir}/seq32k/recovery_cpt_seq32k.jsonl
  {output_dir}/seq32k/recovery_cpt_seq32k.meta.json
  {output_dir}/summary.json
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
# 第零部分：seq_len 课程学习配置
# ============================================================

# seq_len 分桶配置（按字符数估算 token 数）
# 字符数阈值 = max_tokens * CHARS_PER_TOKEN
# 中文/英文混合语料，平均每个 token 约 2 个字符
SEQ_LEN_CONFIGS = [
    {'name': 'seq2k',  'max_chars': 4096,   'max_tokens': 2048},
    {'name': 'seq4k',  'max_chars': 8192,   'max_tokens': 4096},
    {'name': 'seq8k',  'max_chars': 16384,  'max_tokens': 8192},
    {'name': 'seq16k', 'max_chars': 32768,  'max_tokens': 16384},
    {'name': 'seq32k', 'max_chars': None,   'max_tokens': 32768},  # 无上界
]

# 每个 seq_len 桶占总步数的比例（均分）
# 总步数 = total_steps，每个桶 = total_steps // 5
NUM_SEQ_BUCKETS = len(SEQ_LEN_CONFIGS)

# 文本字段优先级（用于估算字符数）
TEXT_FIELDS_DEFAULT = ('text', 'content', 'input', 'document', 'passage')

# ============================================================
# 第一部分：多文件数据源索引（复用自 v3）
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
        self.files: List[Tuple[str, List[int]]] = []
        self.file_boundaries: List[int] = [0]
        self.total_lines: int = 0

    def add_file(self, jsonl_path: str, offsets: List[int]):
        """添加一个文件的偏移索引"""
        self.files.append((jsonl_path, offsets))
        self.total_lines += len(offsets)
        self.file_boundaries.append(self.total_lines)

    def get_line(self, global_line_id: int) -> str:
        """根据 global line id 读取一行"""
        if global_line_id < 0 or global_line_id >= self.total_lines:
            raise IndexError(
                f"{self.source_name}: global_line_id={global_line_id} "
                f"out of range [0, {self.total_lines})"
            )
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
        return [(path, len(offsets)) for path, offsets in self.files]


def decompress_gz_file(gz_path: str, tmp_dir: str) -> str:
    """将单个 .jsonl.gz 解压到临时目录。如果已解压且非空则跳过。"""
    gz_path = os.path.abspath(gz_path)
    rel_name = os.path.basename(gz_path).replace('.gz', '')
    out_path = os.path.join(tmp_dir, rel_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    print(f"    解压: {os.path.basename(gz_path)} -> {out_path}")
    with gzip.open(gz_path, 'rb') as fin, open(out_path, 'wb') as fout:
        while True:
            chunk = fin.read(64 * 1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)

    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"    完成: {out_path} ({size_gb:.2f} GB)")
    return out_path


def build_offset_index(jsonl_path: str) -> List[int]:
    """构建单文件的字节偏移索引"""
    offsets = []
    with open(jsonl_path, 'rb') as f:
        offsets.append(f.tell())
        while True:
            line = f.readline()
            if not line:
                break
            if line.endswith(b'\n'):
                offsets.append(f.tell())
    if len(offsets) > 0:
        file_size = os.path.getsize(jsonl_path)
        if offsets[-1] == file_size:
            offsets.pop()
    return offsets


def discover_gz_files(directory: str) -> List[str]:
    """扫描目录下所有 .jsonl.gz 文件，按文件名排序返回"""
    if not os.path.isdir(directory):
        return []
    pattern = os.path.join(directory, '*.jsonl.gz')
    return sorted(glob_fn(pattern))


# ============================================================
# 第二部分：数据源配置与训练计划（复用自 v3）
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
# 第三部分：多阶段索引序列生成（复用自 v3）
# ============================================================

def generate_global_index_sequence(
    source_indices: Dict[str, List[int]],
    plan: Dict,
    rng: random.Random,
) -> Tuple[List[str], List[int]]:
    """预生成全局样本序列，返回 (all_src_ids, all_line_ids)"""
    all_src_ids = []
    all_line_ids = []

    for phase_name, phase_info in plan['phases'].items():
        num_samples = phase_info['num_samples']
        ratios = phase_info['ratios']
        print(f"    生成 {phase_name}: {num_samples:,} 样本")

        source_samples = {}
        for src_name, ratio in ratios.items():
            n = int(num_samples * ratio)
            source_samples[src_name] = n

        diff = num_samples - sum(source_samples.values())
        if diff != 0:
            max_src = max(ratios, key=ratios.get)
            source_samples[max_src] += diff

        offset_before = len(all_src_ids)

        for src_name, n_samples in source_samples.items():
            if n_samples <= 0:
                continue

            available = source_indices[src_name]
            n_available = len(available)

            if n_available == 0:
                # 该数据源在此桶已无可用样本（被前面桶消费光），跳过
                print(f"    [跳过] {src_name}: 0 条可用样本 "
                      f"(需要 {n_samples:,})，将被重分配")
                continue

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

        # 重分配：将因 available=0 被跳过的数据源的配额分给有样本的数据源
        allocated_this_phase = len(all_src_ids) - offset_before
        shortfall = num_samples - allocated_this_phase
        if shortfall > 0:
            # 找出有可用样本的数据源
            available_sources = {
                src: source_indices[src]
                for src in source_samples
                if len(source_indices[src]) > 0
            }
            if available_sources:
                # 按配比比例分配 shortfall
                total_ratio = sum(ratios[s] for s in available_sources if s in ratios)
                redistributed = 0
                for src_name, src_pool in available_sources.items():
                    share = int(shortfall * ratios.get(src_name, 0) / max(total_ratio, 1e-9))
                    if share <= 0:
                        continue
                    extra = rng.choices(src_pool, k=share)
                    all_src_ids.extend([src_name] * share)
                    all_line_ids.extend(extra)
                    redistributed += share
                # 余数给最大池
                remaining_shortfall = shortfall - redistributed
                if remaining_shortfall > 0:
                    biggest_src = max(available_sources, key=lambda s: len(available_sources[s]))
                    extra = rng.choices(available_sources[biggest_src], k=remaining_shortfall)
                    all_src_ids.extend([biggest_src] * remaining_shortfall)
                    all_line_ids.extend(extra)
                print(f"    [重分配] {shortfall:,} 样本已从空数据源转移到有样本的数据源")
            else:
                print(f"    [警告] {phase_name} 所有数据源均无可用样本，"
                      f"缺少 {shortfall:,} 条！")

    # 阶段内 shuffle（保持阶段边界不变）
    # 动态获取各阶段实际样本数（避免因四舍五入导致偏移错误）
    offset = 0
    for i, (phase_name, phase_info) in enumerate(plan['phases'].items()):
        planned_n = phase_info['num_samples']
        # 用剩余样本数作为实际 n，避免最后一阶段溢出
        remaining_total = len(all_src_ids) - offset
        if i == len(plan['phases']) - 1:
            # 最后阶段：取所有剩余样本
            n = remaining_total
        else:
            n = min(planned_n, remaining_total)

        phase_pairs = list(zip(
            all_src_ids[offset:offset + n],
            all_line_ids[offset:offset + n],
        ))
        rng.shuffle(phase_pairs)
        all_src_ids[offset:offset + n] = [p[0] for p in phase_pairs]
        all_line_ids[offset:offset + n] = [p[1] for p in phase_pairs]
        offset += n

    assert len(all_src_ids) == len(all_line_ids)
    print(f"    全局索引序列: {len(all_src_ids):,} 样本")
    return all_src_ids, all_line_ids


# ============================================================
# 第四部分：seq_len 分桶索引（v4 新增）
# ============================================================

def build_bucket_indices(
    source_indexes: Dict[str, 'MultiFileSourceIndex'],
    seq_len_configs: List[Dict],
    text_fields: Tuple[str, ...] = TEXT_FIELDS_DEFAULT,
) -> Dict[str, Dict[str, List[int]]]:
    """
    扫描所有数据源的每一行，按字符数分桶。

    返回：bucket_indices[src_name][bucket_name] = [global_line_id, ...]

    分桶规则：
      - 取样本 text/content/input 字段的字符数作为长度估算
      - 按 seq_len_configs 中的 max_chars 阈值分到最近上界桶
      - 最后一个桶（seq32k）接收所有 chars > 32768 的样本
    """
    bucket_names = [cfg['name'] for cfg in seq_len_configs]
    bucket_indices = {}

    for src_name, idx in source_indexes.items():
        print(f"  分桶: {src_name} ({idx.total_lines:,} 行) ...")
        buckets = {name: [] for name in bucket_names}
        stats = {name: 0 for name in bucket_names}

        for line_id in range(idx.total_lines):
            try:
                line = idx.get_line(line_id)
                obj = json.loads(line)
            except Exception:
                # 无法解析的行分到最小桶，训练框架会处理
                buckets[bucket_names[0]].append(line_id)
                stats[bucket_names[0]] += 1
                continue

            # 取第一个存在的文本字段
            text = ''
            for field in text_fields:
                if field in obj and isinstance(obj[field], str):
                    text = obj[field]
                    break

            char_len = len(text)

            # 按阈值分桶（找最小满足条件的桶）
            assigned = bucket_names[-1]  # 默认最大桶
            for cfg in seq_len_configs[:-1]:
                if char_len <= cfg['max_chars']:
                    assigned = cfg['name']
                    break

            buckets[assigned].append(line_id)
            stats[assigned] += 1

            if (line_id + 1) % 500000 == 0:
                pct = (line_id + 1) / idx.total_lines * 100
                print(f"    进度: {line_id + 1:,}/{idx.total_lines:,} ({pct:.1f}%)")

        bucket_indices[src_name] = buckets

        # 打印分桶统计
        total = idx.total_lines
        stat_str = ', '.join(
            f"{name}={cnt:,}({cnt/max(total,1):.1%})"
            for name, cnt in stats.items()
        )
        print(f"    {src_name} 分桶: {stat_str}")

    return bucket_indices


def get_seq_source_indices(
    bucket_indices: Dict[str, Dict[str, List[int]]],
    seq_cfg: Dict,
    all_seq_configs: List[Dict],
    plan: Dict,
    rng: random.Random,
    consumed: Optional[Dict[str, set]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict]]:
    """
    为指定 seq_len 桶构建每个数据源的可用样本列表，不足时从最近邻桶填充。

    返回：
      source_indices: {src_name: [global_line_id, ...]}
      fill_stats: {src_name: {'native': int, 'filled': int, 'fill_from': [bucket_name, ...]}}

    填充策略（修正后）：
      1. 计算各数据源在各阶段的配比需求，取最大需求量作为阈值
      2. 本桶原生样本可独立使用（不加入 consumed，允许各桶使用自己的原生数据）
      3. 若可用样本 < 阈值，按从近到远顺序追加相邻桶样本（合并而非替换）
      4. 从邻居桶借走的样本会被标记为已消费，不会出现在后续桶的填充中
      5. 一直追加直到满足阈值或耗尽所有邻居桶

    consumed: {src_name: set(line_id, ...)} 跨桶共享，仅记录被借走的样本（原生样本不追踪）
    """
    bucket_name = seq_cfg['name']
    bucket_idx = next(i for i, c in enumerate(all_seq_configs) if c['name'] == bucket_name)
    n_buckets = len(all_seq_configs)

    # 初始化 consumed（首次调用时）
    if consumed is None:
        consumed = {src: set() for src in DATA_SOURCE_CONFIG}

    # 计算每个数据源在此桶中的需求量阈值
    src_needed = {}
    for src_name in DATA_SOURCE_CONFIG:
        max_needed = 0
        for phase_name, phase_info in plan['phases'].items():
            ratio = phase_info['ratios'].get(src_name, 0)
            needed = int(phase_info['num_samples'] * ratio) + 1
            max_needed = max(max_needed, needed)
        src_needed[src_name] = max_needed

    # 构造相邻桶搜索顺序（从近到远）
    neighbor_order = []
    for delta in range(1, n_buckets):
        for sign in (-1, +1):
            nb = bucket_idx + sign * delta
            if 0 <= nb < n_buckets:
                nb_name = all_seq_configs[nb]['name']
                if nb_name not in neighbor_order:
                    neighbor_order.append(nb_name)

    source_indices = {}
    fill_stats = {}

    for src_name in DATA_SOURCE_CONFIG:
        used = consumed[src_name]

        # 1. 从本桶取原生样本（不加入 consumed，允许各桶独立使用自己的原生样本）
        all_native_ids = bucket_indices[src_name].get(bucket_name, [])
        native_ids = list(all_native_ids)  # 不过滤 consumed
        rng.shuffle(native_ids)

        fill_stats[src_name] = {
            'native': len(native_ids),
            'filled': 0,
            'fill_from': [],
        }

        # 2. 判断是否需要填充：本桶样本 < 需求量阈值
        needed = src_needed[src_name]
        combined = list(native_ids)

        if len(combined) < needed:
            shortfall = needed - len(combined)
            for nb_name in neighbor_order:
                if shortfall <= 0:
                    break
                # 从邻居桶取样本，排除已被消费的
                all_nb_ids = bucket_indices[src_name].get(nb_name, [])
                nb_available = [lid for lid in all_nb_ids if lid not in used]
                if nb_available:
                    rng.shuffle(nb_available)
                    # 追加（合并）到可用池
                    combined.extend(nb_available)
                    # 标记借走的样本为已消费，后续桶不会再使用
                    for lid in nb_available:
                        used.add(lid)
                    fill_stats[src_name]['filled'] += len(nb_available)
                    fill_stats[src_name]['fill_from'].append(
                        f"{nb_name}({len(nb_available):,})"
                    )
                    shortfall -= len(nb_available)

            if fill_stats[src_name]['fill_from']:
                print(f"    [填充] {src_name}: "
                      f"原生 {len(native_ids):,} < 需求 {needed:,}, "
                      f"从 {fill_stats[src_name]['fill_from']} 借走 "
                      f"{fill_stats[src_name]['filled']:,} 条 -> "
                      f"可用池 {len(combined):,}（借走样本已标记消费，不会被其他桶重复借用）")

        source_indices[src_name] = combined

    return source_indices, fill_stats


# ============================================================
# 第五部分：流式写出（复用自 v3）
# ============================================================

def stream_write_merged(
    output_path: str,
    source_indexes: Dict[str, MultiFileSourceIndex],
    all_src_ids: List[str],
    all_line_ids: List[int],
    log_interval: int = 100000,
):
    """流式写出合并 jsonl"""
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
    return written, skipped


# ============================================================
# 第六部分：元信息文件（v4 新增 seq_len 信息）
# ============================================================

def write_seq_meta_json(
    output_path: str,
    seq_cfg: Dict,
    plan: Dict,
    source_indexes: Dict[str, MultiFileSourceIndex],
    fill_stats: Dict[str, Dict],
    all_src_ids: List[str],
    phase_offsets: List[Tuple[str, int, int]],
):
    """写出单个 seq_len 的 .meta.json"""
    meta = {
        'version': 'v4',
        'description': f'Recovery CPT seq_len curriculum data - {seq_cfg["name"]}',
        'seq_len_config': seq_cfg,
        'total_sources': len(source_indexes),
        'sources': {},
        'phases': {},
        'total_samples': plan['total_samples'],
        'total_steps': plan['total_steps'],
        'samples_per_step': plan['samples_per_step'],
    }

    for src_name, idx in source_indexes.items():
        fstat = fill_stats.get(src_name, {})
        meta['sources'][src_name] = {
            'total_lines': idx.total_lines,
            'num_files': len(idx.files),
            'file_stats': [
                {'path': os.path.basename(p), 'lines': n}
                for p, n in idx.get_file_stats()
            ],
            'total_tokens': DATA_SOURCE_CONFIG[src_name]['total_tokens'],
            'desc': DATA_SOURCE_CONFIG[src_name]['desc'],
            'bucket_native_count': fstat.get('native', 0),
            'bucket_filled_count': fstat.get('filled', 0),
            'bucket_fill_from': fstat.get('fill_from', []),
        }

    for phase_name, sample_start, sample_end in phase_offsets:
        phase_info = plan['phases'][phase_name]
        n = sample_end - sample_start
        phase_srcs = all_src_ids[sample_start:sample_end]
        dist = {}
        for s in phase_srcs:
            dist[s] = dist.get(s, 0) + 1

        meta['phases'][phase_name] = {
            'sample_start': sample_start,
            'sample_end': sample_end,
            'step_start': phase_info['step_start'],
            'step_end': phase_info['step_end'],
            'num_samples': n,
            'ratios': phase_info['ratios'],
            'actual_distribution': {k: v / max(n, 1) for k, v in dist.items()},
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  元信息: {output_path}")


def write_summary_json(
    output_path: str,
    seq_results: List[Dict],
    total_steps: int,
    batch_size_per_step: int,
    seed: int,
):
    """写出汇总 summary.json"""
    summary = {
        'version': 'v4',
        'description': 'Recovery CPT seq_len curriculum learning - training summary',
        'training_config': {
            'total_steps': total_steps,
            'steps_per_seq': total_steps // NUM_SEQ_BUCKETS,
            'batch_size_per_step': batch_size_per_step,
            'seed': seed,
            'curriculum_order': [cfg['name'] for cfg in SEQ_LEN_CONFIGS],
        },
        'seq_len_buckets': seq_results,
        'usage': {
            'description': '按以下顺序依次训练各 seq_len 阶段',
            'steps': [
                f"Phase {i+1}/{NUM_SEQ_BUCKETS}: seq_len={r['seq_cfg']['max_tokens']} "
                f"(steps {r['step_range'][0]}-{r['step_range'][1]}, "
                f"{r['total_samples']:,} samples)"
                for i, r in enumerate(seq_results)
            ],
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  汇总信息: {output_path}")


# ============================================================
# 第七部分：Manifest 解析（复用自 v3）
# ============================================================

def resolve_source_directory(
    manifest_entry: dict,
    manifest_key: str,
    data_root: Optional[str],
) -> Optional[str]:
    """从 manifest entry 中解析出数据源目录路径"""
    for key in ['output_dir', 'data_dir', 'dir', 'directory', 'path']:
        if key in manifest_entry:
            val = manifest_entry[key]
            if isinstance(val, str):
                if data_root and not os.path.isabs(val):
                    val = os.path.join(data_root, val)
                if os.path.isdir(val):
                    return val

    if 'files' in manifest_entry and isinstance(manifest_entry['files'], list):
        first_file = manifest_entry['files'][0]
        if isinstance(first_file, str):
            if data_root and not os.path.isabs(first_file):
                first_file = os.path.join(data_root, first_file)
            parent = os.path.dirname(first_file)
            if os.path.isdir(parent):
                return parent

    for v in manifest_entry.values():
        if isinstance(v, str):
            candidate = v
            if data_root and not os.path.isabs(candidate):
                candidate = os.path.join(data_root, candidate)
            if os.path.isdir(candidate):
                return candidate

    for v in manifest_entry.values():
        if isinstance(v, str):
            if '.jsonl.gz' in v or '.jsonl' in v:
                candidate = v
                if data_root and not os.path.isabs(candidate):
                    candidate = os.path.join(data_root, candidate)
                candidate = candidate if os.path.isdir(candidate) else os.path.dirname(candidate)
                if os.path.isdir(candidate):
                    return candidate

    return None


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Recovery CPT 数据预处理 v4 (seq_len 课程学习)'
    )
    parser.add_argument('--manifest', type=str, required=True,
                        help='manifest.json 路径')
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据文件根目录（manifest 中相对路径的前缀）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--total_steps', type=int, default=100000,
                        help='总训练步数（将被均分为 5 个 seq_len 阶段）')
    parser.add_argument('--batch_size_per_step', type=int, default=256,
                        help='每步样本数 = batch_size * gradient_accumulation_steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--tmp_dir', type=str, default=None,
                        help='临时解压目录（默认自动创建）')
    parser.add_argument('--keep_temp', action='store_true',
                        help='保留临时解压文件（调试用）')
    parser.add_argument('--text_field', type=str, default=None,
                        help='指定文本字段名（默认自动探测 text/content/input）')
    parser.add_argument('--chars_per_token', type=float, default=2.0,
                        help='字符/token 估算比例（默认 2.0）')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 如果指定了 text_field，插入到探测列表最前面
    text_fields = TEXT_FIELDS_DEFAULT
    if args.text_field:
        text_fields = (args.text_field,) + tuple(
            f for f in TEXT_FIELDS_DEFAULT if f != args.text_field
        )

    # 如果 chars_per_token 非默认，按比例调整 max_chars
    if args.chars_per_token != 2.0:
        ratio = args.chars_per_token / 2.0
        for cfg in SEQ_LEN_CONFIGS:
            if cfg['max_chars'] is not None:
                cfg['max_chars'] = int(cfg['max_chars'] * ratio)

    os.makedirs(args.output_dir, exist_ok=True)

    # 每个 seq_len 均分步数
    steps_per_seq = args.total_steps // NUM_SEQ_BUCKETS
    remainder = args.total_steps - steps_per_seq * NUM_SEQ_BUCKETS
    print(f"总步数 {args.total_steps:,} 均分为 {NUM_SEQ_BUCKETS} 个 seq_len 桶，"
          f"每桶 {steps_per_seq:,} 步"
          + (f"（最后一桶+{remainder}步）" if remainder else ""))

    # 临时目录
    if args.tmp_dir:
        tmp_base = args.tmp_dir
        os.makedirs(tmp_base, exist_ok=True)
        auto_cleanup = False
        tmp_obj = None
    else:
        _tmp_root = '/data2/tmp'
        os.makedirs(_tmp_root, exist_ok=True)
        tmp_obj = tempfile.TemporaryDirectory(prefix='recovery_cpt_v4_', dir=_tmp_root)
        tmp_base = tmp_obj.name
        auto_cleanup = True

    try:
        # ================================================================
        # Step 1: 读取 manifest，解析每个数据源的目录
        # ================================================================
        print()
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

        source_dirs = {}
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
                print(f"  entry: {json.dumps(entry, indent=2, ensure_ascii=False)[:500]}")
                sys.exit(1)

            if not os.path.isdir(data_dir):
                print(f"  [错误] 目录不存在: {data_dir}")
                sys.exit(1)

            source_dirs[src_name] = data_dir
            gz_files = discover_gz_files(data_dir)
            print(f"  {src_name}: {data_dir}")
            print(f"    -> 发现 {len(gz_files)} 个 .jsonl.gz 文件")

            if len(gz_files) == 0:
                print(f"  [错误] 目录下无 .jsonl.gz 文件: {data_dir}")
                sys.exit(1)

            for f_path in gz_files[:5]:
                size_mb = os.path.getsize(f_path) / (1024**2)
                print(f"    - {os.path.basename(f_path)} ({size_mb:.1f} MB)")
            if len(gz_files) > 6:
                print(f"    ... 省略 {len(gz_files) - 6} 个文件 ...")
                f_path = gz_files[-1]
                size_mb = os.path.getsize(f_path) / (1024**2)
                print(f"    - {os.path.basename(f_path)} ({size_mb:.1f} MB)")

        # ================================================================
        # Step 2: 解压所有文件并构建多文件偏移索引
        # ================================================================
        print()
        print("=" * 60)
        print("Step 2: 解压源文件并构建多文件偏移索引")
        print("=" * 60)

        source_indexes = {}

        for src_name, data_dir in source_dirs.items():
            print(f"\n  处理数据源: {src_name}")
            gz_files = discover_gz_files(data_dir)
            src_index = MultiFileSourceIndex(src_name)
            src_tmp_dir = os.path.join(tmp_base, src_name)
            os.makedirs(src_tmp_dir, exist_ok=True)

            for fi, gz_path in enumerate(gz_files):
                jsonl_path = decompress_gz_file(gz_path, src_tmp_dir)
                print(f"    构建索引 [{fi+1}/{len(gz_files)}]: "
                      f"{os.path.basename(gz_path)} ...")
                offsets = build_offset_index(jsonl_path)
                index_mb = len(offsets) * 8 / (1024**2)
                print(f"    索引完成: {len(offsets):,} 行, 索引大小 {index_mb:.1f} MB")
                src_index.add_file(jsonl_path, offsets)

            source_indexes[src_name] = src_index
            print(f"    {src_name} 汇总: {len(gz_files)} 个文件, "
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
        # Step 2.5（新增）: 构建 seq_len 分桶索引
        # ================================================================
        print()
        print("=" * 60)
        print("Step 2.5: 构建 seq_len 分桶索引")
        print("=" * 60)
        print(f"  文本字段探测顺序: {text_fields}")
        print(f"  分桶阈值 (chars): "
              + ', '.join(
                  f"{cfg['name']}≤{cfg['max_chars']}"
                  if cfg['max_chars'] else f"{cfg['name']}=any"
                  for cfg in SEQ_LEN_CONFIGS
              ))

        bucket_indices = build_bucket_indices(source_indexes, SEQ_LEN_CONFIGS, text_fields)

        # 打印全局分桶统计
        print()
        print("  全局分桶统计:")
        for bucket_cfg in SEQ_LEN_CONFIGS:
            bname = bucket_cfg['name']
            total_in_bucket = sum(
                len(bucket_indices[src].get(bname, []))
                for src in DATA_SOURCE_CONFIG
            )
            print(f"    {bname}: {total_in_bucket:,} 条样本")

        # ================================================================
        # Step 3-6: 逐 seq_len 生成数据
        # ================================================================
        print()
        print("=" * 60)
        print("Step 3-6: 逐 seq_len 桶生成训练数据")
        print("=" * 60)

        seq_results = []
        cumulative_steps = 0

        # 跨桶共享的消费记录：保证各 seq_len 桶之间样本零重叠
        # consumed[src_name] = set(line_id, ...)
        consumed = {src_name: set() for src_name in DATA_SOURCE_CONFIG}

        for seq_idx, seq_cfg in enumerate(SEQ_LEN_CONFIGS):
            seq_name = seq_cfg['name']
            # 最后一个桶获得余数步数
            cur_steps = steps_per_seq + (remainder if seq_idx == NUM_SEQ_BUCKETS - 1 else 0)

            print()
            print(f"  {'=' * 50}")
            print(f"  [{seq_idx+1}/{NUM_SEQ_BUCKETS}] {seq_name} "
                  f"(max_tokens={seq_cfg['max_tokens']}, steps={cur_steps:,})")
            print(f"  {'=' * 50}")

            seq_output_dir = os.path.join(args.output_dir, seq_name)
            os.makedirs(seq_output_dir, exist_ok=True)
            output_jsonl = os.path.join(seq_output_dir, f'recovery_cpt_{seq_name}.jsonl')
            output_meta = os.path.join(seq_output_dir, f'recovery_cpt_{seq_name}.meta.json')

            # Step 3a: 计算训练计划
            plan = compute_phase_plan(cur_steps, args.batch_size_per_step)
            print(f"  训练计划: {plan['total_steps']:,} 步 × "
                  f"{plan['samples_per_step']} 样本/步 = "
                  f"{plan['total_samples']:,} 样本")

            for phase_name, phase_info in plan['phases'].items():
                ratios_str = ', '.join(
                    f"{k}={v:.0%}" for k, v in phase_info['ratios'].items()
                )
                print(f"    {phase_name} "
                      f"(step {phase_info['step_start']}-{phase_info['step_end']}, "
                      f"{phase_info['num_samples']:,} 样本): {ratios_str}")

            # Step 3b: 获取此桶的 source_indices（含填充，消费追踪）
            print(f"\n  构建 {seq_name} 样本索引（含填充 + 去重）...")
            seq_source_indices_raw, fill_stats = get_seq_source_indices(
                bucket_indices, seq_cfg, SEQ_LEN_CONFIGS, plan, rng,
                consumed=consumed,
            )

            # 打印填充情况
            for src_name, fstat in fill_stats.items():
                native = fstat['native']
                filled = fstat['filled']
                fill_from = fstat['fill_from']
                if filled > 0:
                    print(f"    {src_name}: 原生 {native:,} + "
                          f"填充 {filled:,} (来自 {fill_from})")
                else:
                    print(f"    {src_name}: 原生 {native:,}")

            # Step 4: 生成全局索引序列
            print(f"\n  生成 {seq_name} 全局索引序列...")
            # 对每个数据源 shuffle 一次（每个 seq 使用独立随机状态）
            seq_source_indices = {}
            for src_name, ids in seq_source_indices_raw.items():
                ids_copy = list(ids)
                rng.shuffle(ids_copy)
                seq_source_indices[src_name] = ids_copy
                print(f"    {src_name}: {len(ids_copy):,} 行 (已打散)")

            all_src_ids, all_line_ids = generate_global_index_sequence(
                seq_source_indices, plan, rng
            )

            # 验证各阶段分布（动态处理，与 generate_global_index_sequence 保持一致）
            print(f"\n  [验证] {seq_name} 各阶段数据源实际分布:")
            phase_offsets = []
            offset = 0
            for i, (phase_name, phase_info) in enumerate(plan['phases'].items()):
                remaining_total = len(all_src_ids) - offset
                if i == len(plan['phases']) - 1:
                    n = remaining_total  # 最后阶段取剩余全部
                else:
                    n = min(phase_info['num_samples'], remaining_total)
                if n <= 0:
                    phase_offsets.append((phase_name, offset, offset))
                    continue
                phase_srcs = all_src_ids[offset:offset + n]
                dist = {}
                for s in phase_srcs:
                    dist[s] = dist.get(s, 0) + 1
                dist_str = ', '.join(
                    f"{k}={v/n:.1%}" for k, v in sorted(dist.items())
                )
                print(f"    {phase_name} ({n:,} 条): {dist_str}")
                phase_offsets.append((phase_name, offset, offset + n))
                offset += n

            # Step 5: 流式写出
            print()
            written, skipped = stream_write_merged(
                output_path=output_jsonl,
                source_indexes=source_indexes,
                all_src_ids=all_src_ids,
                all_line_ids=all_line_ids,
                log_interval=200000,
            )

            # Step 6: 写出 meta.json
            write_seq_meta_json(
                output_path=output_meta,
                seq_cfg=seq_cfg,
                plan=plan,
                source_indexes=source_indexes,
                fill_stats=fill_stats,
                all_src_ids=all_src_ids,
                phase_offsets=phase_offsets,
            )

            output_size_gb = os.path.getsize(output_jsonl) / (1024**3)
            seq_results.append({
                'seq_cfg': seq_cfg,
                'output_jsonl': output_jsonl,
                'output_meta': output_meta,
                'total_samples': plan['total_samples'],
                'total_steps': plan['total_steps'],
                'step_range': [cumulative_steps, cumulative_steps + cur_steps],
                'written': written,
                'skipped': skipped,
                'size_gb': round(output_size_gb, 3),
                'fill_stats': fill_stats,
            })

            cumulative_steps += cur_steps
            print(f"  {seq_name} 完成: {output_size_gb:.2f} GB, "
                  f"{written:,} 行写出, {skipped:,} 行跳过")

        # ================================================================
        # 打印跨桶填充统计（借走样本追踪）
        # ================================================================
        print()
        print("=" * 60)
        print("跨桶填充样本统计（从邻居桶借走的样本，防止同一借走样本被多桶重复使用）")
        print("=" * 60)
        total_borrowed = 0
        for src_name in DATA_SOURCE_CONFIG:
            n_consumed = len(consumed[src_name])
            total_borrowed += n_consumed
            print(f"  {src_name}: 被借走 {n_consumed:,} 条样本（不会出现在其他桶的填充中）")
        print(f"  总计: {total_borrowed:,} 条样本被跨桶借用（原生样本各桶独立使用）")

        # ================================================================
        # 写出 summary.json
        # ================================================================
        print()
        print("=" * 60)
        print("写出汇总信息")
        print("=" * 60)

        summary_path = os.path.join(args.output_dir, 'summary.json')
        write_summary_json(
            output_path=summary_path,
            seq_results=seq_results,
            total_steps=args.total_steps,
            batch_size_per_step=args.batch_size_per_step,
            seed=args.seed,
        )

        # ================================================================
        # 完成
        # ================================================================
        print()
        print("=" * 60)
        print("全部完成!")
        print("=" * 60)
        print(f"  输出目录: {args.output_dir}")
        print()
        print("  各 seq_len 数据集:")
        for r in seq_results:
            name = r['seq_cfg']['name']
            steps = r['total_steps']
            samples = r['total_samples']
            gb = r['size_gb']
            print(f"    {name}: {steps:,} 步, {samples:,} 样本, {gb:.2f} GB")
            print(f"      -> {r['output_jsonl']}")
        print()
        print("  训练命令（按课程顺序依次执行）:")
        step_acc = 0
        for i, r in enumerate(seq_results):
            name = r['seq_cfg']['name']
            max_len = r['seq_cfg']['max_tokens']
            steps = r['total_steps']
            print(f"\n  # [{i+1}/{NUM_SEQ_BUCKETS}] {name} (step {step_acc}-{step_acc+steps})")
            print("  " + "\n  ".join([
                "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\",
                "swift sft \\",
                "  --model /path/to/pruned_deepseek_v4_flash_4b_a1.5b \\",
                "  --model_type deepseek-v4 \\",
                "  --train_type full \\",
                f"  --dataset {r['output_jsonl']} \\",
                "  --template_type completion \\",
                f"  --max_length {max_len} \\",
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
                f"  --output_dir /path/to/output/recovery_cpt_{name} \\",
                "  --save_steps 200 \\",
                "  --save_total_limit 5 \\",
                "  --logging_steps 10 \\",
                f"  --seed {args.seed}",
            ]))
            step_acc += steps

    finally:
        if auto_cleanup and tmp_obj is not None:
            if not args.keep_temp:
                print(f"\n  清理临时目录: {tmp_base}")
                tmp_obj.cleanup()
            else:
                print(f"\n  保留临时目录 (--keep_temp): {tmp_base}")


if __name__ == '__main__':
    main()
