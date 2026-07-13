"""把 checkpoint 里的 tid2eid 替换为 dedup 后的值。

训练时 router.py 的 _dedup_top_indices 在每次 forward 运行时去重，
但 checkpoint 里存的 tid2eid 还是原始 broken 值（大量重复 63）。
此脚本把整个 tid2eid 表预计算 dedup，写回 checkpoint，使保存的值
与训练时实际使用的路由一致。

用法:
  python fix_tid2eid_in_checkpoint.py <checkpoint_dir>

支持:
  - HF 格式 (含 model-*.safetensors + model.safetensors.index.json)
  - Megatron distcp 格式 (含 __0_0.distcp 等)

注意: 会自动备份原始文件到 .bak_fix_tid2eid
"""

import json
import os
import shutil
import sys

import torch


def _dedup_top_indices(top_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """向量化去重，跟 router.py 的 _dedup_top_indices 逐字相同。"""
    topk = top_indices.shape[1]
    device = top_indices.device
    result = top_indices.clone()

    used_mask = torch.zeros(
        result.shape[0], num_experts, dtype=torch.bool, device=device
    )

    for k in range(topk):
        col = result[:, k]
        row_idx = torch.arange(result.shape[0], device=device)
        already_used = used_mask[row_idx, col]

        if already_used.any():
            dup_row_idx = row_idx[already_used]
            dup_used = used_mask[dup_row_idx]
            available = ~dup_used
            num_available = available.sum(dim=1)

            no_avail = num_available == 0
            if no_avail.any():
                reset_idx = dup_row_idx[no_avail]
                used_mask[reset_idx] = False
                for j in range(k):
                    used_mask[reset_idx, result[reset_idx, j]] = True
                available = ~used_mask[reset_idx]
                num_available = available.sum(dim=1)

            first_avail = available.long().argmax(dim=1)

            still_zero = num_available == 0
            if still_zero.any():
                fallback_idx = dup_row_idx[still_zero]
                first_avail[still_zero] = (
                    result[fallback_idx, 0] + k
                ) % num_experts

            result[dup_row_idx, k] = first_avail

        used_mask[row_idx, result[:, k]] = True
    return result


def fix_hf(path):
    """修复 HF 格式 checkpoint 的 tid2eid。"""
    from safetensors import safe_open
    from safetensors.torch import save_file

    idx_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        return False

    with open(idx_path) as f:
        idx = json.load(f)

    tid2eid_keys = [k for k in idx["weight_map"] if "tid2eid" in k]
    if not tid2eid_keys:
        print("[SKIP] no tid2eid keys found")
        return False

    # 读 config 获取 num_experts
    config_path = os.path.join(path, "config.json")
    num_experts = 64  # 默认
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        num_experts = cfg.get("n_routed_experts", cfg.get("num_experts", 64))

    print(f"num_experts: {num_experts}")

    # 按 shard 分组
    shard_keys = {}
    for k in tid2eid_keys:
        shard = idx["weight_map"][k]
        if shard not in shard_keys:
            shard_keys[shard] = []
        shard_keys[shard].append(k)

    for shard, keys in shard_keys.items():
        shard_path = os.path.join(path, shard)
        bak_path = shard_path + ".bak_fix_tid2eid"

        # 备份
        if not os.path.exists(bak_path):
            shutil.copy2(shard_path, bak_path)
            print(f"  backup: {bak_path}")

        # 读取整个 shard 的所有 tensor
        tensors = {}
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        # 修复 tid2eid
        for key in keys:
            t = tensors[key]
            print(f"  {key}: before dedup, row 0 = {t[0].tolist()}")
            dup_before = sum(
                1 for i in range(min(1000, t.shape[0]))
                if len(set(t[i].tolist())) < t.shape[1]
            )
            print(f"    rows with duplicates (first 1000): {dup_before}/1000")

            fixed = _dedup_top_indices(t, num_experts)
            print(f"    after dedup, row 0 = {fixed[0].tolist()}")
            dup_after = sum(
                1 for i in range(min(1000, fixed.shape[0]))
                if len(set(fixed[i].tolist())) < fixed.shape[1]
            )
            print(f"    rows with duplicates (first 1000): {dup_after}/1000")

            tensors[key] = fixed

        # 写回
        save_file(tensors, shard_path, metadata=idx.get("metadata", {}))
        print(f"  saved: {shard_path}")

    return True


def fix_distcp(path):
    """修复 Megatron distcp 格式 checkpoint 的 tid2eid。"""
    from torch.distributed.checkpoint import FileSystemReader
    import torch.distributed.checkpoint as dcp

    reader = FileSystemReader(path)
    meta = reader.read_metadata()

    tid2eid_keys = [k for k in meta.state_dict_metadata if "tid2" in k.lower()]
    if not tid2eid_keys:
        print("[SKIP] no tid2eid keys found in distcp")
        return False

    # 读 config 获取 num_experts
    num_experts = 64
    config_path = os.path.join(os.path.dirname(path), "..", "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        num_experts = cfg.get("n_routed_experts", cfg.get("num_experts", 64))

    print(f"num_experts: {num_experts}")

    # 备份
    bak_dir = path + ".bak_fix_tid2eid"
    if not os.path.exists(bak_dir):
        shutil.copytree(path, bak_dir)
        print(f"  backup: {bak_dir}")

    # 加载 tid2eid
    state_dict = {}
    for k in tid2eid_keys:
        shape = tuple(meta.state_dict_metadata[k].size)
        dtype = meta.state_dict_metadata[k].properties.dtype
        state_dict[k] = torch.empty(shape, dtype=dtype)

    dcp.load(state_dict, checkpoint_id=path, no_dist=True)

    # dedup
    for k in tid2eid_keys:
        t = state_dict[k]
        print(f"  {k}: before dedup, row 0 = {t[0].tolist()}")
        dup_before = sum(
            1 for i in range(min(1000, t.shape[0]))
            if len(set(t[i].tolist())) < t.shape[1]
        )
        print(f"    rows with duplicates (first 1000): {dup_before}/1000")

        fixed = _dedup_top_indices(t, num_experts)
        print(f"    after dedup, row 0 = {fixed[0].tolist()}")
        dup_after = sum(
            1 for i in range(min(1000, fixed.shape[0]))
            if len(set(fixed[i].tolist())) < fixed.shape[1]
        )
        print(f"    rows with duplicates (first 1000): {dup_after}/1000")

        state_dict[k] = fixed

    # 写回
    dcp.save(state_dict, checkpoint_id=path)
    print(f"  saved: {path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python fix_tid2eid_in_checkpoint.py <checkpoint_dir>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isdir(path):
        print(f"ERROR: {path} is not a directory")
        sys.exit(1)

    print(f"Fixing tid2eid in: {path}\n")

    # 先试 HF，再试 distcp
    ok = fix_hf(path)
    if not ok:
        ok = fix_distcp(path)

    if ok:
        print("\n✅ Done! tid2eid has been deduped in the checkpoint.")
        print("   Backup saved as .bak_fix_tid2eid")
        print("   The router.py dedup is no longer needed for this checkpoint.")
    else:
        print("\n❌ Failed to fix: neither HF nor distcp format detected.")


if __name__ == "__main__":
    main()
