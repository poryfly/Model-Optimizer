"""检查 tid2eid 是否有重复 expert id。

用法:
  python check_tid2eid.py <path>

path 可以是:
  - HF 格式目录（含 model-*.safetensors + model.safetensors.index.json）
  - Megatron distcp 目录（含 __0_0.distcp 等 + .metadata）
"""

import json
import os
import sys

import torch


def check_tensor(name, t):
    print(f"  {name}: shape={t.shape}, dtype={t.dtype}")
    print(f"    min={t.min().item()}, max={t.max().item()}")
    print(f"    row 0:   {t[0].tolist()}")
    print(f"    row 1:   {t[1].tolist()}")
    print(f"    row 100: {t[100].tolist()}")
    n = min(1000, t.shape[0])
    dup = sum(1 for i in range(n) if len(set(t[i].tolist())) < t.shape[1])
    print(f"    rows with duplicates (first {n}): {dup}/{n}")
    print()


def check_hf(path):
    from safetensors import safe_open

    idx_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        print(f"[SKIP] {idx_path} not found, not a HF directory")
        return False

    with open(idx_path) as f:
        idx = json.load(f)

    tid2eid_keys = [k for k in idx["weight_map"] if "tid2eid" in k]
    if not tid2eid_keys:
        print("[SKIP] no tid2eid keys in index")
        return False

    print(f"=== HF Format: {path} ===")
    print(f"tid2eid keys: {tid2eid_keys}")

    for k in tid2eid_keys:
        shard = idx["weight_map"][k]
        shard_path = os.path.join(path, shard)
        if not os.path.exists(shard_path):
            print(f"  [WARN] shard not found: {shard_path}")
            continue
        with safe_open(shard_path, framework="pt") as f:
            t = f.get_tensor(k)
            check_tensor(k, t)
    return True


def check_distcp(path):
    from torch.distributed.checkpoint import FileSystemReader
    import torch.distributed.checkpoint as dcp

    reader = FileSystemReader(path)
    meta = reader.read_metadata()

    tid2eid_keys = [k for k in meta.state_dict_metadata if "tid2" in k.lower()]
    if not tid2eid_keys:
        print("[SKIP] no tid2eid keys in distcp metadata")
        return False

    print(f"=== distcp Format: {path} ===")
    print(f"tid2eid keys: {tid2eid_keys}")

    state_dict = {}
    for k in tid2eid_keys:
        shape = tuple(meta.state_dict_metadata[k].size)
        state_dict[k] = torch.empty(shape, dtype=torch.int32)

    dcp.load(state_dict, checkpoint_id=path, no_dist=True)

    for k in tid2eid_keys:
        check_tensor(k, state_dict[k])
    return True


def main():
    if len(sys.argv) < 2:
        # 默认路径
        paths = [
            "/data/wanghui/deepseekv4_train/megatron_input/dpsk-v4-4B-A1.5B"]
    else:
        paths = [sys.argv[1]]

    for p in paths:
        if not os.path.isdir(p):
            print(f"[SKIP] {p} is not a directory")
            continue

        print(f"\n{'='*60}")
        print(f"Checking: {p}")
        print(f"{'='*60}\n")

        # 尝试 HF 格式
        ok = check_hf(p)
        if not ok:
            # 尝试 distcp 格式
            check_distcp(p)


if __name__ == "__main__":
    main()
