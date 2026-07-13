#!/usr/bin/env python3
"""
对比并修复 iter_0000000 (fresh distcp) 的 tid2eid buffer:
- 从 iter_0000002 (saved, 可训练的) 抽取正确的 tid2eid
- 写入 iter_0000000, 覆盖 round-robin placeholder
- 修复 "Split sizes doesn't match total dim 0 size" 错误

关键: 用完整 state dict 写回, 保留所有模型权重, 只覆盖 tid2eid

用法:
  source /opt/venv/bin/activate   # 训练用的 venv
  python3 /data/wanghui/deepseekv4_train_mcore_bridge/fix_tid2eid.py
"""
import sys, os, shutil
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

# ---- 配置 ----
SAVED_CKPT = "/workdir/model_output/checkpoints/iter_0000002"
FRESH_CKPT = "/workdir/model_output/dpsk-v4-4B-A1.5B_distcp/iter_0000000"
BACKUP_DIR = FRESH_CKPT + ".bak_before_tid2eid_fix"

def read_metadata(ckpt_dir):
    return FileSystemReader(ckpt_dir).read_metadata()

def build_full_state_dict_from_metadata(metadata, dtype_map=None):
    """根据完整 metadata 构造一个填满空 tensor 的 state dict (单进程加载用)"""
    sd = {}
    for key, ts_meta in metadata.state_dict_metadata.items():
        shape = getattr(ts_meta, 'size', None) or getattr(ts_meta, 'shape', None)
        if shape is None:
            continue
        props = getattr(ts_meta, 'properties', None)
        dtype = getattr(props, 'dtype', torch.bfloat16) if props else torch.bfloat16
        if dtype_map and key in dtype_map:
            dtype = dtype_map[key]
        # 用空的 tensor 占位, dcp.load 会填充真实数据
        sd[key] = torch.empty(shape, dtype=dtype)
    return sd

def main():
    print("=" * 60)
    print("Step 0: 恢复 fresh distcp 从备份 (如果存在)")
    print("=" * 60)
    if os.path.exists(BACKUP_DIR):
        print(f"备份存在: {BACKUP_DIR}")
        # 删除当前 fresh (被搞坏的那个) 然后从备份恢复
        if os.path.exists(FRESH_CKPT):
            print(f"删除被破坏的 {FRESH_CKPT}")
            shutil.rmtree(FRESH_CKPT)
        print(f"从备份恢复 -> {FRESH_CKPT}")
        shutil.copytree(BACKUP_DIR, FRESH_CKPT)
    else:
        print("没有备份, 跳过恢复步骤")

    # 1) 扫描两个 distcp 里的 tid2eid keys
    print("\n" + "=" * 60)
    print("Step 1: 扫描两个 distcp 里的 tid2eid keys")
    print("=" * 60)
    saved_meta = read_metadata(SAVED_CKPT)
    fresh_meta = read_metadata(FRESH_CKPT)
    saved_tid_keys = [k for k in saved_meta.state_dict_metadata.keys() if "tid2eid" in k.lower()]
    fresh_tid_keys = [k for k in fresh_meta.state_dict_metadata.keys() if "tid2eid" in k.lower()]
    print(f"saved: {len(saved_tid_keys)} tid2eid keys")
    for k in saved_tid_keys:
        sh = saved_meta.state_dict_metadata[k].size
        print(f"  {k}: shape={sh}")
    print(f"fresh: {len(fresh_tid_keys)} tid2eid keys")
    for k in fresh_tid_keys:
        sh = fresh_meta.state_dict_metadata[k].size
        print(f"  {k}: shape={sh}")

    if not saved_tid_keys or not fresh_tid_keys:
        print("ERROR: tid2eid 不存在")
        sys.exit(1)

    common_keys = set(saved_tid_keys) & set(fresh_tid_keys)
    print(f"\n共同 keys: {len(common_keys)}")
    print(f"saved total keys: {len(saved_meta.state_dict_metadata)}")
    print(f"fresh total keys: {len(fresh_meta.state_dict_metadata)}")
    
    # 列出 fresh 缺失的 keys (诊断) + 查看 _extra_state 是什么类型
    saved_all = set(saved_meta.state_dict_metadata.keys())
    fresh_all = set(fresh_meta.state_dict_metadata.keys())
    missing_in_fresh = sorted(saved_all - fresh_all)
    print(f"\nFresh 缺失 {len(missing_in_fresh)} 个 keys (相对 saved):")
    
    # 过滤掉 optimizer/scheduler 类的 keys
    model_missing = [k for k in missing_in_fresh if 'optimizer' not in k and 'lr_scheduler' not in k and 'rng_state' not in k]
    optim_missing = [k for k in missing_in_fresh if 'optimizer' in k or 'lr_scheduler' in k or 'rng_state' in k]
    print(f"  model 缺失: {len(model_missing)}")
    print(f"  optim/sched 缺失: {len(optim_missing)}")
    if model_missing:
        print(f"\n  Model 缺失 keys (前 20):")
        for k in model_missing[:20]:
            print(f"    {k}")
    
    # 查看 fresh metadata 里 _extra_state 类的属性
    print(f"\nFresh 里 _extra_state 类的 keys 详情:")
    for k in fresh_meta.state_dict_metadata.keys():
        if '_extra_state' in k or 'extra_state' in k:
            meta = fresh_meta.state_dict_metadata[k]
            print(f"  {k}")
            print(f"    type: {type(meta).__name__}")
            print(f"    dir: {[x for x in dir(meta) if not x.startswith('_')][:20]}")
            for attr in ['size', 'shape', 'dtype', 'properties', 'serializer', 'chunk_metadata']:
                if hasattr(meta, attr):
                    v = getattr(meta, attr)
                    print(f"    .{attr}: {v}")
            break
    
    return  # 诊断完就退出

    # 2) 加载 saved 的 tid2eid
    print("\n" + "=" * 60)
    print("Step 2: 加载 saved 的 tid2eid tensors")
    print("=" * 60)
    saved_state = build_full_state_dict_from_metadata(saved_meta)
    dcp.load(saved_state, checkpoint_id=SAVED_CKPT, no_dist=True)
    saved_tid = {k: v.clone() for k, v in saved_state.items() if "tid2eid" in k.lower() and v.numel() > 0}
    print(f"加载了 {len(saved_tid)} 个 tid2eid (非空)")
    for k, v in saved_tid.items():
        print(f"  {k}: shape={tuple(v.shape)} sample={v.flatten()[:6].tolist()}")

    # 3) 加载 fresh 的完整 state dict (所有 keys), 对比 tid2eid
    print("\n" + "=" * 60)
    print("Step 3: 加载 fresh 的完整 state dict (所有 keys), 对比 tid2eid")
    print("=" * 60)
    # 用 fresh 的 metadata 构造完整 state dict (包含所有模型权重)
    fresh_state = build_full_state_dict_from_metadata(fresh_meta)
    dcp.load(fresh_state, checkpoint_id=FRESH_CKPT, no_dist=True)
    print(f"加载了 {len(fresh_state)} 个 tensor (包含所有权重)")

    fixed = 0
    skipped = 0
    for key in sorted(common_keys):
        s = saved_tid.get(key)
        f = fresh_state.get(key)
        if s is None or f is None:
            print(f"\n  {key}: 缺失 (saved={s is not None}, fresh={f is not None})")
            continue
        same = torch.equal(s, f)
        print(f"\n  {key}")
        print(f"    saved: shape={tuple(s.shape)} dtype={s.dtype} sample={s.flatten()[:6].tolist()}")
        print(f"    fresh: shape={tuple(f.shape)} dtype={f.dtype} sample={f.flatten()[:6].tolist()}")
        print(f"    equal: {same}")
        if same:
            skipped += 1
        else:
            fresh_state[key] = s.clone()
            fixed += 1

    if fixed == 0:
        print(f"\n所有 tid2eid 一致 (修复 {fixed}, 跳过 {skipped}), 问题不在 tid2eid")
        return

    # 4) 备份 (如果还没有) 并写回完整的 state dict (包含所有权重)
    print("\n" + "=" * 60)
    print(f"Step 4: 备份 + 写回 {fixed} 个修复的 tid2eid (保留所有其他权重)")
    print("=" * 60)

    # 备份检查
    print(f"备份状态: {'存在' if os.path.exists(BACKUP_DIR) else '不存在'} - {BACKUP_DIR}")

    # 写回: 完整 state dict (包含所有权重 + 修复后的 tid2eid)
    print(f"写回完整 state dict 到 {FRESH_CKPT} (共 {len(fresh_state)} 个 tensor)...")
    writer = FileSystemWriter(FRESH_CKPT, overwrite=True)
    dcp.save(fresh_state, writer)
    print(f"已写入 {FRESH_CKPT}")

    # 验证
    print("\n验证修复结果:")
    verify_meta = read_metadata(FRESH_CKPT)
    print(f"  修复后 fresh total keys: {len(verify_meta.state_dict_metadata)}")
    for k in sorted(common_keys):
        if k in verify_meta.state_dict_metadata:
            print(f"  ✓ {k} 存在")

    print(f"\n完成: 修复 {fixed}, 跳过 {skipped}")
    print("现在可以跑训练测试:")
    print("  bash /data/wanghui/deepseekv4_train_mcore_bridge/run_pretrain.sh")

if __name__ == "__main__":
    main()
