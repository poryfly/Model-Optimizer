#!/usr/bin/env python3
"""Merge per-rank Megatron state_dicts into one complete state_dict.

After pruning with PP>1, each rank saves its own portion of the model
(with local layer indices). This script merges them with correct global
layer index remapping.

Usage:
    python3 merge_pruned_state_dicts.py \
        --megatron_dir /data/output/v4-flash-full-v4_megatron \
        --num_ranks 8
"""

import argparse
import os
import re

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron_dir", required=True, help="Directory containing per-rank .pt files")
    parser.add_argument("--num_ranks", type=int, required=True, help="Number of PP ranks")
    parser.add_argument("--cleanup", action="store_true", help="Delete per-rank files after merge")
    args = parser.parse_args()

    merged = {}
    layer_offset = 0

    for rank in range(args.num_ranks):
        path = os.path.join(args.megatron_dir, f"pruned_model_rank{rank}.pt")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue

        print(f"Loading rank {rank}: {path}")
        sd = torch.load(path, map_location="cpu", weights_only=True)

        rank_layer_indices = set()
        for k in sd.keys():
            m = re.match(r"decoder\.layers\.(\d+)\.", k)
            if m:
                rank_layer_indices.add(int(m.group(1)))

        local_layers = sorted(rank_layer_indices)
        if local_layers:
            max_local = max(local_layers)
            num_local_layers = max_local + 1
        else:
            num_local_layers = 0

        print(f"  {len(sd)} keys, layers {local_layers[:3]}...{local_layers[-3:] if local_layers else []}, "
              f"offset={layer_offset}")

        for k, v in sd.items():
            m = re.match(r"decoder\.layers\.(\d+)\.(.*)", k)
            if m:
                local_idx = int(m.group(1))
                global_idx = local_idx + layer_offset
                new_key = f"decoder.layers.{global_idx}.{m.group(2)}"
            else:
                new_key = k
            merged[new_key] = v

        layer_offset += num_local_layers

    output_path = os.path.join(args.megatron_dir, "pruned_model.pt")
    print(f"\nMerged: {len(merged)} keys, {layer_offset} total layers")
    print(f"Saving to {output_path}")
    torch.save(merged, output_path)
    print("Done!")

    if args.cleanup:
        for rank in range(args.num_ranks):
            path = os.path.join(args.megatron_dir, f"pruned_model_rank{rank}.pt")
            if os.path.exists(path):
                os.remove(path)
                print(f"  Cleaned up {path}")


if __name__ == "__main__":
    main()
