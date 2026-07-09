#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation harness for DeepSeek-V4-4B-A1B model.

Runs a curated benchmark suite via EleutherAI's ``lm-evaluation-harness`` and
prints a summary table.  Falls back to the local ``examples/llm_eval/lm_eval_hf.py``
wrapper if available, otherwise uses the standard ``lm_eval`` CLI.

Supported benchmarks:
  - General: mmlu, mmlu_pro, bbh, arc_challenge
  - Code: humaneval, mbpp
  - Math: gsm8k, math
  - Multilingual: mgsm
  - Long context: needle_in_a_haystack (if supported by lm_eval)

Example:
    python3 eval_deepseek_v4_4b.py \
        --model_path /data/output/v4-4b-a1b-sft \
        --batch_size 8
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any


BENCHMARK_GROUPS = {
    "general": ["mmlu", "mmlu_pro", "bbh", "arc_challenge"],
    "code": ["humaneval", "mbpp"],
    "math": ["gsm8k", "math"],
    "multilingual": ["mgsm"],
    "long_context": [],  # populated conditionally below
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-V4-4B-A1B on a standard benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--groups",
        type=str,
        default="general,code,math,multilingual",
        help="Comma-separated list of benchmark groups to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save per-task JSON results",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code to lm_eval",
    )
    parser.add_argument(
        "--use_modelopt_wrapper",
        action="store_true",
        help="Use examples/llm_eval/lm_eval_hf.py wrapper",
    )
    return parser.parse_args()


def build_tasks(groups: list[str]) -> list[str]:
    tasks = []
    for g in groups:
        tasks.extend(BENCHMARK_GROUPS.get(g, []))
    return list(dict.fromkeys(tasks))  # preserve order, remove dups


def run_lm_eval(
    model_path: str,
    tasks: list[str],
    batch_size: int,
    device: str,
    output_dir: str,
    trust_remote_code: bool,
    use_modelopt_wrapper: bool,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    task_str = ",".join(tasks)
    results_file = os.path.join(output_dir, "results.json")

    if use_modelopt_wrapper:
        script = os.path.join(
            os.path.dirname(__file__), "..", "llm_eval", "lm_eval_hf.py"
        )
        script = os.path.abspath(script)
        if not os.path.exists(script):
            raise FileNotFoundError(f"ModelOpt eval wrapper not found: {script}")
        cmd = [
            sys.executable,
            script,
            "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype=bfloat16{',trust_remote_code=True' if trust_remote_code else ''}",
            "--tasks", task_str,
            "--batch_size", str(batch_size),
            "--device", device,
            "--output_path", output_dir,
            "--log_samples",
        ]
    else:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype=bfloat16{',trust_remote_code=True' if trust_remote_code else ''}",
            "--tasks", task_str,
            "--batch_size", str(batch_size),
            "--device", device,
            "--output_path", output_dir,
        ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # lm_eval writes results to output_dir/<model_name>/results_*.json
    # Find the most recent results file
    results_files = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                results_files.append(os.path.join(root, f))

    if not results_files:
        raise RuntimeError("No results file found after evaluation.")

    results_files.sort(key=os.path.getmtime, reverse=True)
    with open(results_files[0]) as f:
        return json.load(f)


def extract_scores(results: dict[str, Any], tasks: list[str]) -> dict[str, float]:
    scores = {}
    for task in tasks:
        key = f"{task}"
        # Try common metric names
        for metric in ["acc", "acc_norm", "exact_match", "pass_at_1"]:
            val = results.get("results", {}).get(task, {}).get(metric)
            if val is not None:
                scores[task] = float(val)
                break
        if task not in scores:
            scores[task] = float("nan")
    return scores


def print_summary(scores: dict[str, float], group_scores: dict[str, list[float]]) -> None:
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    for group, tasks in group_scores.items():
        print(f"\n[{group}]")
        for task in tasks:
            score = scores.get(task, float("nan"))
            print(f"  {task:30s}: {score * 100:6.2f}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    tasks = build_tasks(groups)
    if not tasks:
        print("No tasks selected.")
        return

    print(f"Evaluating model: {args.model_path}")
    print(f"Benchmark groups: {groups}")
    print(f"Tasks: {tasks}")

    results = run_lm_eval(
        model_path=args.model_path,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
        use_modelopt_wrapper=args.use_modelopt_wrapper,
    )

    scores = extract_scores(results, tasks)

    group_scores = {g: [t for t in BENCHMARK_GROUPS[g] if t in tasks] for g in groups}
    print_summary(scores, group_scores)

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({"scores": scores, "group_scores": group_scores}, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
