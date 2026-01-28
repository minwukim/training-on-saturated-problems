# ============================================================
# Code cell 1: eval_runner_yaml.py (NO confidence logic)
# ============================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified MATH evaluation runner (vLLM) — YAML-configured (NO confidence features)
- Reads key hyperparams from a YAML config using TrlParser.parse_args_and_config().
- Runs multiple datasets in one pass.
- Separate CSVs per dataset.
- N generations per question (configurable).
- Saves: question, response, ground_truth, reward, token_count.

Requires:
  - vLLM >= 0.6.x
  - datasets
  - numpy, pandas
  - pyyaml (via TrlParser config parsing)
  - trl (for TrlParser)
  - math_verify (verify, parse)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from datasets import load_dataset

from vllm import LLM, SamplingParams
from math_verify import verify, parse

from trl import TrlParser


# ──────────────────────────────────────────────────────────────────────────────
# YAML-backed arguments (only the fields you asked for)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalArguments:
    model_name: str
    output_dir: str

    tensor_parallel_size: int
    max_model_len: int

    temperature: float
    top_p: float
    top_k: int
    max_response_tokens: int
    num_generations: int

    # Optional but useful
    seed: int = 1
    batch_size: int = 5


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template (DeepSeek-style)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
    "<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>"
)


# ──────────────────────────────────────────────────────────────────────────────
# Datasets (same as your original; CSV names relative to output_dir)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DATASETS = [
    {
        "name": "math-ai/aime25",
        "split": "test",
        "trust_remote_code": True,
        "question_field": "problem",
        "solution_field": "answer",
        "out_csv": "aime25_results.csv",
        "gt_mode": "wrap_answer_in_boxed",
    },
    {
        "name": "math-ai/aime24",
        "split": "test",
        "trust_remote_code": True,
        "question_field": "problem",
        "solution_field": "solution",
        "out_csv": "aime24_results.csv",
        "gt_mode": "extract_boxed_from_solution",
    },
    {
        "name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "trust_remote_code": True,
        "question_field": "problem",
        "solution_field": "solution",
        "out_csv": "math500_results.csv",
        "gt_mode": "extract_boxed_from_solution",
    },
    {
        "name": "rawsh/2024_AMC12",
        "split": "train",
        "trust_remote_code": True,
        "question_field": "problem",
        "solution_field": "answer",
        "out_csv": "amc12_2024_results.csv",
        "gt_mode": "wrap_answer_in_boxed",
    },
    {
        "name": "MathArena/hmmt_feb_2025",
        "split": "train",
        "trust_remote_code": True,
        "question_field": "problem",
        "solution_field": "answer",
        "out_csv": "hmmt25_results.csv",
        "gt_mode": "wrap_answer_in_boxed",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Return the last \\boxed{...} or \\fbox{...} expression from `string`.
    Mirrors the original behavior you shared.
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def prepare_ground_truths(ds, solution_field: str, gt_mode: str) -> List[Optional[str]]:
    """
    Returns per-example ground truths in boxed form matching reward evaluator.
    gt_mode:
      - "extract_boxed_from_solution": applies last_boxed_only_string to ds[solution_field]
      - "wrap_answer_in_boxed": wraps raw answer with \\boxed{...}
    """
    gts: List[Optional[str]] = []
    if gt_mode == "extract_boxed_from_solution":
        for ex in ds:
            sol = ex[solution_field]
            sol_str = sol if isinstance(sol, str) else str(sol)
            gt = last_boxed_only_string(sol_str)
            gts.append(gt)
    elif gt_mode == "wrap_answer_in_boxed":
        for ex in ds:
            ans = ex[solution_field]
            gts.append(f"\\boxed{{{ans}}}")
    else:
        raise ValueError(f"Unknown gt_mode: {gt_mode}")
    return gts


def reward_without_format(pred_text: str, gt_text_lastboxed: str) -> int:
    """
    EXACTLY mirrors your reference behavior:
      - pred_text is NOT post-processed by any 'last boxed only' function
      - gt_text_lastboxed has already been processed by last_boxed_only_string (or boxed)
    """
    try:
        return int(verify(parse(pred_text), parse(gt_text_lastboxed)))
    except Exception:
        return 0


def make_sampling_params(args: EvalArguments) -> SamplingParams:
    kwargs = dict(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_response_tokens),
        n=int(args.num_generations),
        seed=int(args.seed),
        # NOTE: no logprobs, no confidence
    )
    if args.top_k is not None and int(args.top_k) >= 0:
        kwargs["top_k"] = int(args.top_k)
    return SamplingParams(**kwargs)


def build_llm(args: EvalArguments) -> LLM:
    return LLM(
        model=args.model_name,
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_model_len=int(args.max_model_len),
        trust_remote_code=True,
    )


def write_batch(csv_path: Path, records: List[Dict[str, Any]], first_batch: bool):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records, columns=[
        "dataset_question_index",
        "question",
        "response",
        "ground_truth",
        "reward",
        "token_count",
    ])

    df.to_csv(
        csv_path,
        mode="a",
        header=first_batch,
        index=False,
        quoting=1,       # csv.QUOTE_ALL
        escapechar="\\",
    )


def run_dataset(
    *,
    llm: LLM,
    sampling: SamplingParams,
    batch_size: int,
    num_generations: int,
    out_dir: Path,
    ds_name: str,
    split: str,
    trust_remote_code: bool,
    question_field: str,
    solution_field: str,
    gt_mode: str,
    out_csv_name: str,
):
    out_csv = out_dir / out_csv_name

    print(f"\nLoading dataset: {ds_name} [{split}]")
    ds = load_dataset(ds_name, split=split, trust_remote_code=trust_remote_code)

    questions = [ex[question_field] for ex in ds]
    ground_truths = prepare_ground_truths(ds, solution_field, gt_mode)

    total = len(questions)
    print(f"Loaded {total} questions — batch size {batch_size}, n={num_generations}")
    first_batch = True

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_idx = list(range(start, end))
        batch_questions = [questions[i] for i in batch_idx]
        batch_gts = [ground_truths[i] for i in batch_idx]

        prompts = [SYSTEM_PROMPT.format(prompt=q) for q in batch_questions]

        print(f"  • Generating [{start}..{end-1}] ({len(prompts)} prompts)")
        try:
            outputs = llm.generate(prompts, sampling)
        except Exception as e:
            print(f"[ERROR] Generation failed for batch {start}-{end-1}: {e}")
            continue

        # Build rows; for each input i, we take outputs in order
        records: List[Dict[str, Any]] = []
        for local_i, q_i in enumerate(batch_idx):
            q_text = batch_questions[local_i]
            gt_text = batch_gts[local_i]

            out_for_prompt = outputs[local_i].outputs  # list length ~ num_generations
            for g, piece in enumerate(out_for_prompt[:num_generations]):
                resp = piece.text
                token_ids = getattr(piece, "token_ids", []) or []
                token_count = len(token_ids)

                rw = reward_without_format(resp, gt_text if isinstance(gt_text, str) else str(gt_text))

                records.append({
                    "dataset_question_index": q_i,
                    "question": q_text,
                    "response": resp,
                    "ground_truth": gt_text,
                    "reward": rw,
                    "token_count": token_count,
                })

        write_batch(out_csv, records, first_batch)
        first_batch = False
        print(f"    ✓ Saved {len(records)} rows → {out_csv}")

    print(f"Finished {ds_name} [{split}] → {out_csv}")


def main():
    # Parse CLI + YAML config using TrlParser (same style you referenced)
    parser = TrlParser(dataclass_types=[EvalArguments])
    args = parser.parse_args_and_config()[0]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== vLLM • Multi-dataset eval (YAML-configured; NO confidence) ===")
    print(f"Model: {args.model_name}")
    print(f"Output dir: {out_dir}")
    print(f"TP={args.tensor_parallel_size}  MaxModelLen={args.max_model_len}  MaxNewTokens={args.max_response_tokens}")
    print(f"T={args.temperature}  top_p={args.top_p}  top_k={args.top_k}  n={args.num_generations}  seed={args.seed}")
    print(f"Batch Size: {args.batch_size}")

    llm = build_llm(args)
    sampling = make_sampling_params(args)

    for cfg in DEFAULT_DATASETS:
        run_dataset(
            llm=llm,
            sampling=sampling,
            batch_size=int(args.batch_size),
            num_generations=int(args.num_generations),
            out_dir=out_dir,
            ds_name=cfg["name"],
            split=cfg["split"],
            trust_remote_code=cfg["trust_remote_code"],
            question_field=cfg["question_field"],
            solution_field=cfg["solution_field"],
            gt_mode=cfg["gt_mode"],
            out_csv_name=cfg["out_csv"],
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
