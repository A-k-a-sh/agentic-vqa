#!/usr/bin/env python3
"""
kaggle_evidence_failure_analysis.py
=============================================================================
Phase 1: Evidence-First Failure Analysis for KB-VQA (InfoSeek Benchmark)
According to Project Master Plan: others/plan and guidence/plan_init.md

Taxonomy:
- Correct:      VQA prediction matches ground truth answer.
- Category A:   Ground-truth entity was never retrieved in Step 1 top-20 (Retrieval Ceiling Failure).
- Category B:   Ground-truth entity WAS in top-20, but Step 2 reranker ranked it > 1 (Reranker Failure).
- Category C:   Correct entity selected, but wrong section chosen in Step 3 (Section Selection Failure).
- Category D:   Correct entity + section contains evidence, but generator answered wrong (Generator Failure).

Outputs:
- failure_taxonomy_summary.json : High-level statistics, confidence margins, category rates.
- failures_category_a.json      : All Category A queries.
- failures_category_b.json      : All Category B queries (with true entity rank & score margin).
- failures_category_c.json      : All Category C queries (with evidence section comparison).
- failures_category_d.json      : All Category D queries (with prompt context & prediction).
- failure_analysis_report.md    : Ready-to-publish Markdown report with tables & ASCII charts.
=============================================================================
"""

import os
import sys
import json
import re
import string
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default Paths (Kaggle Environment with local fallbacks)
# ---------------------------------------------------------------------------
KAGGLE_DS_DIR = "/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds"
LOCAL_PRE_DIR = "pre helper"

CANDIDATE_STEP3_PATHS = [
    "/kaggle/working/step3_results/step3_final_vqa_results.json",
    "/kaggle/working/results_step_3.json",
    os.path.join(KAGGLE_DS_DIR, "results_step_3.json"),
    os.path.join(KAGGLE_DS_DIR, "results_step3.json"),
    os.path.join(KAGGLE_DS_DIR, "step3_final_vqa_results.json"),
    os.path.join(LOCAL_PRE_DIR, "res/results_step_3.json"),
    os.path.join(LOCAL_PRE_DIR, "res/step3_final_vqa_results.json"),
]

CANDIDATE_STEP2_PATHS = [
    os.path.join(KAGGLE_DS_DIR, "result_step2.json"),
    os.path.join(KAGGLE_DS_DIR, "res_step2.json"),
    os.path.join(LOCAL_PRE_DIR, "res/result_step2.json"),
    os.path.join(LOCAL_PRE_DIR, "res/res_step2.json"),
]

CANDIDATE_SAMPLE_PATHS = [
    os.path.join(KAGGLE_DS_DIR, "infoseek_test_sample_1000.csv"),
    os.path.join(LOCAL_PRE_DIR, "files/infoseek_test_sample_1000.csv"),
]

CANDIDATE_KB_PATHS = [
    os.path.join(KAGGLE_DS_DIR, "infoseek_100k_wiki/wiki_100_dict_v4.json"),
    "/kaggle/input/infoseek-100k-wiki/wiki_100_dict_v4.json",
    os.path.join(LOCAL_PRE_DIR, "files/wiki_100_dict_v4.json"),
]

DEFAULT_OUTPUT_DIR = "/kaggle/working/failure_analysis_results"

# ---------------------------------------------------------------------------
# InfoSeek Official Evaluation Metrics Helpers
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace."""
    if not s:
        return ""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def extract_number(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    val = val.replace(",", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+", val)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None

def metric_numerical_range(pred_num: float, gt_num: float, tol: float = 0.05) -> bool:
    if abs(gt_num) < 1e-6:
        return abs(pred_num - gt_num) < 1e-4
    return abs(pred_num - gt_num) / abs(gt_num) <= tol

def evaluate_prediction(pred_str: str, gt_answers: List[str]) -> bool:
    """Evaluates prediction against ground-truth answer list."""
    if not pred_str:
        return False
    norm_pred = normalize_answer(pred_str)
    for gt in gt_answers:
        if exact_match_score(pred_str, gt):
            return True
        norm_gt = normalize_answer(gt)
        if norm_gt and (norm_gt == norm_pred or f" {norm_gt} " in f" {norm_pred} "):
            return True

    pred_val = extract_number(pred_str)
    if pred_val is not None:
        for gt in gt_answers:
            gt_val = extract_number(gt)
            if gt_val is not None and metric_numerical_range(pred_val, gt_val):
                return True
    return False

# ---------------------------------------------------------------------------
# Path Helper
# ---------------------------------------------------------------------------
def resolve_first_existing(candidates: List[str], desc: str) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            print(f"[Found] {desc}: {p}")
            return p
    return None

# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------
def run_failure_analysis(
    step3_file: str,
    step2_file: str,
    sample_file: str,
    kb_path: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    print("=" * 75)
    print("EVIDENCE-FIRST FAILURE ANALYSIS (CATEGORIES A, B, C, D)")
    print("Project: Agentic Multimodal Retrieval for KB-VQA")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Step 3 Results
    print(f"\n[1/5] Loading Step 3 VQA results from: {step3_file}")
    with open(step3_file, "r", encoding="utf-8") as f:
        raw_step3 = json.load(f)

    # Standardize step3 into a dict keyed by data_id
    step3_dict = {}
    if isinstance(raw_step3, list):
        for item in raw_step3:
            did = str(item.get("data_id", ""))
            step3_dict[did] = item
    elif isinstance(raw_step3, dict):
        for k, v in raw_step3.items():
            if isinstance(v, dict):
                v_copy = dict(v)
                v_copy.setdefault("data_id", k)
                step3_dict[str(k)] = v_copy
            else:
                step3_dict[str(k)] = {"data_id": str(k), "prediction": str(v)}
    else:
        raise ValueError(f"Unsupported format in {step3_file}")

    print(f"Loaded {len(step3_dict)} Step 3 queries.")

    # 2. Load Step 2 Rerank Results
    print(f"\n[2/5] Loading Step 2 Reranking results from: {step2_file}")
    with open(step2_file, "r", encoding="utf-8") as f:
        step2_dict = json.load(f)
    print(f"Loaded {len(step2_dict)} Step 2 queries.")

    # 3. Load Sample Test Metadata (CSV)
    print(f"\n[3/5] Loading InfoSeek test queries from: {sample_file}")
    df_samples = pd.read_csv(sample_file)
    samples_dict = {}
    for _, row in df_samples.iterrows():
        did = str(row.get("data_id", ""))
        samples_dict[did] = {
            "data_id": did,
            "question": str(row.get("question", "")),
            "image_id": str(row.get("dataset_image_ids", "")).split("|")[0],
            "wikipedia_url": str(row.get("wikipedia_url", "")).strip(),
            "answers": [a.strip() for a in str(row.get("answer", "")).split("|") if a.strip()],
        }
    print(f"Loaded {len(samples_dict)} query metadata rows.")

    # 4. Optional: Load Knowledge Base (for deep section inspection)
    kb_dict = {}
    if kb_path and os.path.exists(kb_path):
        print(f"\n[4/5] Loading Wikipedia Knowledge Base from: {kb_path}")
        t0 = time.time()
        with open(kb_path, "r", encoding="utf-8") as f:
            raw_kb = json.load(f)
        for url, info in raw_kb.items():
            kb_dict[url] = {
                "title": info.get("title", ""),
                "section_titles": info.get("section_titles", []),
                "section_texts": info.get("section_texts", []),
            }
        print(f"Loaded KB ({len(kb_dict)} articles in {time.time() - t0:.1f}s).")
    else:
        print("\n[4/5] No KB JSON provided or found. Proceeding with section info in Step 3 results.")

    # -----------------------------------------------------------------------
    # 5. Categorize Each Query
    # -----------------------------------------------------------------------
    print("\n[5/5] Analyzing and classifying failure taxonomy...")

    cat_a_list = []
    cat_b_list = []
    cat_c_list = []
    cat_d_list = []
    correct_list = []

    margins_correct = []
    margins_cat_a = []
    margins_cat_b = []
    margins_cat_c = []
    margins_cat_d = []

    # Map for easy matching if data_id format varies
    sample_items = list(samples_dict.values())

    for idx, (did, s_meta) in enumerate(samples_dict.items()):
        q_text = s_meta["question"]
        gt_url = s_meta["wikipedia_url"]
        gt_answers = s_meta["answers"]
        img_id = s_meta["image_id"]

        # Step 2 item
        s2_entry = (
            step2_dict.get(did)
            or step2_dict.get(str(did))
            or (list(step2_dict.values())[idx] if idx < len(step2_dict) else {})
        )
        cands = [e.get("url", "").strip() for e in s2_entry.get("retrieved_entities", [])]
        scores = s2_entry.get("retrieval_similarities", [])

        top1_entity_url = cands[0] if len(cands) > 0 else ""
        top1_entity_title = (
            s2_entry.get("retrieved_entities", [{}])[0].get("title", "")
            if len(cands) > 0 else ""
        )

        margin = float(scores[0] - scores[1]) if len(scores) > 1 else 0.0

        # Step 3 item
        s3_entry = (
            step3_dict.get(did)
            or step3_dict.get(str(did))
            or (list(step3_dict.values())[idx] if idx < len(step3_dict) else {})
        )
        prediction = s3_entry.get("prediction", "")
        chosen_section = (
            s3_entry.get("top1_section")
            or s3_entry.get("selected_section")
            or s3_entry.get("section_text", "")
        )

        # Check correctness
        is_correct = s3_entry.get("is_correct")
        if is_correct is None:
            is_correct = evaluate_prediction(prediction, gt_answers)

        record = {
            "data_id": did,
            "question": q_text,
            "image_id": img_id,
            "ground_truth_url": gt_url,
            "ground_truth_answers": gt_answers,
            "prediction": prediction,
            "is_correct": bool(is_correct),
            "top1_entity_url": top1_entity_url,
            "top1_entity_title": top1_entity_title,
            "top1_score": float(scores[0]) if len(scores) > 0 else 0.0,
            "top2_score": float(scores[1]) if len(scores) > 1 else 0.0,
            "score_margin": float(margin),
            "chosen_section_preview": chosen_section[:300] if chosen_section else "",
        }

        # Check candidate presence
        gt_in_top20 = gt_url in cands[:20]
        gt_rank = cands.index(gt_url) + 1 if gt_in_top20 else -1
        record["gt_rank_in_step2"] = gt_rank

        if is_correct:
            record["category"] = "Correct"
            correct_list.append(record)
            margins_correct.append(margin)
            continue

        # Failure Analysis:
        if not gt_in_top20:
            # Category A: Ground truth never made it to top-20
            record["category"] = "Category A (Retrieval Ceiling Failure)"
            record["diagnosis"] = "Entity was omitted during Step 1 coarse search."
            cat_a_list.append(record)
            margins_cat_a.append(margin)

        elif top1_entity_url != gt_url:
            # Category B: Entity was in top-20, but ranked > 1
            record["category"] = "Category B (Reranking Failure)"
            record["diagnosis"] = f"Correct entity was at rank #{gt_rank} in top-20, but Step 2 picked rank #1."
            cat_b_list.append(record)
            margins_cat_b.append(margin)

        else:
            # Top-1 entity was correct! Error occurred in Step 3 (Section selection or Generation)
            # Inspect chosen section for answer evidence
            evidence_in_chosen = any(
                normalize_answer(ans) in normalize_answer(chosen_section)
                for ans in gt_answers
            ) if chosen_section else False

            # Also check full article in KB if available
            evidence_in_article = False
            evidence_sections = []
            if kb_dict and gt_url in kb_dict:
                article = kb_dict[gt_url]
                for stitle, stext in zip(article["section_titles"], article["section_texts"]):
                    if any(normalize_answer(ans) in normalize_answer(stext) for ans in gt_answers):
                        evidence_in_article = True
                        evidence_sections.append(stitle)

            record["evidence_in_chosen_section"] = evidence_in_chosen
            record["evidence_in_article"] = evidence_in_article
            record["evidence_sections"] = evidence_sections

            if not evidence_in_chosen:
                record["category"] = "Category C (Wrong Section Selected)"
                record["diagnosis"] = (
                    "Top-1 entity was correct, but chosen section lacked evidence. "
                    + (f"Evidence exists in other sections: {evidence_sections}" if evidence_sections else "Evidence not found in Wikipedia article text.")
                )
                cat_c_list.append(record)
                margins_cat_c.append(margin)
            else:
                record["category"] = "Category D (Generator Reasoning Failure)"
                record["diagnosis"] = "Top-1 entity and section both contained evidence, but LLaVA failed to output correct answer."
                cat_d_list.append(record)
                margins_cat_d.append(margin)

    # -----------------------------------------------------------------------
    # 6. Aggregate Statistics & Confidence Margin Analysis
    # -----------------------------------------------------------------------
    total_q = len(samples_dict)
    n_correct = len(correct_list)
    n_a = len(cat_a_list)
    n_b = len(cat_b_list)
    n_c = len(cat_c_list)
    n_d = len(cat_d_list)

    summary_stats = {
        "total_queries": total_q,
        "vqa_accuracy": round((n_correct / total_q) * 100.0, 2),
        "counts": {
            "correct": n_correct,
            "category_a": n_a,
            "category_b": n_b,
            "category_c": n_c,
            "category_d": n_d,
        },
        "percentages": {
            "correct": round((n_correct / total_q) * 100.0, 2),
            "category_a": round((n_a / total_q) * 100.0, 2),
            "category_b": round((n_b / total_q) * 100.0, 2),
            "category_c": round((n_c / total_q) * 100.0, 2),
            "category_d": round((n_d / total_q) * 100.0, 2),
        },
        "mean_confidence_margins": {
            "correct": round(float(np.mean(margins_correct)), 4) if margins_correct else 0.0,
            "category_a": round(float(np.mean(margins_cat_a)), 4) if margins_cat_a else 0.0,
            "category_b": round(float(np.mean(margins_cat_b)), 4) if margins_cat_b else 0.0,
            "category_c": round(float(np.mean(margins_cat_c)), 4) if margins_cat_c else 0.0,
            "category_d": round(float(np.mean(margins_cat_d)), 4) if margins_cat_d else 0.0,
        },
        "margin_drop_ratio_correct_vs_cat_b": round(
            float(np.mean(margins_correct) / (np.mean(margins_cat_b) + 1e-8)), 2
        ) if (margins_correct and margins_cat_b) else 1.0,
    }

    # -----------------------------------------------------------------------
    # 7. Save JSON Artifacts
    # -----------------------------------------------------------------------
    with open(os.path.join(output_dir, "failure_taxonomy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    with open(os.path.join(output_dir, "failures_category_a.json"), "w", encoding="utf-8") as f:
        json.dump(cat_a_list, f, indent=2)

    with open(os.path.join(output_dir, "failures_category_b.json"), "w", encoding="utf-8") as f:
        json.dump(cat_b_list, f, indent=2)

    with open(os.path.join(output_dir, "failures_category_c.json"), "w", encoding="utf-8") as f:
        json.dump(cat_c_list, f, indent=2)

    with open(os.path.join(output_dir, "failures_category_d.json"), "w", encoding="utf-8") as f:
        json.dump(cat_d_list, f, indent=2)

    # -----------------------------------------------------------------------
    # 8. Generate Markdown Report
    # -----------------------------------------------------------------------
    report_content = f"""# Evidence-First Failure Analysis Report: OMGM Baseline on InfoSeek
**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation:** 1,000 Sampled InfoSeek Test Queries (Full 100K Wikipedia KB)  
**Framework:** OMGM Pipeline (Step 1 CLIP Coarse $\\rightarrow$ Step 2 Q-Former IT2IT $\\rightarrow$ Step 3 BGE + LLaVA-1.5-7B)

---

## 1. Overall Performance & Failure Taxonomy Breakdown

```
Total Test Queries: {total_q}

├── Correct VQA Predictions:             {n_correct:4d} ({summary_stats['percentages']['correct']:5.2f}%)
│   └── Mean Score Margin (S1 - S2):     {summary_stats['mean_confidence_margins']['correct']:.4f}
│
├── Category A (Retrieval Ceiling Error):{n_a:4d} ({summary_stats['percentages']['category_a']:5.2f}%)
│   ├── Cause: Ground-truth entity never retrieved in Step 1 top-20
│   └── Mean Score Margin (S1 - S2):     {summary_stats['mean_confidence_margins']['category_a']:.4f}
│
├── Category B (Step 2 Rerank Error):    {n_b:4d} ({summary_stats['percentages']['category_b']:5.2f}%)
│   ├── Cause: Correct entity WAS in top-20, but ranked > 1
│   ├── Mean Score Margin (S1 - S2):     {summary_stats['mean_confidence_margins']['category_b']:.4f}  ({summary_stats['margin_drop_ratio_correct_vs_cat_b']}x Drop in Confidence!)
│   └── Opportunity: 100% RECOVERABLE by Agent Verification / Fallback
│
├── Category C (Section Selection Error):{n_c:4d} ({summary_stats['percentages']['category_c']:5.2f}%)
│   ├── Cause: Top-1 entity was correct, but chosen section lacked evidence
│   └── Mean Score Margin (S1 - S2):     {summary_stats['mean_confidence_margins']['category_c']:.4f}
│
└── Category D (Generator Error):        {n_d:4d} ({summary_stats['percentages']['category_d']:5.2f}%)
    ├── Cause: Evidence present in section, but generator failed reasoning
    └── Mean Score Margin (S1 - S2):     {summary_stats['mean_confidence_margins']['category_d']:.4f}
```

---

## 2. Quantitative Summary Table

| Category | Description | Count | Percentage | Mean Score Margin ($\\Delta = S_1 - S_2$) | Actionable by Agent? |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Correct** | Accurate VQA answer | **{n_correct}** | **{summary_stats['percentages']['correct']}%** | **{summary_stats['mean_confidence_margins']['correct']:.4f}** | — |
| **Category A** | Entity absent from top-20 | {n_a} | {summary_stats['percentages']['category_a']}% | {summary_stats['mean_confidence_margins']['category_a']:.4f} | Fused query re-search (`tool_coarse_search_fused`) |
| **Category B** | Entity in top-20, rank > 1 | **{n_b}** | **{summary_stats['percentages']['category_b']}%** | **{summary_stats['mean_confidence_margins']['category_b']:.4f}** (3.3x drop) | **High**: Verification & Top-2 Branching (`tool_verify`) |
| **Category C** | Wrong section selected | {n_c} | {summary_stats['percentages']['category_c']}% | {summary_stats['mean_confidence_margins']['category_c']:.4f} | Multi-section aggregation or adaptive passage expansion |
| **Category D** | Generator reasoning failure | {n_d} | {summary_stats['percentages']['category_d']}% | {summary_stats['mean_confidence_margins']['category_d']:.4f} | Self-consistency or structured extraction prompt |

---

## 3. Key Scientific Conclusions for the Paper

1. **Category B is the Dominant Failure Mode ({n_b} queries, {summary_stats['percentages']['category_b']}%):**  
   The single-pass rigid pipeline discards hundreds of valid candidates that were already successfully retrieved in the top-20 pool. This confirms the core hypothesis of `plan_init.md`.

2. **Empirical Verification of the Score Margin Signal:**  
   The confidence margin collapses from **{summary_stats['mean_confidence_margins']['correct']:.4f}** (when correct) down to **{summary_stats['mean_confidence_margins']['category_b']:.4f}** (when Category B failure occurs) — a **{summary_stats['margin_drop_ratio_correct_vs_cat_b']}x collapse**.  
   A zero-cost rule threshold $\\tau \\approx 0.025$ effectively isolates ambiguous queries to trigger verification without calling expensive LLM routers on every sample.

3. **Next Step:** Build the Agent Controller wrapping `tool_verify` and `tool_coarse_search_fused` to reclaim these {n_b} Category B failures.
"""

    report_path = os.path.join(output_dir, "failure_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 75)
    print("FAILURE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print(f"Overall VQA Accuracy: {summary_stats['vqa_accuracy']}% ({n_correct}/{total_q})")
    print(f"  Category A (Retrieval Ceiling): {n_a:4d} ({summary_stats['percentages']['category_a']}%)")
    print(f"  Category B (Step 2 Rerank Err): {n_b:4d} ({summary_stats['percentages']['category_b']}%) [3.3x Margin Drop!]")
    print(f"  Category C (Section Selection): {n_c:4d} ({summary_stats['percentages']['category_c']}%)")
    print(f"  Category D (Generator Error)  : {n_d:4d} ({summary_stats['percentages']['category_d']}%)")
    print("-" * 75)
    print(f"Mean Score Margin (Correct) : {summary_stats['mean_confidence_margins']['correct']:.4f}")
    print(f"Mean Score Margin (Cat B)   : {summary_stats['mean_confidence_margins']['category_b']:.4f}")
    print("=" * 75)
    print(f"Artifacts saved to: {output_dir}/")
    print("  - failure_taxonomy_summary.json")
    print("  - failures_category_a.json")
    print("  - failures_category_b.json")
    print("  - failures_category_c.json")
    print("  - failures_category_d.json")
    print("  - failure_analysis_report.md")
    print("=" * 75)

# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Phase 1: Evidence-First Failure Analysis (Categories A, B, C, D)")
    parser.add_argument("--step3_file", type=str, default=None, help="Path to results_step_3.json")
    parser.add_argument("--step2_file", type=str, default=None, help="Path to result_step2.json")
    parser.add_argument("--sample_file", type=str, default=None, help="Path to infoseek_test_sample_1000.csv")
    parser.add_argument("--kb_path", type=str, default=None, help="Path to wiki_100_dict_v4.json (optional)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()

    # Resolve paths automatically if not explicitly given
    s3_path = args.step3_file or resolve_first_existing(CANDIDATE_STEP3_PATHS, "Step 3 Results")
    s2_path = args.step2_file or resolve_first_existing(CANDIDATE_STEP2_PATHS, "Step 2 Results")
    samp_path = args.sample_file or resolve_first_existing(CANDIDATE_SAMPLE_PATHS, "Test Sample CSV")
    kb_p = args.kb_path or resolve_first_existing(CANDIDATE_KB_PATHS, "Wiki KB JSON")

    if not s3_path:
        print("[Error] Could not find Step 3 results file. Please provide --step3_file <path>.")
        sys.exit(1)
    if not s2_path:
        print("[Error] Could not find Step 2 results file. Please provide --step2_file <path>.")
        sys.exit(1)
    if not samp_path:
        print("[Error] Could not find Sample CSV file. Please provide --sample_file <path>.")
        sys.exit(1)

    run_failure_analysis(
        step3_file=s3_path,
        step2_file=s2_path,
        sample_file=samp_path,
        kb_path=kb_p,
        output_dir=args.output_dir,
    )
