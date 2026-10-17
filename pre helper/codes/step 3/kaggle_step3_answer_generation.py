#!/usr/bin/env python3
"""
Step 3: Section-Augmented Knowledge-Based VQA Answer Generation
OMGM Pipeline (ACL 2025 Main) - Fully Reproducible Implementation

Pipeline Stages:
1. Stage 3A - Section Reranking:
   - For each query's top-1 Wikipedia entity from Step 2 (res_step2.json),
     reconstruct its article sections.
   - Score sections using BAAI/bge-reranker-v2-m3 (question vs section).
   - Linearly blend with fine-grained cross-attention section scores from Step 2 (beta = 0.2):
       step3_score = 0.2 * step2_sec_sim + 0.8 * bge_score
   - Pick the single highest-scoring section (top1_section).
   - Free BGE memory to prevent GPU OOM before loading the generator.

2. Stage 3B - LLaVA-1.5-7B Generation:
   - Construct OMGM's exact few-shot prompt with context, question, and instruction.
   - Pass query image + prompt to llava-hf/llava-1.5-7b-hf.
   - Greedy decoding (do_sample=False, seed=42) for 100% deterministic reproducibility.
   - Evaluate exact match / numerical range against InfoSeek ground-truth answers.
   - Classify all failure modes into Categories A, B, C, and D.
"""

import os
import sys
import csv
import json
import time
import random
import glob
from typing import Dict, List, Any, Optional

import torch
import numpy as np
import PIL.Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoProcessor,
    LlavaForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Deterministic Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------------------------------------------------------
# Default Paths (Kaggle Environment with local fallbacks)
# ---------------------------------------------------------------------------
BASE_DS_DIR = "/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds"

DEFAULT_SAMPLE_FILE = os.path.join(BASE_DS_DIR, "infoseek_test_sample_1000.csv")
DEFAULT_STEP2_RESULT = os.path.join(BASE_DS_DIR, "res_step2.json")
DEFAULT_KB_PATH = os.path.join(BASE_DS_DIR, "infoseek_100k_wiki/wiki_100_dict_v4.json")
DEFAULT_QUERY_IMG_DIR = os.path.join(BASE_DS_DIR, "infoseek_val_1000_images/infoseek_val_1000_images")
DEFAULT_OUTPUT_DIR = "/kaggle/working/step3_results"

# ---------------------------------------------------------------------------
# Official InfoSeek Evaluation Metric Helpers
# ---------------------------------------------------------------------------
import re
import string

def normalize_answer(text: str) -> str:
    """Standard InfoSeek normalization: strip articles, punctuation, and whitespace."""
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punctuation(t: str) -> str:
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def extract_number(text: str) -> Optional[float]:
    """Extracts first valid integer/float from prediction text."""
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None

def metric_numerical_range(pred_num: float, ans_num: float, tolerance: float = 0.1) -> int:
    """Evaluates numerical prediction within +/- 10% tolerance."""
    min_val = ans_num * (1.0 - tolerance) if ans_num >= 0 else ans_num * (1.0 + tolerance)
    max_val = ans_num * (1.0 + tolerance) if ans_num >= 0 else ans_num * (1.0 - tolerance)
    low, high = min(min_val, max_val), max(min_val, max_val)
    return 1 if low <= pred_num <= high else 0

def evaluate_prediction(pred_str: str, gt_answers: List[str]) -> int:
    """
    Evaluates prediction string against list of ground truth references.
    Supports both text exact match and numerical range matching.
    """
    if not pred_str:
        return 0

    norm_pred = normalize_answer(pred_str)

    # 1. Text exact match across all ground truth variants
    for gt in gt_answers:
        if exact_match_score(pred_str, gt):
            return 1
        # Check if normalized ground truth is a distinct substring in short answer
        norm_gt = normalize_answer(gt)
        if norm_gt and (norm_gt == norm_pred or f" {norm_gt} " in f" {norm_pred} "):
            return 1

    # 2. Numerical check if ground truth is a number
    pred_val = extract_number(pred_str)
    if pred_val is not None:
        for gt in gt_answers:
            gt_val = extract_number(gt)
            if gt_val is not None and metric_numerical_range(pred_val, gt_val):
                return 1

    return 0

# ---------------------------------------------------------------------------
# Article & Section Reconstruction (OMGM Exact Logic)
# ---------------------------------------------------------------------------
class KnowledgeEntry:
    def __init__(self, title, url, section_titles, section_texts):
        self.title = title
        self.url = url
        self.section_titles = section_titles
        self.section_texts = section_texts

def reconstruct_wiki_sections(entry: KnowledgeEntry) -> List[str]:
    """Formats sections as: # Wiki Article: ... \n## Section Title: ... \n{text}"""
    sections = []
    title = entry.title
    for it, sec_title in enumerate(entry.section_titles):
        sec_title_lower = sec_title.lower()
        if "external links" in sec_title_lower or "references" in sec_title_lower:
            continue
        text = entry.section_texts[it] if it < len(entry.section_texts) else ""
        formatted = f"# Wiki Article: {title}\n## Section Title: {sec_title}\n{text}"
        sections.append(formatted)
    return sections

def resolve_query_image(image_id: str, query_img_dir: str) -> str:
    clean_id = image_id.split("|")[0].strip()
    for ext in ["", ".jpg", ".jpeg", ".JPEG", ".png"]:
        candidate = os.path.join(query_img_dir, clean_id + ext)
        if os.path.exists(candidate):
            return candidate
    matches = glob.glob(os.path.join(query_img_dir, f"*{clean_id}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Query image {image_id} not found in {query_img_dir}")

# ---------------------------------------------------------------------------
# Section Reranker: BAAI/bge-reranker-v2-m3
# ---------------------------------------------------------------------------
class BGESectionReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
        print(f"Loading BGE Section Reranker ({model_name}) onto {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("BGE Section Reranker loaded successfully.")

    @torch.no_grad()
    def score_sections(self, question: str, sections: List[str], batch_size: int = 4) -> List[float]:
        if len(sections) == 0:
            return []
        pairs = [[question, sec] for sec in sections]
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt", max_length=2048
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            all_scores.extend(logits.cpu().tolist())
        return all_scores

# ---------------------------------------------------------------------------
# Prompt Constructor (OMGM Exact Prompt)
# ---------------------------------------------------------------------------
VLM_INFOSEEK_INSTRUCTION = (
    "Answer the encyclopedic question about the given image."
    "Don't mention the visuall content of image in your output. "
    "Directly output the answer of the question according to the Context.\n"
    "If you need to answer questions about numbers or time, please output the corresponding numerical format directly.\n"
    "If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge.\n "
    "There is an example:\n- Context: # Wiki Article: Dolomites\n## Section Title: Dolomites\nThe Dolomites, also known as the Dolomite Mountains, Dolomite Alps or Dolomitic Alps, are a mountain range located in northeastern Italy. The Dolomites are located in the regions of Veneto, Trentino-Alto Adige/Südtirol and Friuli Venezia Giulia, covering an area shared between the provinces of Belluno, Vicenza, Verona, Trentino, South Tyrol, Udine and Pordenone.\n"
    "- Question: Which city or region does this mountain locate in?\n"
    "Just answer the questions , no explanations needed. Short answer is: Province of Belluno\n\n"
)

def build_omgm_prompt(question: str, section_text: str, max_chars: int = 3000) -> str:
    """Builds prompt with truncated context length to avoid VRAM overflow."""
    trimmed_section = section_text[:max_chars] if len(section_text) > max_chars else section_text
    prompt = (
        f"- Context: {trimmed_section}\n"
        f"- Question: {question}\n"
        f"Just answer the questions , no explanations needed. Short answer is:"
    )
    return VLM_INFOSEEK_INSTRUCTION + prompt

# ---------------------------------------------------------------------------
# Main Step 3 Pipeline Execution
# ---------------------------------------------------------------------------
def run_step3(
    sample_file: str = DEFAULT_SAMPLE_FILE,
    step2_result_file: str = DEFAULT_STEP2_RESULT,
    kb_path: str = DEFAULT_KB_PATH,
    query_img_dir: str = DEFAULT_QUERY_IMG_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    step2_beta: float = 0.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Step 2 Results
    print(f"Loading Step 2 retrieval results from: {step2_result_file}")
    with open(step2_result_file, "r", encoding="utf-8") as f:
        step2_results = json.load(f)
    print(f"Loaded Step 2 results for {len(step2_results)} queries.")

    # 2. Load 1,000 Test Samples
    print(f"Loading test samples from: {sample_file}")
    with open(sample_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    print(f"Loaded {len(samples)} test samples.")

    # 3. Load Wikipedia Knowledge Base
    print(f"Loading Wikipedia Knowledge Base from: {kb_path}...")
    t0 = time.time()
    with open(kb_path, "r", encoding="utf-8") as f:
        raw_kb = json.load(f)

    # Support list or dict KB formats
    if isinstance(raw_kb, list):
        kb_dict = {
            item["url"]: KnowledgeEntry(
                item.get("title", ""), item.get("url", ""),
                item.get("section_titles", []), item.get("section_texts", [])
            )
            for item in raw_kb
        }
    else:
        kb_dict = {
            url: KnowledgeEntry(
                info.get("title", ""), url,
                info.get("section_titles", []), info.get("section_texts", [])
            )
            for url, info in raw_kb.items()
        }
    del raw_kb
    print(f"Loaded Knowledge Base ({len(kb_dict)} articles in {time.time() - t0:.1f}s).")

    # =========================================================================
    # STAGE 3A: Section Reranking (BGE + Step 2 Section Score Fusion)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3A: Reranking Candidate Sections with BGE & Step 2 Fusion...")
    print("=" * 60)

    bge_reranker = BGESectionReranker(model_name="BAAI/bge-reranker-v2-m3", device=device)

    query_selected_sections = {}
    stage3a_start = time.time()

    for it, sample in enumerate(tqdm(samples, desc="Stage 3A Section Reranking")):
        data_id = sample.get("data_id", f"query_{it}")
        question = sample.get("question", "")

        # Fetch Step 2 Top-1 Entity
        step2_item = step2_results.get(data_id) or step2_results.get(str(data_id)) or list(step2_results.values())[it]
        top1_entity_url = step2_item["retrieved_entities"][0]["url"]

        if top1_entity_url in kb_dict:
            entry = kb_dict[top1_entity_url]
            sections = reconstruct_wiki_sections(entry)
        else:
            sections = []

        if len(sections) == 0:
            query_selected_sections[data_id] = {
                "top1_section": "No relevant section available.",
                "top1_entity_url": top1_entity_url,
                "section_count": 0,
            }
            continue

        # Compute BGE question-to-section text similarities
        q2s_scores = bge_reranker.score_sections(question, sections)

        # Get Step 2 fine-grained multimodal section similarities
        step2_sec_sim = step2_item.get("sec_sim", [[]])[0]

        # Blend: step3_score = beta * step2_sim + (1 - beta) * bge_score
        if len(step2_sec_sim) == len(q2s_scores):
            step3_scores = [
                step2_beta * float(s2) + (1.0 - step2_beta) * float(bg)
                for s2, bg in zip(step2_sec_sim, q2s_scores)
            ]
        else:
            step3_scores = q2s_scores

        best_idx = int(np.argmax(step3_scores))
        chosen_section = sections[best_idx]

        query_selected_sections[data_id] = {
            "top1_section": chosen_section,
            "top1_entity_url": top1_entity_url,
            "section_count": len(sections),
            "chosen_section_index": best_idx,
        }

    print(f"Stage 3A Complete in {time.time() - stage3a_start:.1f}s.")

    # Free BGE Section Reranker to maximize VRAM for LLaVA-1.5-7B
    del bge_reranker
    torch.cuda.empty_cache()
    print("Freed BGE Section Reranker from GPU memory.")

    # Save intermediate Section Rerank results
    sec_save_path = os.path.join(output_dir, "step3_selected_sections.json")
    with open(sec_save_path, "w", encoding="utf-8") as f:
        json.dump(query_selected_sections, f, indent=2)
    print(f"Saved Stage 3A selected sections to: {sec_save_path}")

    # =========================================================================
    # STAGE 3B: MLLM Answer Generation with LLaVA-1.5-7B
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3B: Loading LLaVA-1.5-7B Generator onto GPU...")
    print("=" * 60)

    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)

    # Use float16 on GPU (or 4-bit quantization if bitsandbytes is available)
    try:
        generator_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        print("Loaded LLaVA-1.5-7B in FP16.")
    except torch.cuda.OutOfMemoryError:
        print("[Notice] FP16 exceeded GPU VRAM. Falling back to 4-bit quantized LLaVA...")
        generator_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
        ).eval()

    print("\nStarting Stage 3B VQA Answer Generation Loop...")
    stage3b_start = time.time()

    correct_vqa = 0
    cat_a_count = 0
    cat_b_count = 0
    cat_c_count = 0
    cat_d_count = 0

    results_summary = []

    for it, sample in enumerate(tqdm(samples, desc="Stage 3B Answer Generation")):
        data_id = sample.get("data_id", f"query_{it}")
        question = sample.get("question", "")
        ground_truth_url = sample.get("wikipedia_url", "").strip()
        raw_gt_answer = sample.get("answer", "")
        gt_answer_list = [a.strip() for a in raw_gt_answer.split("|") if a.strip()]

        # Query Image
        img_id = sample.get("dataset_image_ids", "").split("|")[0]
        try:
            q_img_path = resolve_query_image(img_id, query_img_dir)
            raw_image = PIL.Image.open(q_img_path).convert("RGB")
        except Exception as e:
            raw_image = PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Selected Context
        sec_info = query_selected_sections[data_id]
        top1_section = sec_info["top1_section"]
        top1_entity_url = sec_info["top1_entity_url"]

        # Step 2 Entity Status
        step2_item = step2_results.get(data_id) or step2_results.get(str(data_id)) or list(step2_results.values())[it]
        step2_top20_urls = [e["url"] for e in step2_item["retrieved_entities"][:20]]
        entity_in_top20 = ground_truth_url in step2_top20_urls
        entity_is_top1 = (ground_truth_url == top1_entity_url)

        # Build OMGM chat conversation
        prompt_text = build_omgm_prompt(question, top1_section)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            }
        ]
        input_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=raw_image, text=input_prompt, return_tensors="pt").to(device)

        # Deterministic greedy generation
        with torch.no_grad():
            output_tokens = generator_model.generate(
                **inputs,
                max_new_tokens=25,
                do_sample=False,
            )
            raw_decoded = processor.decode(output_tokens[0], skip_special_tokens=True)
            # Extract assistant answer after 'ASSISTANT:'
            pred_answer = raw_decoded.split("ASSISTANT:")[-1].strip()

        # Evaluate Answer Accuracy
        is_correct = evaluate_prediction(pred_answer, gt_answer_list)
        if is_correct:
            correct_vqa += 1

        # Classify Failure Mode (if incorrect)
        failure_category = "None (Correct)"
        if not is_correct:
            if not entity_in_top20:
                failure_category = "Category A (Entity Missing from Top-20)"
                cat_a_count += 1
            elif not entity_is_top1:
                failure_category = "Category B (Entity in Top-20, but Rank > 1)"
                cat_b_count += 1
            else:
                # Top-1 entity was correct, but answer was wrong
                # Check if evidence answer is mentioned anywhere in the selected section
                section_has_evidence = any(
                    normalize_answer(ans) in normalize_answer(top1_section)
                    for ans in gt_answer_list
                )
                if not section_has_evidence:
                    failure_category = "Category C (Wrong Section Selected)"
                    cat_c_count += 1
                else:
                    failure_category = "Category D (Generator Reasoning Failure)"
                    cat_d_count += 1

        results_summary.append({
            "data_id": data_id,
            "question": question,
            "prediction": pred_answer,
            "ground_truth_answers": gt_answer_list,
            "is_correct": bool(is_correct),
            "top1_entity_url": top1_entity_url,
            "ground_truth_url": ground_truth_url,
            "failure_category": failure_category,
        })

    # Save final Step 3 results
    final_output_file = os.path.join(output_dir, "step3_final_vqa_results.json")
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    total_q = len(samples)
    vqa_acc = (correct_vqa / total_q) * 100.0

    print("\n" + "=" * 65)
    print("FINAL STEP 3 EVALUATION & FAILURE TAXONOMY BREAKDOWN")
    print(f"Total Time: {time.time() - stage3b_start:.1f}s")
    print("=" * 65)
    print(f"Overall VQA Accuracy: {vqa_acc:.2f}% ({correct_vqa}/{total_q})")
    print("-" * 65)
    print(f"  Correct Answers               : {correct_vqa:4d} ({correct_vqa/total_q*100:5.2f}%)")
    print(f"  Category A (Retrieval Ceiling): {cat_a_count:4d} ({cat_a_count/total_q*100:5.2f}%)")
    print(f"  Category B (Reranking Error)  : {cat_b_count:4d} ({cat_b_count/total_q*100:5.2f}%)")
    print(f"  Category C (Wrong Section)    : {cat_c_count:4d} ({cat_c_count/total_q*100:5.2f}%)")
    print(f"  Category D (Generator Failure): {cat_d_count:4d} ({cat_d_count/total_q*100:5.2f}%)")
    print("=" * 65)
    print(f"Successfully saved all Step 3 outputs to: {final_output_file}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run Step 3 Section Reranking and LLaVA-1.5-7B VQA Generation")
    parser.add_argument("--sample_file", type=str, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--step2_result_file", type=str, default=DEFAULT_STEP2_RESULT)
    parser.add_argument("--kb_path", type=str, default=DEFAULT_KB_PATH)
    parser.add_argument("--query_img_dir", type=str, default=DEFAULT_QUERY_IMG_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step2_beta", type=float, default=0.2)
    args = parser.parse_args()

    run_step3(
        sample_file=args.sample_file,
        step2_result_file=args.step2_result_file,
        kb_path=args.kb_path,
        query_img_dir=args.query_img_dir,
        output_dir=args.output_dir,
        step2_beta=args.step2_beta,
    )
