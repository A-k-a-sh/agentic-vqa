"""
Kaggle Runner for OMGM Step 2: Fine-Grained Multimodal (IT2IT) Reranking.

Pipeline Summary:
1. Loads 1,000 InfoSeek test questions and ground-truth Wikipedia URLs.
2. Loads Step 1 coarse retrieval candidates from result.json (top-20 entities per query).
3. Directly loads candidate Wikipedia image mappings from candidate_wiki_images_max3.csv (instant load).
4. Reconstructs Wikipedia article sections from wiki_100_dict_v4.json.
5. Employs the Q-Former IT2IT reranker (model_23912.pth) to score query multimodal tokens against candidate section-image multimodal tokens.
6. Blends Step 1 and Step 2 similarities: score = alpha * sim_step1 + (1 - alpha) * sim_step2.
7. Evaluates and reports Recall@1, 5, 10, 20.
8. Saves reranked outputs to /kaggle/working/step2_results/result.json.
"""

import os
import sys
import csv
import json
import glob
import time
from argparse import ArgumentParser
from typing import Dict, List, Any

import torch
import PIL.Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# HuggingFace Transformers Comprehensive Compatibility Bridge
# In newer transformers (>=4.45), apply_chunking_to_forward was moved,
# model_parallel_utils was removed, and pruning helpers were deprecated.
# We inject them into modeling_utils, pytorch_utils, and utils.model_parallel_utils
# so LAVIS's Qformer, med, and modeling_t5 import cleanly on modern Python.
# ---------------------------------------------------------------------------
try:
    import sys, types
    import transformers.modeling_utils
    import transformers.pytorch_utils

    if hasattr(transformers.pytorch_utils, "apply_chunking_to_forward"):
        transformers.modeling_utils.apply_chunking_to_forward = transformers.pytorch_utils.apply_chunking_to_forward
    elif hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
        transformers.pytorch_utils.apply_chunking_to_forward = transformers.modeling_utils.apply_chunking_to_forward

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    transformers.modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            b = layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    transformers.modeling_utils.prune_linear_layer = prune_linear_layer
    transformers.pytorch_utils.prune_linear_layer = prune_linear_layer

    mpu = types.ModuleType("transformers.utils.model_parallel_utils")
    mpu.assert_device_map = lambda *args, **kwargs: None
    mpu.get_device_map = lambda *args, **kwargs: None
    sys.modules["transformers.utils.model_parallel_utils"] = mpu
    if hasattr(transformers, "utils"):
        transformers.utils.model_parallel_utils = mpu

    if "transformers.file_utils" not in sys.modules:
        try:
            import transformers.file_utils
        except ModuleNotFoundError:
            fu = types.ModuleType("transformers.file_utils")
            from transformers.utils import ModelOutput
            fu.ModelOutput = ModelOutput
            sys.modules["transformers.file_utils"] = fu

    # 6. all_tied_weights_keys & _tied_weights_keys (transformers >= 5.0 compatibility)
    class TiedWeightsDescriptor:
        def __get__(self, obj, objtype=None):
            if obj is None:
                return {}
            if "_all_tied_weights_keys_storage" not in obj.__dict__:
                obj.__dict__["_all_tied_weights_keys_storage"] = {}
            return obj.__dict__["_all_tied_weights_keys_storage"]
        def __set__(self, obj, value):
            obj.__dict__["_all_tied_weights_keys_storage"] = value

    if hasattr(transformers.modeling_utils, "PreTrainedModel"):
        if not hasattr(transformers.modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
            transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = TiedWeightsDescriptor()
        if not hasattr(transformers.modeling_utils.PreTrainedModel, "_tied_weights_keys"):
            transformers.modeling_utils.PreTrainedModel._tied_weights_keys = []

    # 7. Convert _keys_to_ignore_on_load_* from list to set (transformers >= 5.0 set operations)
    if hasattr(transformers.modeling_utils.PreTrainedModel, "_adjust_missing_and_unexpected_keys"):
        _orig_adjust = transformers.modeling_utils.PreTrainedModel._adjust_missing_and_unexpected_keys
        def _patched_adjust(self, *args, **kwargs):
            if getattr(self, "_keys_to_ignore_on_load_unexpected", None) is not None:
                self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected)
                type(self)._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected)
            if getattr(self, "_keys_to_ignore_on_load_missing", None) is not None:
                self._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing)
                type(self)._keys_to_ignore_on_load_missing = set(self._keys_to_ignore_on_load_missing)
            return _orig_adjust(self, *args, **kwargs)
        transformers.modeling_utils.PreTrainedModel._adjust_missing_and_unexpected_keys = _patched_adjust

    # 8. Synchronize cls.predictions.bias with resize_token_embeddings (30522 -> 30523)
    if hasattr(transformers.modeling_utils.PreTrainedModel, "resize_token_embeddings"):
        _orig_resize_tokens = transformers.modeling_utils.PreTrainedModel.resize_token_embeddings
        def _patched_resize_tokens(self, new_num_tokens=None, pad_to_multiple_of=None):
            res = _orig_resize_tokens(self, new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)
            if new_num_tokens is not None:
                if hasattr(self, "cls") and hasattr(self.cls, "predictions") and hasattr(self.cls.predictions, "bias"):
                    pred = self.cls.predictions
                    if pred.bias.shape[0] != new_num_tokens:
                        old_bias = pred.bias
                        new_bias = torch.nn.Parameter(torch.zeros(new_num_tokens, dtype=old_bias.dtype, device=old_bias.device))
                        n = min(old_bias.shape[0], new_num_tokens)
                        new_bias.data[:n] = old_bias.data[:n]
                        pred.bias = new_bias
                        if hasattr(pred, "decoder") and hasattr(pred.decoder, "bias"):
                            pred.decoder.bias = new_bias
            return res
    # 9. get_head_mask compatibility (removed in newer transformers ModuleUtilsMixin)
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        dtype = getattr(self, "dtype", torch.float32)
        return head_mask.to(dtype=dtype)

    def get_head_mask(self, head_mask=None, num_hidden_layers=12, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    transformers.modeling_utils.PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
    transformers.modeling_utils.PreTrainedModel.get_head_mask = get_head_mask
except Exception as e:
    print("[Warning] Could not apply full transformers bridge:", e)

# Ensure OMGM repo is in path for lavis and model imports
REPO_ROOT = "/kaggle/working/OMGM"
if os.path.exists(REPO_ROOT) and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from lavis.models import load_model_and_preprocess
    from data_utils import targetpad_transform, process_images_in_parallel
    from model import WikipediaKnowledgeBase, reconstruct_wiki_sections
except ImportError:
    print("[Warning] Could not import OMGM modules directly. Make sure OMGM repo is cloned to /kaggle/working/OMGM.")


def eval_recall(candidates: List[str], ground_truth: str, top_ks: List[int]) -> Dict[int, int]:
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def load_test_csv(sample_file_path: str):
    """Loads 1000 sampled test questions from CSV."""
    with open(sample_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    print(f"Loaded {len(samples)} test samples from {sample_file_path}")
    return samples


def load_candidate_wiki_images(csv_path: str, wiki_img_prefix: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Directly loads candidate Wikipedia image metadata from candidate_wiki_images_max3.csv.
    Takes < 1 second instead of 15 minutes traversing 13 giant CSVs.
    """
    title2wikiimg = {}
    found_count = 0
    missing_count = 0

    # Possible prefix paths for resilience against nested archives
    candidate_prefixes = [
        wiki_img_prefix,
        os.path.join(wiki_img_prefix, "wiki_imgs"),
        os.path.join(wiki_img_prefix, "full"),
    ]

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            rel_path = row.get("relative_path", "").strip()
            caption = row.get("caption", "").strip()

            if not title or not rel_path:
                continue

            # Resolve actual image path on disk
            img_path = None
            for pfx in candidate_prefixes:
                test_p = os.path.join(pfx, rel_path)
                if os.path.exists(test_p):
                    img_path = test_p
                    break
            
            if img_path is None:
                # Store fallback path
                img_path = os.path.join(wiki_img_prefix, rel_path)
                missing_count += 1
            else:
                found_count += 1

            if title not in title2wikiimg:
                title2wikiimg[title] = []

            # Store image info (capped at 10 as per OMGM)
            if len(title2wikiimg[title]) < 10:
                cur_captions = [x["caption"] for x in title2wikiimg[title]]
                if caption not in cur_captions:
                    title2wikiimg[title].append({"caption": caption, "img_path": img_path})

    print(f"Loaded {len(title2wikiimg)} entity image records from {csv_path}")
    print(f"  Images verified on disk : {found_count}")
    print(f"  Images not found on disk: {missing_count}")
    return title2wikiimg


def resolve_query_image(image_id: str, query_img_dir: str) -> str:
    """Finds query image file by id (oven_*.jpg)."""
    clean_id = image_id.split("|")[0].strip()
    
    # Try exact matches and extensions
    candidates = [
        os.path.join(query_img_dir, clean_id),
        os.path.join(query_img_dir, f"{clean_id}.jpg"),
        os.path.join(query_img_dir, f"{clean_id}.jpeg"),
        os.path.join(query_img_dir, f"{clean_id}.JPEG"),
        os.path.join(query_img_dir, f"{clean_id}.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    
    # Search glob as fallback
    matches = glob.glob(os.path.join(query_img_dir, f"*{clean_id}*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Query image {image_id} not found in {query_img_dir}")


def run_step2_rerank(
    sample_file_path: str,
    step1_result_path: str,
    reranker_ckpt_path: str,
    knowledge_base_path: str,
    candidate_img_csv_path: str,
    wiki_img_path_prefix: str,
    query_img_dir: str,
    save_result_path: str,
    top_ks: List[int] = [1, 5, 10, 20],
    step1_alpha: float = 0.9,
    vis_process: int = 0,
    device: str = "cuda"
):
    print("=" * 70)
    print("OMGM STEP 2: FINE-GRAINED MULTIMODAL (IT2IT) RERANKING")
    print("=" * 70)
    print(f"Test Samples CSV     : {sample_file_path}")
    print(f"Step 1 Results JSON  : {step1_result_path}")
    print(f"Reranker Checkpoint  : {reranker_ckpt_path}")
    print(f"Knowledge Base JSON  : {knowledge_base_path}")
    print(f"Candidate Images CSV : {candidate_img_csv_path}")
    print(f"Wiki Images Prefix   : {wiki_img_path_prefix}")
    print(f"Query Images Dir     : {query_img_dir}")
    print(f"Alpha (Step 1 weight): {step1_alpha}")
    print(f"Save Result Path     : {save_result_path}")
    print("=" * 70)

    # 1. Load Samples
    samples = load_test_csv(sample_file_path)

    # 2. Load Step 1 Results
    with open(step1_result_path, "r", encoding="utf-8") as f:
        step1_result_dict = json.load(f)
    print(f"Loaded Step 1 results for {len(step1_result_dict)} queries.")

    # 3. Load Candidate Images
    title2wikiimg = load_candidate_wiki_images(candidate_img_csv_path, wiki_img_path_prefix)

    # 4. Load Wikipedia Knowledge Base
    print("Loading Wikipedia Knowledge Base (this may take ~30-60s)...")
    kb_start = time.time()
    kb = WikipediaKnowledgeBase(knowledge_base_path)
    kb_list = kb.load_knowledge_base()
    kb_dict = {entry.url: entry for entry in kb_list}
    del kb, kb_list
    print(f"Knowledge Base loaded ({len(kb_dict)} articles in {time.time() - kb_start:.1f}s).")

    # 5. Load Reranker Model
    print(f"Loading Q-Former IT2IT Reranker Model onto {device}...")
    reranker_model, vis_processors, txt_processors = load_model_and_preprocess(
        name="qformer_IT2IT_reranker", model_type="pretrain", is_eval=True, device=device
    )

    if reranker_ckpt_path and reranker_ckpt_path.lower() != "none":
        print(f"Loading checkpoint weights from: {reranker_ckpt_path}")
        checkpoint = torch.load(reranker_ckpt_path, map_location=device)
        msg = reranker_model.load_state_dict(checkpoint, strict=False)
        print(f"Checkpoint loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    else:
        print("[Warning] No reranker checkpoint provided, using default pretrained weights.")

    reranker_model = reranker_model.half()

    if vis_process == 1:
        preprocess = vis_processors["eval"]
    else:
        preprocess = targetpad_transform(1.25, 224)

    recalls = {k: 0 for k in top_ks}
    retrieval_result = {}

    print("\nStarting IT2IT Reranking Loop across 1,000 queries...")
    reranker_model.eval()
    torch.cuda.empty_cache()
    loop_start = time.time()

    for it, example in enumerate(tqdm(samples, desc="Step 2 Reranking")):
        ground_truth = example.get("wikipedia_url", "")
        dataset_name = example.get("dataset_name", "infoseek")
        data_id = example.get("data_id", f"query_{it}")

        # Retrieve Step 1 results for this query
        if data_id in step1_result_dict:
            step1_item = step1_result_dict[data_id]
        elif str(data_id) in step1_result_dict:
            step1_item = step1_result_dict[str(data_id)]
        elif f"E-VQA_{it}" in step1_result_dict:
            step1_item = step1_result_dict[f"E-VQA_{it}"]
        else:
            # Fallback to index if keyed by list
            step1_item = list(step1_result_dict.values())[it]

        step1_sim = step1_item["retrieval_similarities"]
        step1_entities = step1_item["retrieved_entities"]

        # Load Query Image & Question
        img_id = example.get("dataset_image_ids", "").split("|")[0]
        q_img_path = resolve_query_image(img_id, query_img_dir)
        query_image = PIL.Image.open(q_img_path).convert("RGB")
        ref_image = preprocess(query_image).to(device).unsqueeze(0)
        question_text = txt_processors["eval"](example.get("question", ""))

        # Build Candidate Pairs (Candidate Image + Wikipedia Section)
        IT2IT_null = []
        candidate_images = []
        candidate_sections = []
        entry2ITindex = {}

        for entry_id, title_url in enumerate(step1_entities):
            wiki_title = title_url.get("title", "")
            wiki_url = title_url.get("url", "")

            # Check image existence
            if wiki_title not in title2wikiimg or len(title2wikiimg[wiki_title]) == 0:
                IT2IT_null.append(True)
                continue

            wiki_img_path = title2wikiimg[wiki_title][0]["img_path"]
            if not os.path.exists(wiki_img_path):
                IT2IT_null.append(True)
                continue

            # Check article sections
            if wiki_url not in kb_dict:
                IT2IT_null.append(True)
                continue

            entry = kb_dict[wiki_url]
            entry_sections = reconstruct_wiki_sections(entry)
            if len(entry_sections) == 0:
                IT2IT_null.append(True)
                continue

            # Open image once and repeat for each section of this entity
            try:
                cand_img_obj = PIL.Image.open(wiki_img_path).convert("RGB")
            except Exception:
                IT2IT_null.append(True)
                continue

            candidate_images.extend([cand_img_obj] * len(entry_sections))
            processed_sections = [txt_processors["eval"](sec) for sec in entry_sections]
            candidate_sections.extend(processed_sections)
            
            entry2ITindex[entry_id] = [
                len(candidate_sections) - len(entry_sections) + i
                for i in range(len(entry_sections))
            ]
            IT2IT_null.append(False)

        # Multimodal Scoring with Q-Former
        if len(candidate_images) > 0:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                # Extract multimodal query embedding
                query_fusion_embs = reranker_model.extract_features(
                    {"image": ref_image, "text_input": question_text},
                    mode="multimodal_query",
                )["multimodal_embeds"]

                # Process candidates in safe batches (rerank_step = 32 fits comfortably in 15GB VRAM)
                rerank_step = 32
                candidate_fusion_embs_list = []
                for sp in range(0, len(candidate_images), rerank_step):
                    batch_imgs = candidate_images[sp : sp + rerank_step]
                    batch_secs = candidate_sections[sp : sp + rerank_step]
                    proc_imgs = torch.stack(process_images_in_parallel(batch_imgs, preprocess)).to(device)
                    
                    split_embs = reranker_model.extract_features(
                        {"image": proc_imgs, "text_input": batch_secs},
                        mode="multimodal_candidate",
                    )["multimodal_embeds"]
                    candidate_fusion_embs_list.append(split_embs)

                candidate_fusion_embs = torch.cat(candidate_fusion_embs_list, dim=0)

                # Token-level similarity matching: [candidate_num, 32, 32]
                sim_q2c_token2token = torch.matmul(
                    query_fusion_embs, candidate_fusion_embs.permute(0, 2, 1)
                ).squeeze()

                if sim_q2c_token2token.dim() == 2:
                    sim_q2c_token2token = sim_q2c_token2token.unsqueeze(0)

                sim_q2c_querytoken, _ = sim_q2c_token2token.max(-1)
                query_token_length = sim_q2c_querytoken.size(-1)
                sim_q2c = sim_q2c_querytoken.sum(-1) / query_token_length

                # Aggregate max section score per entity
                step2_sim = [0.0] * len(step1_sim)
                for entry_id, is_null in enumerate(IT2IT_null):
                    if is_null or entry_id not in entry2ITindex:
                        continue
                    step2_sim[entry_id] = max([sim_q2c[idx].item() for idx in entry2ITindex[entry_id]])

                # Linear score interpolation
                alpha_1 = step1_alpha
                alpha_2 = 1.0 - alpha_1
                scores = (
                    alpha_1 * torch.tensor(step1_sim, device=device)
                    + alpha_2 * torch.tensor(step2_sim, device=device)
                )
                sorted_scores, reranked_idx = torch.sort(scores, descending=True)
        else:
            # Fallback if no images found for any candidate entity
            sorted_scores = torch.tensor(step1_sim, device=device)
            reranked_idx = torch.arange(len(step1_sim), device=device)
            sim_q2c = []

        top_k_wiki = [step1_entities[i.item()]["url"] for i in reranked_idx]

        # Record top-20 section similarities
        top_20_sec_sim = []
        for i in reranked_idx:
            idx = i.item()
            if idx in entry2ITindex and len(sim_q2c) > 0:
                top_20_sec_sim.append([sim_q2c[it_idx].item() for it_idx in entry2ITindex[idx]])
            else:
                top_20_sec_sim.append([])

        retrieval_result[data_id] = {
            "retrieved_entities": [step1_entities[i.item()] for i in reranked_idx],
            "retrieval_similarities": [s.item() for s in sorted_scores],
            "sec_sim": top_20_sec_sim
        }

        # Track Recall
        q_recall = eval_recall(top_k_wiki, ground_truth, top_ks)
        for k in top_ks:
            recalls[k] += q_recall[k]

    total_time = time.time() - loop_start
    print("\n" + "=" * 70)
    print("STEP 2 EVALUATION RESULTS (1,000 InfoSeek Test Samples)")
    print(f"Total Inference Time: {total_time:.1f}s ({total_time / len(samples):.2f}s per query)")
    print("=" * 70)
    for k in top_ks:
        print(f"  Recall@{k:<2}: {recalls[k] / len(samples):.4f} ({recalls[k]}/{len(samples)})")
    print("=" * 70)

    # Save output
    os.makedirs(save_result_path, exist_ok=True)
    out_file = os.path.join(save_result_path, "result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(retrieval_result, f, indent=4)
    print(f"Saved Step 2 retrieval results to: {out_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="OMGM Step 2: IT2IT Fine-Grained Reranking on Kaggle")
    parser.add_argument("--sample_file", type=str, default="/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/infoseek_test_sample_1000.csv")
    parser.add_argument("--step1_result", type=str, default="/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/result.json")
    parser.add_argument("--reranker_ckpt_path", type=str, default="/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/reranker_model.pth/reranker_model.pth/model_23912.pth")
    parser.add_argument("--knowledge_base_path", type=str, default="/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/infoseek_100k_wiki/wiki_100_dict_v4.json")
    parser.add_argument("--candidate_img_csv", type=str, default="/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/candidate_wiki_images_max3.csv")
    parser.add_argument("--wiki_img_path_prefix", type=str, default="/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/wiki_imgs/wiki_imgs")
    parser.add_argument("--query_img_dir", type=str, default="/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds/infoseek_val_1000_images/infoseek_val_1000_images")
    parser.add_argument("--save_result_path", type=str, default="/kaggle/working/step2_results")
    parser.add_argument("--top_ks", type=str, default="1,5,10,20")
    parser.add_argument("--step1_alpha", type=float, default=0.9)
    parser.add_argument("--vis_process", type=int, default=0)
    args = parser.parse_args()

    top_ks = [int(k.strip()) for k in args.top_ks.split(",")]
    run_step2_rerank(
        sample_file_path=args.sample_file,
        step1_result_path=args.step1_result,
        reranker_ckpt_path=args.reranker_ckpt_path,
        knowledge_base_path=args.knowledge_base_path,
        candidate_img_csv_path=args.candidate_img_csv,
        wiki_img_path_prefix=args.wiki_img_path_prefix,
        query_img_dir=args.query_img_dir,
        save_result_path=args.save_result_path,
        top_ks=top_ks,
        step1_alpha=args.step1_alpha,
        vis_process=args.vis_process
    )
