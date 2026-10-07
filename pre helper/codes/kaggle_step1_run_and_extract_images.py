# %% [markdown]
# # Step 1: Coarse Entity Retrieval (Image-to-Summary) & Wiki Image Target Extractor
# 
# This notebook implements OMGM's **Step 1** coarse retrieval pipeline on the 1,000 InfoSeek test subset:
# 1. Encodes query images using **CLIP-ViT-L/14**.
# 2. Queries the pre-built FAISS index (`summary_llama3.index`) of 100k Wikipedia summaries.
# 3. Evaluates **Recall@1, @5, @10, @20** against ground-truth Wikipedia URLs.
# 4. Saves `result.json` with top-20 retrieved candidates per query.
# 5. Filters the 13 Wiki CSVs to extract **only the specific Wikipedia images** belonging to top-20 candidates, with fixed Wikimedia URLs ready for downloading.

# %% [markdown]
# ### Cell 1: Environment Verification & Path Configuration

# %%
import os
import re
import csv
import json
import glob
import torch
import PIL.Image
from tqdm import tqdm

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Path Configuration (Kaggle Default Dataset Paths)
# ---------------------------------------------------------------------------
BASE_DS_DIR = "/kaggle/input/datasets/alphacentauriv/agentic-vqa-ds"

# 1. 1000 sampled test questions
SAMPLE_FILE = os.path.join(BASE_DS_DIR, "infoseek_test_sample_1000.csv")

# 2. Wikipedia 100k Knowledge Base JSON
KB_PATH = os.path.join(BASE_DS_DIR, "infoseek_100k_wiki/wiki_100_dict_v4.json")

# 3. 287MB FAISS index of Wikipedia article summaries
FAISS_INDEX_PATH = os.path.join(BASE_DS_DIR, "InfoSeek_summary_faiss_index/summary_llama3.index")

# 4. Extracted InfoSeek query images (983 oven_*.jpg files)
# Check both potential unzipped nesting structures
CANDIDATE_IMG_DIRS = [
    os.path.join(BASE_DS_DIR, "infoseek_val_1000_images/infoseek_val_1000_images"),
    os.path.join(BASE_DS_DIR, "infoseek_val_1000_images"),
]
QUERY_IMG_DIR = None
for p in CANDIDATE_IMG_DIRS:
    if os.path.isdir(p) and len(glob.glob(os.path.join(p, "oven_*.*"))) > 0:
        QUERY_IMG_DIR = p
        break
if QUERY_IMG_DIR is None:
    # Fallback to the first path
    QUERY_IMG_DIR = CANDIDATE_IMG_DIRS[0]

# 5. 13 Wikipedia Image URL CSVs
WIKI_CSV_DIR = os.path.join(BASE_DS_DIR, "wiki_img_output/output")

# 6. Output directory for Step 1 results
OUTPUT_DIR = "/kaggle/working/step1_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n--- Verified Paths ---")
print(f"Sample File:       {SAMPLE_FILE} (Exists: {os.path.exists(SAMPLE_FILE)})")
print(f"Knowledge Base:    {KB_PATH} (Exists: {os.path.exists(KB_PATH)})")
print(f"FAISS Index:       {FAISS_INDEX_PATH} (Exists: {os.path.exists(FAISS_INDEX_PATH)})")
print(f"Query Images Dir:  {QUERY_IMG_DIR} (Exists: {os.path.isdir(QUERY_IMG_DIR)})")
print(f"Wiki CSVs Dir:     {WIKI_CSV_DIR} (Exists: {os.path.isdir(WIKI_CSV_DIR)})")
print(f"Output Directory:  {OUTPUT_DIR}")

assert os.path.exists(SAMPLE_FILE), f"Missing {SAMPLE_FILE}"
assert os.path.exists(KB_PATH), f"Missing {KB_PATH}"
assert os.path.exists(FAISS_INDEX_PATH), f"Missing {FAISS_INDEX_PATH}"

# %% [markdown]
# ### Cell 2: Core Data Structures & ClipRetriever (Matching OMGM Codebase)

# %%
import faiss
try:
    import faiss.contrib.torch_utils
except ImportError:
    pass
from transformers import AutoModel, AutoProcessor


class WikipediaKnowledgeBaseEntry:
    """Represents a single Wikipedia article entry in the knowledge base."""
    def __init__(self, entry_dict):
        self.title = entry_dict.get("title", "")
        self.url = entry_dict.get("url", "")
        self.image_urls = entry_dict.get("image_urls", [])
        self.image_reference_descriptions = entry_dict.get("image_reference_descriptions", [])
        self.image_section_indices = entry_dict.get("image_section_indices", [])
        self.section_titles = entry_dict.get("section_titles", [])
        self.section_texts = entry_dict.get("section_texts", [])
        self.image = {}
        self.score = {}


class ClipRetriever:
    """Step 1 Image-to-Summary Retriever with CLIP ViT-Large/14."""
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda:0"):
        self.device = device
        print(f"Loading CLIP model: {model_name} on {device}...")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        # Avoid unnecessary memory usage by deleting text tower
        if hasattr(self.model, "text_projection"):
            del self.model.text_projection
        if hasattr(self.model, "text_model"):
            del self.model.text_model
        
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.knowledge_base = []
        self.entity_faiss_index = None

    def load_knowledge_base(self, kb_path):
        print(f"Loading Knowledge Base from {kb_path}...")
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_dict = json.load(f)
        
        self.knowledge_base = [WikipediaKnowledgeBaseEntry(entry) for entry in kb_dict.values()]
        print(f"Loaded Knowledge Base with {len(self.knowledge_base):,} entries.")

    def load_entity_faiss_index(self, index_path):
        print(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        # If GPU faiss is available, move index to GPU for maximum retrieval speed
        if hasattr(faiss, "StandardGpuResources") and "cuda" in self.device:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("FAISS index loaded on GPU.")
            except Exception as e:
                print(f"Warning: GPU FAISS initialization skipped ({e}). Using CPU FAISS.")
        else:
            print("FAISS index loaded on CPU.")
        self.entity_faiss_index = index
        print(f"FAISS index ready with {self.entity_faiss_index.ntotal:,} indexed vectors.")

    @torch.no_grad()
    def I2T_faiss(self, image: PIL.Image.Image, top_k: int = 20):
        """Retrieve top-K most similar Wikipedia article summaries using query image."""
        inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device).half()
        image_score = self.model.get_image_features(inputs)
        query = image_score.float()
        query = torch.nn.functional.normalize(query, dim=-1)

        if hasattr(self.entity_faiss_index, "search"):
            # Check if index accepts PyTorch CUDA tensors or numpy arrays
            try:
                D, I = self.entity_faiss_index.search(query, top_k)
            except (TypeError, AssertionError):
                D, I = self.entity_faiss_index.search(query.cpu().numpy().astype("float32"), top_k)
        else:
            raise RuntimeError("FAISS index not loaded.")

        top_k_entries = []
        for i in range(top_k):
            idx = int(I[0][i])
            kb_entry = self.knowledge_base[idx]
            sim = float(D[0][i])
            top_k_entries.append({
                "url": kb_entry.url,
                "title": kb_entry.title,
                "knowledge_base_index": idx,
                "similarity": sim,
                "kb_entry": kb_entry,
            })
        return top_k_entries


def load_csv_data(sample_file):
    sample_list = []
    with open(sample_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            sample_list.append(row)
    return sample_list, header


def get_image_path(img_id, search_dir):
    """Finds image with .jpg, .JPEG, .png or direct filename."""
    img_id = img_id.split("|")[0].strip()
    candidates = [
        os.path.join(search_dir, f"{img_id}.jpg"),
        os.path.join(search_dir, f"{img_id}.JPEG"),
        os.path.join(search_dir, f"{img_id}.png"),
        os.path.join(search_dir, img_id),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Query image {img_id} not found in {search_dir}")


def remove_list_duplicates(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall

# %% [markdown]
# ### Cell 3: Run Step 1 Coarse Retrieval & Compute Recall@K

# %%
RETRIEVAL_TOP_K = 20
TOP_KS = [1, 5, 10, 20]

# 1. Initialize Retriever
retriever = ClipRetriever(model_name="openai/clip-vit-large-patch14", device=device)
retriever.load_knowledge_base(KB_PATH)
retriever.load_entity_faiss_index(FAISS_INDEX_PATH)

# 2. Load Sample Test Queries
sample_list, sample_header = load_csv_data(SAMPLE_FILE)
print(f"\nLoaded {len(sample_list):,} sample test questions.")

recalls = {k: 0 for k in TOP_KS}
retrieval_results = {}

print(f"\nRunning Step 1 Img2Sum Retrieval (top_k={RETRIEVAL_TOP_K})...")
for it, row in tqdm(enumerate(sample_list), total=len(sample_list), desc="Step 1 Retrieval"):
    example = {sample_header[i]: row[i] for i in range(len(sample_header))}
    ground_truth_url = example["wikipedia_url"].strip()
    data_id = example["data_id"]
    raw_img_id = example["dataset_image_ids"]

    # Load query image
    img_path = get_image_path(raw_img_id, QUERY_IMG_DIR)
    image = PIL.Image.open(img_path).convert("RGB")

    # Run I2T FAISS search
    top_k = retriever.I2T_faiss(image, top_k=RETRIEVAL_TOP_K)
    
    top_k_urls = [entry["url"] for entry in top_k]
    top_k_urls = remove_list_duplicates(top_k_urls)

    # Save retrieval result format identical to OMGM
    seen_urls = set()
    dedup_entries = []
    dedup_sims = []
    for item in top_k:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            dedup_entries.append({"url": item["url"], "title": item["title"]})
            dedup_sims.append(item["similarity"])

    retrieval_results[data_id] = {
        "question": example.get("question", ""),
        "ground_truth_url": ground_truth_url,
        "retrieved_entities": dedup_entries[:RETRIEVAL_TOP_K],
        "retrieval_similarities": dedup_sims[:RETRIEVAL_TOP_K],
    }

    # Evaluate recall
    recall = eval_recall(top_k_urls, ground_truth_url, TOP_KS)
    for k in TOP_KS:
        recalls[k] += recall[k]

total_queries = len(sample_list)
print("\n" + "="*45)
print(f"STEP 1 RETRIEVAL EVALUATION RESULTS (N={total_queries})")
print("="*45)
for k in TOP_KS:
    avg_rec = (recalls[k] / total_queries) * 100.0
    print(f"Avg Recall@{k:2d}: {avg_rec:6.2f}% ({recalls[k]}/{total_queries})")
print("="*45)

# Save result.json
result_json_path = os.path.join(OUTPUT_DIR, "result.json")
with open(result_json_path, "w", encoding="utf-8") as f:
    json.dump(retrieval_results, f, indent=2)
print(f"\nSaved Step 1 retrieval results to: {result_json_path}")

# %% [markdown]
# ### Cell 4: Extract Unique Top-20 Entities & Filter Target Wiki Images

# %%
def fix_wikimedia_url(url: str) -> str:
    """
    Transforms deprecated Wikimedia thumbnail URLs to original image URLs.
    Stripping /thumb/ and the trailing size suffix avoids HTTP 400 Bad Request.
    """
    pattern = r"(https?://upload\.wikimedia\.org/wikipedia/[^/]+)/thumb/(.+)/[^/]+$"
    match = re.match(pattern, url)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return url


# 1. Collect all unique Wikipedia titles and URLs appearing in Top-20
unique_candidate_titles = set()
unique_candidate_urls = set()

for q_id, q_data in retrieval_results.items():
    for ent in q_data["retrieved_entities"]:
        if ent.get("title"):
            unique_candidate_titles.add(ent["title"].strip())
        if ent.get("url"):
            unique_candidate_urls.add(ent["url"].strip())

print("\n--- Candidate Entities Summary ---")
print(f"Total Queries Evaluated:            {len(retrieval_results):,}")
print(f"Total Candidate Mentions:           {len(retrieval_results) * RETRIEVAL_TOP_K:,}")
print(f"Unique Wikipedia Entities in Top-20: {len(unique_candidate_titles):,}")

# 2. Scan the 13 Wiki CSVs and filter rows for our unique candidates
wiki_csv_files = sorted(glob.glob(os.path.join(WIKI_CSV_DIR, "wiki_image_url_part_*_processed.csv")))
print(f"\nFound {len(wiki_csv_files)} Wikipedia image metadata CSV files.")

matched_images = []
seen_img_paths = set()

for csv_path in tqdm(wiki_csv_files, desc="Filtering Wiki Image CSVs"):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or len(row) < 4:
                continue
            title = row[0].strip()
            raw_url = row[1].strip()
            relative_path = row[2].strip()
            caption = row[3].strip()

            if title in unique_candidate_titles:
                if relative_path not in seen_img_paths:
                    seen_img_paths.add(relative_path)
                    fixed_url = fix_wikimedia_url(raw_url)
                    matched_images.append({
                        "title": title,
                        "original_url": raw_url,
                        "download_url": fixed_url,
                        "relative_path": relative_path,
                        "caption": caption
                    })

print("\n" + "="*45)
print("FILTERED WIKI IMAGES SUMMARY")
print("="*45)
print(f"Unique Top-20 Entities:             {len(unique_candidate_titles):,}")
print(f"Total Target Wiki Images to Download: {len(matched_images):,}")
avg_imgs_per_entity = len(matched_images) / max(len(unique_candidate_titles), 1)
print(f"Average Images per Entity:          {avg_imgs_per_entity:.2f}")
# Assuming average image size ~150 KB
estimated_size_mb = (len(matched_images) * 150) / 1024
print(f"Estimated Download Size:            ~{estimated_size_mb:.1f} MB ({estimated_size_mb/1024:.2f} GB)")
print("="*45)

# Save filtered target images CSV
target_images_csv = os.path.join(OUTPUT_DIR, "candidate_wiki_images_to_download.csv")
with open(target_images_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "download_url", "relative_path", "caption", "original_url"])
    writer.writeheader()
    writer.writerows(matched_images)

print(f"\nSaved download manifest to: {target_images_csv}")

# Preview first 3 entries
print("\nPreview of first 3 target image download entries:")
for i, item in enumerate(matched_images[:3], 1):
    print(f"[{i}] Title: {item['title']}")
    print(f"    URL:   {item['download_url']}")
    print(f"    Path:  {item['relative_path']}")
    print(f"    Desc:  {item['caption'][:60]}...\n")
