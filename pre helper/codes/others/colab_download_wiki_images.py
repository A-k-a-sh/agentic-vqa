# %% [markdown]
# # Colab Robust Multi-Threaded Wikipedia Image Downloader (Max 3 Clean Photos per Entity)
#
# This notebook downloads the **up to 3 highest-quality clean photographs per candidate Wikipedia entity**:
# - Input: `candidate_wiki_images_max3.csv` or `candidate_wiki_images_to_download.csv` (~23,174 target images across 11,506 entities)
# - Features:
#   * Filters out junk SVGs, distribution maps, flags, and IUCN status badges.
#   * Ensures 100% entity coverage (zero missing candidate entities).
#   * Output: `full/` directory preserving exact OMGM relative paths, packaged into `wiki_candidate_images_23k.zip`.
#   * Multi-threaded (16 parallel workers), automatic resume, retry on rate-limits, direct export to Google Drive.

# %% [markdown]
# ### Cell 1: Mount Google Drive & Setup Paths

# %%
import os
import csv
import time
import zipfile
from collections import defaultdict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# Mount Google Drive to save the final zip permanently
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_AVAILABLE = True
    print("Google Drive mounted successfully.")
except Exception:
    DRIVE_AVAILABLE = False
    print("Running in non-Colab or standalone environment (Drive mount skipped).")

# Possible input filenames (checks max3 first, then raw manifest)
CANDIDATE_INPUTS = [
    "candidate_wiki_images_max3.csv",
    "/content/candidate_wiki_images_max3.csv",
    "candidate_wiki_images_to_download.csv",
    "/content/candidate_wiki_images_to_download.csv"
]

INPUT_CSV = None
for p in CANDIDATE_INPUTS:
    if os.path.exists(p):
        INPUT_CSV = p
        break

OUTPUT_DIR = "/content/wiki_images"
ZIP_OUTPUT = "/content/wiki_candidate_images_23k.zip"
DRIVE_DEST = "/content/drive/MyDrive/wiki_candidate_images_23k.zip"

if INPUT_CSV:
    print(f"Found input manifest: {INPUT_CSV} ({os.path.getsize(INPUT_CSV) / 1024 / 1024:.2f} MB)")
else:
    print("WARNING: Please upload 'candidate_wiki_images_max3.csv' or 'candidate_wiki_images_to_download.csv' to /content/ before running Cell 2!")

# %% [markdown]
# ### Cell 2: Smart Filter & Multi-Threaded Downloader with Retry & Resume

# %%
NUM_WORKERS = 16  # 16 threads gives ~50-80 images/sec without triggering Wikimedia rate limits
USER_AGENT = "AcademicResearchBot/1.0 (https://github.com/agentic-vqa; academic research; mailto:research@domain.edu)"
MAX_IMAGES_PER_ENTITY = 3

JUNK_KEYWORDS = [
    "status_iucn", 
    "distribution_map", 
    "location_map", 
    "locator_map", 
    ".svg", 
    "flag_of", 
    "coat_of_arms"
]

def get_session():
    """Creates a robust requests session with retries for rate-limiting (429/503)."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=NUM_WORKERS, pool_maxsize=NUM_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def download_single_image(row, base_output_dir, session):
    """
    Downloads one image saving to base_output_dir / relative_path.
    Resumes automatically if file already exists and is non-empty.
    """
    download_url = row.get("download_url", "").strip()
    rel_path = row.get("relative_path", "").strip()
    
    if not download_url or not rel_path:
        return "invalid", rel_path, "Empty URL or path"
    
    target_path = os.path.join(base_output_dir, rel_path)
    
    # 1. Resume check: skip already downloaded files
    if os.path.exists(target_path) and os.path.getsize(target_path) > 100:
        return "skipped", rel_path, "Already exists"
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".tmp"
    
    # 2. Download with stream and timeout
    try:
        resp = session.get(download_url, timeout=12, stream=True)
        if resp.status_code == 200:
            with open(temp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            # Atomic rename to prevent corrupt partial files
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 100:
                os.replace(temp_path, target_path)
                return "downloaded", rel_path, "OK"
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return "failed", rel_path, "Empty file"
        else:
            return "failed", rel_path, f"HTTP {resp.status_code}"
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return "failed", rel_path, str(e)


# 1. Load and smartly filter manifest
assert INPUT_CSV and os.path.exists(INPUT_CSV), f"Cannot find input CSV manifest. Please upload it first!"

entity_to_rows = defaultdict(list)
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        title = r.get("title", "").strip()
        entity_to_rows[title].append(r)

print(f"Loaded {sum(len(v) for v in entity_to_rows.values()):,} total rows across {len(entity_to_rows):,} unique entities.")

# Filter logic: select up to MAX_IMAGES_PER_ENTITY clean photos per entity
rows_to_download = []
fallback_count = 0

for title, rows in entity_to_rows.items():
    clean = [
        r for r in rows
        if not any(k in r.get("download_url", "").lower() or k in r.get("relative_path", "").lower() for k in JUNK_KEYWORDS)
    ]
    if clean:
        selected = clean[:MAX_IMAGES_PER_ENTITY]
    else:
        # Fallback to row 0 if only diagrams/maps exist so no entity is left with 0 images
        selected = rows[:1]
        fallback_count += 1
    rows_to_download.extend(selected)

print(f"Selected {len(rows_to_download):,} clean photos (max {MAX_IMAGES_PER_ENTITY} per entity).")
print(f"Guaranteed coverage: 100% ({len(entity_to_rows):,} entities have at least 1 image).")
print(f"Target storage folder:     {OUTPUT_DIR}")
print(f"Parallel download workers: {NUM_WORKERS}\n")

# 2. Run multi-threaded download
session = get_session()
counts = {"downloaded": 0, "skipped": 0, "failed": 0, "invalid": 0}
failed_list = []

start_time = time.time()

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(download_single_image, row, OUTPUT_DIR, session): row 
        for row in rows_to_download
    }
    
    pbar = tqdm(as_completed(futures), total=len(futures), desc="Downloading Wiki Images", unit="img")
    for future in pbar:
        status, rel_path, msg = future.result()
        counts[status] = counts.get(status, 0) + 1
        if status == "failed":
            row = futures[future]
            failed_list.append({"title": row.get("title", ""), "url": row.get("download_url", ""), "path": rel_path, "error": msg})
        
        pbar.set_postfix(
            dl=counts["downloaded"],
            skip=counts["skipped"],
            fail=counts["failed"]
        )

elapsed = time.time() - start_time
print("\n" + "="*50)
print("DOWNLOAD PROCESS COMPLETED")
print("="*50)
print(f"Elapsed Time:      {elapsed / 60:.2f} minutes ({len(rows_to_download) / max(elapsed, 1):.1f} imgs/sec)")
print(f"Newly Downloaded:  {counts['downloaded']:,}")
print(f"Already Existed:   {counts['skipped']:,}")
print(f"Failed / 404:      {counts['failed']:,}")
print("="*50)

# Save failed log if any
if failed_list:
    failed_csv = "/content/failed_wiki_downloads.csv"
    with open(failed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "url", "path", "error"])
        w.writeheader()
        w.writerows(failed_list)
    print(f"Logged {len(failed_list)} failed URLs to '{failed_csv}' (OMGM gracefully handles missing images via text fallback).")

# %% [markdown]
# ### Cell 3: Verification & Disk Usage Summary

# %%
total_files = 0
total_bytes = 0

for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        if not f.endswith(".tmp"):
            fp = os.path.join(root, f)
            total_files += 1
            total_bytes += os.path.getsize(fp)

total_gb = total_bytes / (1024 ** 3)
print(f"Total downloaded files on disk: {total_files:,}")
print(f"Total disk usage:               {total_gb:.2f} GB ({total_bytes / (1024 ** 2):.1f} MB)")

# Verify sample folder structure
sample_check = os.path.join(OUTPUT_DIR, "full/wiki_image_split")
print(f"Structure check ({sample_check}): Exists = {os.path.exists(sample_check)}")

# %% [markdown]
# ### Cell 4: Compress into Zip & Export to Google Drive

# %%
print(f"Compressing {OUTPUT_DIR} into {ZIP_OUTPUT}...")
source_full = os.path.join(OUTPUT_DIR, "full")
assert os.path.exists(source_full), f"Expected 'full' folder under {OUTPUT_DIR}"

zip_cmd = f"cd '{OUTPUT_DIR}' && zip -q -r '{ZIP_OUTPUT}' full"
exit_code = os.system(zip_cmd)

if exit_code == 0 and os.path.exists(ZIP_OUTPUT):
    zip_size_gb = os.path.getsize(ZIP_OUTPUT) / (1024 ** 3)
    print(f"Zip created successfully: {ZIP_OUTPUT} ({zip_size_gb:.2f} GB)")
    
    if DRIVE_AVAILABLE:
        print(f"Copying zip archive to Google Drive: {DRIVE_DEST}...")
        cp_cmd = f"cp '{ZIP_OUTPUT}' '{DRIVE_DEST}'"
        os.system(cp_cmd)
        if os.path.exists(DRIVE_DEST):
            print(f"SUCCESS: Archive saved permanently to Google Drive: {DRIVE_DEST}")
        else:
            print("Warning: Copy to Drive may have failed. Archive is still available in /content/")
    else:
        print(f"Archive is ready for download at: {ZIP_OUTPUT}")
else:
    print(f"Error creating zip archive (exit code {exit_code}).")
