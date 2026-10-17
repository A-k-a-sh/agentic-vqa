# ==============================================================================
# Fresh Google Colab Script: InfoSeek 1000 Query Images Downloader & Extractor
# ==============================================================================
# Workflow:
#   1. Mounts Google Drive
#   2. Downloads clean 8.3 GB master archive (BByrneLab/M2KR_Images) via fast CDN
#   3. Backs up the master 8.3 GB tar to Google Drive for future seeds
#   4. Selectively extracts ONLY your 983 required query images
#   5. Compresses the 983 images into infoseek_val_1000_images.zip (~80 MB)
#   6. Saves the clean zip directly to your Google Drive
#
# Prerequisite:
#   Upload 'required_image_ids_1000.txt' to Colab root (/content/)
# ==============================================================================

# --- [CELL 1]: Mount Google Drive & Set Paths ---
from google.colab import drive
import os
import shutil

print("Mounting Google Drive...")
drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/OMGM_Data"
os.makedirs(DRIVE_DIR, exist_ok=True)

LOCAL_TAR = "/content/infoseek_val_images.tar"
DRIVE_TAR = os.path.join(DRIVE_DIR, "infoseek_val_images.tar")
REQUIRED_IDS_FILE = "/content/required_image_ids_1000.txt"
OUTPUT_DIR = "/content/infoseek_val_1000_images"
LOCAL_ZIP = "/content/infoseek_val_1000_images.zip"
DRIVE_ZIP = os.path.join(DRIVE_DIR, "infoseek_val_1000_images.zip")

# Clean output dir to avoid any leftovers from previous runs
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Drive backup location: {DRIVE_DIR}")
print(f"Extraction directory:  {OUTPUT_DIR}")


# --- [CELL 2]: Install Dependencies & High-Speed Master Download ---
!pip install -q huggingface_hub tqdm

from huggingface_hub import hf_hub_download

# Check if 8.3 GB master archive already exists in Google Drive
if os.path.exists(DRIVE_TAR):
    print(f"Found master archive in Google Drive: {DRIVE_TAR}")
    print("Copying to Colab local SSD for high-speed extraction (~30s)...")
    shutil.copy2(DRIVE_TAR, LOCAL_TAR)
    print("Copied to local SSD!")
else:
    print("Downloading master infoseek_val_images.tar (8.3 GB) via Hugging Face CDN...")
    downloaded_path = hf_hub_download(
        repo_id="BByrneLab/M2KR_Images",
        filename="Infoseek/infoseek_val_images.tar",
        repo_type="dataset",
        local_dir="/content/"
    )
    if downloaded_path != LOCAL_TAR and os.path.exists(downloaded_path):
        shutil.move(downloaded_path, LOCAL_TAR)
    
    print(f"Download complete! Size: {os.path.getsize(LOCAL_TAR) / (1024**3):.2f} GB")
    
    # Back up master archive to Drive so you never have to download it again
    print("Backing up master archive to Google Drive...")
    shutil.copy2(LOCAL_TAR, DRIVE_TAR)
    print("Master archive backed up to Google Drive successfully!")


# --- [CELL 3]: Match & Extract ONLY the 983 Images ---
import tarfile
from tqdm import tqdm

if not os.path.exists(REQUIRED_IDS_FILE):
    raise FileNotFoundError(f"Please upload '{REQUIRED_IDS_FILE}' to Colab root (/content/) first!")

with open(REQUIRED_IDS_FILE, "r") as f:
    target_ids = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(target_ids)} target query image IDs to extract.")

found_ids = set()
print("\nExtracting matching images from master archive...")
with tarfile.open(LOCAL_TAR_PATH := LOCAL_TAR, "r:") as tar:
    for member in tqdm(tar, desc="Scanning & Extracting"):
        if not member.isfile():
            continue
        
        base_name = os.path.basename(member.name)
        img_id, ext = os.path.splitext(base_name)
        
        if img_id in target_ids:
            dest_path = os.path.join(OUTPUT_DIR, base_name)
            source = tar.extractfile(member)
            if source:
                with open(dest_path, "wb") as out_f:
                    out_f.write(source.read())
                found_ids.add(img_id)

missing_ids = target_ids - found_ids

print("\n" + "=" * 50)
print("EXTRACTION SUMMARY:")
print(f"  - Target requested: {len(target_ids)}")
print(f"  - Found & extracted: {len(found_ids)}")
print(f"  - Missing:          {len(missing_ids)}")
print("=" * 50)

if missing_ids:
    missing_file = "/content/missing_image_ids.txt"
    with open(missing_file, "w") as f:
        for m in sorted(missing_ids):
            f.write(f"{m}\n")
    print(f"Saved missing IDs to: {missing_file}")


# --- [CELL 4]: Zip Extracted Images & Save Clean Package to Drive ---
import zipfile

print(f"\nCompressing {len(found_ids)} clean images into zip...")
with zipfile.ZipFile(LOCAL_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in os.listdir(OUTPUT_DIR):
        zipf.write(os.path.join(OUTPUT_DIR, f), arcname=f)

zip_mb = os.path.getsize(LOCAL_ZIP) / (1024**2)
print(f"Zip created! Size: {zip_mb:.2f} MB")

print(f"Saving zip to Google Drive: {DRIVE_ZIP}...")
shutil.copy2(LOCAL_ZIP, DRIVE_ZIP)
print("SUCCESS! Clean package saved to Google Drive.")
print(f"\nReady for Kaggle: {DRIVE_ZIP}")
