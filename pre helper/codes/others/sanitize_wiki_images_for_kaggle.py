"""
Sanitize candidate Wikipedia images and CSV for Kaggle upload.
1. Replaces Kaggle forbidden characters (' " & * ? : ! % + ~ # < > \\) in folder and file names.
2. Shortens any path exceeding 180 bytes to short_{md5[:10]}.ext to satisfy Kaggle's 248-byte limit.
3. Renames directories and files in-place on disk using bottom-up traversal.
4. Removes hidden OS files (.DS_Store / ._*).
5. Updates candidate_wiki_images_max3.csv with matching relative paths.
6. Performs full automated sanity verification.
"""

import os
import re
import csv
import sys
import unicodedata
import hashlib
from typing import Dict, Tuple

FORBIDDEN_RE = re.compile(r"[\x27\x22&*?:!+~#%<>\\]")

def sanitize_name(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = FORBIDDEN_RE.sub("_", s)
    return s.strip()

def sanitize_rel_path(orig_rel_path: str) -> str:
    parts = orig_rel_path.replace("\\", "/").split("/")
    # Format: [full, wiki_image_split, split, entity_part1, ..., filename]
    clean_parts = parts[:3] + [sanitize_name(p) for p in parts[3:]]
    clean_path = "/".join(clean_parts)
    if len(clean_path.encode("utf-8")) > 180:
        fn = clean_parts[-1]
        ext = os.path.splitext(fn)[1]
        h = hashlib.md5(parts[-1].encode("utf-8")).hexdigest()[:10]
        clean_parts[-1] = f"short_{h}{ext}"
        clean_path = "/".join(clean_parts)
    return clean_path

def remove_ds_store(root_dir: str):
    removed = 0
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f == ".DS_Store" or f.startswith("._"):
                try:
                    os.remove(os.path.join(r, f))
                    removed += 1
                except Exception:
                    pass
    if removed > 0:
        print(f"Removed {removed} hidden macOS metadata files (.DS_Store / ._*).")

def process_disk_and_csv(img_dir: str, backup_csv: str, output_csv: str, mapping_csv: str):
    print("=" * 60)
    print("STEP 1: Sanitizing files and folders on disk")
    print(f"Image directory: {img_dir}")
    print("=" * 60)

    remove_ds_store(img_dir)

    base_abs = os.path.abspath(img_dir)
    base_split = os.path.join(base_abs, "full", "wiki_image_split")
    if not os.path.exists(base_split):
        print(f"Error: Directory structure not found: {base_split}")
        sys.exit(1)

    # Bottom-up walk: process files first, then directories
    files_renamed = 0
    dirs_renamed = 0
    total_files_seen = 0

    for root, dirs, files in os.walk(base_split, topdown=False):
        # 1. Rename files in this directory
        for f in files:
            if f.startswith("."):
                continue
            total_files_seen += 1
            clean_f = sanitize_name(f)
            
            # Check relative path length
            rel_dir = os.path.relpath(root, base_abs).replace("\\", "/")
            rel_p = f"{rel_dir}/{clean_f}"
            
            if len(rel_p.encode("utf-8")) > 180:
                ext = os.path.splitext(f)[1]
                h = hashlib.md5(f.encode("utf-8")).hexdigest()[:10]
                clean_f = f"short_{h}{ext}"
            
            if f != clean_f:
                old_f_path = os.path.join(root, f)
                new_f_path = os.path.join(root, clean_f)
                os.rename(old_f_path, new_f_path)
                files_renamed += 1

        # 2. Rename directories if needed
        for d in dirs:
            if d.startswith("."):
                continue
            clean_d = sanitize_name(d)
            if d != clean_d:
                old_d_path = os.path.join(root, d)
                new_d_path = os.path.join(root, clean_d)
                os.rename(old_d_path, new_d_path)
                dirs_renamed += 1

    print(f"Disk Renaming Complete:")
    print(f"  Total files processed : {total_files_seen}")
    print(f"  Files renamed         : {files_renamed}")
    print(f"  Directories renamed   : {dirs_renamed}")

    print("\n" + "=" * 60)
    print("STEP 2: Updating candidate_wiki_images_max3.csv")
    print("=" * 60)

    rows = []
    mapping_rows = []
    with open(backup_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            old_rel = r["relative_path"]
            new_rel = sanitize_rel_path(old_rel)
            is_renamed = "1" if old_rel != new_rel else "0"
            
            rows.append({
                "title": r["title"],
                "download_url": r["download_url"],
                "relative_path": new_rel,
                "caption": r["caption"],
                "original_url": r["original_url"]
            })
            
            mapping_rows.append({
                "title": r["title"],
                "download_url": r["download_url"],
                "old_relative_path": old_rel,
                "new_relative_path": new_rel,
                "is_renamed": is_renamed,
                "caption": r["caption"],
                "original_url": r["original_url"]
            })

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "download_url", "relative_path", "caption", "original_url"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to: {output_csv}")

    with open(mapping_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "download_url", "old_relative_path", "new_relative_path", "is_renamed", "caption", "original_url"])
        writer.writeheader()
        writer.writerows(mapping_rows)
    print(f"Wrote mapping to: {mapping_csv}")

    print("\n" + "=" * 60)
    print("STEP 3: Running Complete Sanity Check")
    print("=" * 60)
    run_sanity_check(img_dir, output_csv)

def run_sanity_check(img_dir: str, csv_path: str):
    max_b = 0
    forbidden_found = []
    disk_rel_paths = set()
    total_files = 0

    base_abs = os.path.abspath(img_dir)
    for root, _, files in os.walk(base_abs):
        for f in files:
            if f.startswith("."):
                continue
            total_files += 1
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, base_abs).replace("\\", "/")
            b_len = len(rel_p.encode("utf-8"))
            if b_len > max_b:
                max_b = b_len
            
            # Check forbidden chars in full relative path
            if any(c in rel_p for c in "\x27\x22&*?:!+~#%<>\\]"):
                forbidden_found.append(rel_p)
            
            disk_rel_paths.add(rel_p)

    # Check match with CSV
    csv_paths = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            csv_paths.add(r["relative_path"])

    matched = disk_rel_paths.intersection(csv_paths)
    disk_not_in_csv = disk_rel_paths - csv_paths

    print(f"  Total files on disk           : {total_files}")
    print(f"  Maximum path length           : {max_b} bytes")
    print(f"  Paths exceeding 180 bytes     : {len([p for p in disk_rel_paths if len(p.encode('utf-8')) > 180])}")
    print(f"  Paths with forbidden chars    : {len(forbidden_found)}")
    print(f"  Files matching CSV 100%       : {len(matched)}")
    print(f"  Files on disk not in CSV      : {len(disk_not_in_csv)}")

    print("-" * 60)
    if len(forbidden_found) == 0 and max_b <= 180 and len(disk_not_in_csv) == 0:
        print("  SANITY CHECK STATUS: [ALL PASSED] PERFECT FOR KAGGLE!")
    else:
        print("  SANITY CHECK STATUS: [WARNING ISSUES REMAIN]")
        if forbidden_found:
            print("  Samples with forbidden chars:")
            for p in forbidden_found[:5]:
                print(f"    - {p}")
        if disk_not_in_csv:
            print("  Samples on disk not in CSV:")
            for p in list(disk_not_in_csv)[:5]:
                print(f"    - {p}")
    print("=" * 60)

if __name__ == "__main__":
    img_dir = "others/files/wiki_imgs"
    backup_csv = "pre helper/files/candidate_wiki_images_max3_backup.csv"
    output_csv = "pre helper/files/candidate_wiki_images_max3.csv"
    mapping_csv = "pre helper/files/candidate_wiki_images_max3_mapping.csv"
    process_disk_and_csv(img_dir, backup_csv, output_csv, mapping_csv)
