"""
Standalone script to rename long Wikipedia image filenames on local disk.
Uses candidate_wiki_images_max3_mapping.csv to rename only files exceeding 180 bytes.

Usage:
    python rename_long_wiki_images.py --img_dir "/path/to/downloaded_images" --mapping_csv "candidate_wiki_images_max3_mapping.csv"

Example on Windows:
    python rename_long_wiki_images.py --img_dir "D:\\Wiki_Candidates" --mapping_csv "candidate_wiki_images_max3_mapping.csv"
"""

import os
import sys
import csv
import argparse

def get_platform_safe_path(p: str) -> str:
    """Handle Windows MAX_PATH (>260 chars) if running on Windows."""
    abs_p = os.path.abspath(p)
    if sys.platform.startswith("win32") and not abs_p.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_p
    return abs_p

def rename_images(img_dir: str, mapping_csv: str):
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        sys.exit(1)
        
    if not os.path.exists(mapping_csv):
        print(f"Error: Mapping CSV not found: {mapping_csv}")
        sys.exit(1)

    print(f"Loading mapping from: {mapping_csv}")
    print(f"Target image directory: {img_dir}")

    to_rename = []
    with open(mapping_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_renamed") == "1":
                to_rename.append((row["old_relative_path"], row["new_relative_path"]))

    print(f"Total candidate files to rename: {len(to_rename)}")
    
    renamed_count = 0
    already_renamed_count = 0
    not_found_count = 0

    for old_rel, new_rel in to_rename:
        old_full = os.path.join(img_dir, old_rel)
        new_full = os.path.join(img_dir, new_rel)

        safe_old = get_platform_safe_path(old_full)
        safe_new = get_platform_safe_path(new_full)

        if os.path.exists(safe_old):
            try:
                os.rename(safe_old, safe_new)
                renamed_count += 1
            except Exception as e:
                print(f"Failed to rename {old_rel} -> {new_rel}: {e}")
        elif os.path.exists(safe_new):
            already_renamed_count += 1
        else:
            not_found_count += 1

    print("\n" + "=" * 50)
    print("RENAME SUMMARY:")
    print(f"  Successfully renamed : {renamed_count}")
    print(f"  Already renamed      : {already_renamed_count}")
    print(f"  Not found on disk    : {not_found_count}")
    print(f"  Total mapped         : {len(to_rename)}")
    print("=" * 50)

    # Automated Sanity Check
    print("\nRunning post-rename sanity check...")
    run_sanity_check(img_dir)

def run_sanity_check(img_dir: str):
    max_bytes = 0
    over_limit = []
    total_files = 0

    base_abs = os.path.abspath(img_dir)
    for root, _, files in os.walk(base_abs):
        for f in files:
            if f.startswith("."):  # skip hidden files like .DS_Store
                continue
            total_files += 1
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, base_abs).replace("\\", "/")
            b_len = len(rel_path.encode("utf-8"))
            if b_len > max_bytes:
                max_bytes = b_len
            if b_len > 180:
                over_limit.append((rel_path, b_len))

    print("=" * 50)
    print("SANITY CHECK RESULTS:")
    print(f"  Total image files scanned   : {total_files}")
    print(f"  Maximum path length on disk : {max_bytes} bytes")
    print(f"  Paths exceeding 180 bytes   : {len(over_limit)}")
    
    if len(over_limit) == 0:
        print("  STATUS: [PASSED] ALL PATHS ARE SAFE!")
        print("  Kaggle's 248-byte limit will NOT be triggered.")
    else:
        print(f"  STATUS: [WARNING] {len(over_limit)} files still exceed 180 bytes:")
        for p, b in over_limit[:5]:
            print(f"    - {b} bytes: {p}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename long wiki image files to prevent Kaggle 248-byte errors.")
    parser.add_argument("--img_dir", type=str, required=True, help="Base path to downloaded images containing full/wiki_image_split/...")
    parser.add_argument("--mapping_csv", type=str, default="candidate_wiki_images_max3_mapping.csv", help="Path to mapping CSV")
    args = parser.parse_args()

    rename_images(args.img_dir, args.mapping_csv)
