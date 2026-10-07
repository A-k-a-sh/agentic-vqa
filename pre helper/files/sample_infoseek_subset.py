#!/usr/bin/env python3
"""
Sample a fixed, reproducible subset of InfoSeek test questions for evaluation.
"""

import argparse
import csv
import os
import random
from collections import Counter


def sample_subset(
    input_csv_path: str,
    output_csv_path: str,
    num_samples: int = 1000,
    seed: int = 42,
    save_image_ids_list: bool = True,
):
    print(f"Reading input from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found at: {input_csv_path}")

    # Read rows
    with open(input_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    total_rows = len(rows)
    print(f"Total entries in original file: {total_rows}")

    if num_samples > total_rows:
        raise ValueError(
            f"Requested {num_samples} samples, but file only has {total_rows} rows."
        )

    # Set random seed for reproducibility
    random.seed(seed)
    sampled_indices = sorted(random.sample(range(total_rows), num_samples))
    sampled_rows = [rows[i] for i in sampled_indices]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write sampled subset to CSV
    with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sampled_rows)

    print(f"\nSuccessfully wrote {num_samples} sampled entries to: {output_csv_path}")

    # Compute and display statistics
    img_col_idx = header.index("dataset_image_ids") if "dataset_image_ids" in header else -1
    title_col_idx = header.index("wikipedia_title") if "wikipedia_title" in header else -1
    qtype_col_idx = header.index("question_type") if "question_type" in header else -1

    unique_images = set()
    unique_entities = set()
    qtype_counts = Counter()

    for r in sampled_rows:
        if img_col_idx != -1:
            # Handle potential pipe-separated IDs
            primary_img_id = r[img_col_idx].split("|")[0].strip()
            unique_images.add(primary_img_id)
        if title_col_idx != -1:
            unique_entities.add(r[title_col_idx].strip())
        if qtype_col_idx != -1:
            qtype_counts[r[qtype_col_idx].strip()] += 1

    print("\n--- Sample Statistics ---")
    print(f"Total sampled questions: {num_samples}")
    print(f"Random seed used: {seed}")
    print(f"Unique query images needed: {len(unique_images)}")
    print(f"Unique Wikipedia entities covered: {len(unique_entities)}")

    if qtype_counts:
        print("\nQuestion types:")
        for qtype, count in qtype_counts.items():
            print(f"  - {qtype}: {count}")

    # Optionally save image ID list for targeted downloading
    if save_image_ids_list and img_col_idx != -1:
        img_list_path = os.path.join(output_dir, f"required_image_ids_{num_samples}.txt")
        with open(img_list_path, "w", encoding="utf-8") as f:
            for img_id in sorted(unique_images):
                f.write(f"{img_id}\n")
        print(f"\nSaved list of unique required image IDs to: {img_list_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample a subset from InfoSeek test CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default="others/files/infoseek_test_filtered.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pre helper/infoseek_test_sample_1000.csv",
        help="Path to output sampled CSV file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of samples to draw (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()
    sample_subset(
        input_csv_path=args.input,
        output_csv_path=args.output,
        num_samples=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
