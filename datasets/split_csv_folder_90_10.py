#!/usr/bin/env python3
"""
Split a folder of CSVs into training/validation sets by a 90/10 line cutoff.

Steps:
1) Open every CSV in a specified folder and record how many lines there are in each.
2) Calculate total lines.
3) cutoff = floor(0.9 * total)  (configurable)
4) Determine which file contains the cutoff line, and which line within that file it is.
5) Create two CSVs from that cutoff-file: before_cutoff and at_or_after_cutoff.
6) Create "training data" and "validation data" folders in the parent folder; transfer files accordingly.
7) Print labeled logs of cutoff, cutoff file, and what went where.
8) End.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# =========================
# Configuration
# =========================
# Set this to the folder that contains your CSV files:
INPUT_FOLDER = r"CHANGE_ME_TO_YOUR_FOLDER_PATH"

# If True, treat the first row of each CSV as a header and do NOT count it toward "lines".
ASSUME_HEADER = True

# If True, move files into training/validation folders (originals removed).
# If False, copy files (originals remain in INPUT_FOLDER).
MOVE_FILES = False

# Use floor by default for "9/10 of total".
# If you want the cutoff to be "at least 90%" you could use math.ceil instead.
USE_CEIL_FOR_CUTOFF = False

# Output folder names (created in INPUT_FOLDER's parent directory)
TRAIN_DIR_NAME = "training data"
VAL_DIR_NAME = "validation data"


# =========================
# Data structures
# =========================
@dataclass
class CsvInfo:
    path: Path
    data_line_count: int  # number of data rows (excluding header if ASSUME_HEADER=True)


# =========================
# Helpers
# =========================
def list_csv_files(folder: Path) -> List[Path]:
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    return files


def count_data_lines(csv_path: Path, assume_header: bool) -> int:
    """
    Counts number of data rows in a CSV (excluding header if assume_header=True).
    Uses csv.reader for robustness.
    """
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        row_count = 0
        for _ in reader:
            row_count += 1

    if assume_header and row_count > 0:
        return row_count - 1
    return row_count


def compute_cutoff(total: int) -> int:
    """
    Returns the cutoff line number (1-based) within the concatenation of all data rows.
    Example: total=100 -> cutoff=90 (floor) or 90 (ceil), total=101 -> floor=90, ceil=91.
    If total == 0, cutoff == 0.
    """
    if total <= 0:
        return 0

    raw = (9 * total) / 10
    return math.ceil(raw) if USE_CEIL_FOR_CUTOFF else math.floor(raw)


def find_cutoff_location(infos: List[CsvInfo], cutoff: int) -> Tuple[int, int]:
    """
    Given cutoff as a 1-based index into the concatenated data rows,
    return (file_index, line_in_file) where:
      - file_index is index into infos
      - line_in_file is 1-based index of the data row within that file
    If cutoff == 0, returns (-1, -1).
    """
    if cutoff <= 0:
        return (-1, -1)

    running = 0
    for i, info in enumerate(infos):
        if running + info.data_line_count >= cutoff:
            line_in_file = cutoff - running  # 1-based within this file's data
            return (i, line_in_file)
        running += info.data_line_count

    # If cutoff exceeds total (shouldn't happen), return last file end.
    return (len(infos) - 1, infos[-1].data_line_count)


def split_cutoff_file(
    cutoff_file: Path,
    out_before: Path,
    out_after: Path,
    assume_header: bool,
    cutoff_line_in_file: int,
) -> Tuple[int, int]:
    """
    Splits cutoff_file into out_before and out_after based on cutoff_line_in_file
    (1-based data row index). Rows 1..(cutoff_line_in_file-1) go to out_before,
    rows cutoff_line_in_file..end go to out_after.

    Returns (before_data_rows, after_data_rows)
    """
    with cutoff_file.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        rows = list(reader)

    header = None
    data_rows = rows
    if assume_header and rows:
        header = rows[0]
        data_rows = rows[1:]

    # Guard: cutoff line within file might be 1..len(data_rows)+1 depending on totals
    # We clamp safely.
    n = len(data_rows)
    k = max(1, min(cutoff_line_in_file, n + 1))

    before = data_rows[: max(0, k - 1)]
    after = data_rows[max(0, k - 1) :]

    # Write outputs
    def write_csv(path: Path, header_row, data: List[List[str]]):
        with path.open("w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            if header_row is not None:
                w.writerow(header_row)
            w.writerows(data)

    write_csv(out_before, header, before)
    write_csv(out_after, header, after)

    return (len(before), len(after))


def ensure_clean_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dest_dir: Path, move: bool) -> Path:
    dest = dest_dir / src.name
    if move:
        return Path(shutil.move(str(src), str(dest)))
    else:
        shutil.copy2(str(src), str(dest))
        return dest


# =========================
# Main
# =========================
def main() -> None:
    input_folder = Path(INPUT_FOLDER).expanduser().resolve()
    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(f"[ERROR] INPUT_FOLDER does not exist or is not a directory: {input_folder}")

    csv_files = list_csv_files(input_folder)
    if not csv_files:
        raise SystemExit(f"[ERROR] No .csv files found in: {input_folder}")

    print("========== CSV LINE COUNTING ==========")
    infos: List[CsvInfo] = []
    for p in csv_files:
        cnt = count_data_lines(p, ASSUME_HEADER)
        infos.append(CsvInfo(path=p, data_line_count=cnt))
        print(f"[COUNT] {p.name}: data_lines={cnt}")

    total_lines = sum(i.data_line_count for i in infos)
    cutoff = compute_cutoff(total_lines)

    print("\n========== TOTALS / CUTOFF ==========")
    print(f"[TOTAL] Total data lines across all CSV files: {total_lines}")
    print(f"[CUTOFF] Cutoff fraction: 0.9")
    print(f"[CUTOFF] Cutoff method: {'ceil' if USE_CEIL_FOR_CUTOFF else 'floor'}")
    print(f"[CUTOFF] Cutoff data-line index (1-based into concatenation): {cutoff}")

    if total_lines == 0:
        raise SystemExit("[ERROR] Total data lines is 0 (nothing to split).")

    cutoff_file_idx, cutoff_line_in_file = find_cutoff_location(infos, cutoff)
    if cutoff_file_idx < 0:
        raise SystemExit("[ERROR] Cutoff computed as 0; nothing to split.")

    cutoff_info = infos[cutoff_file_idx]
    cutoff_file = cutoff_info.path

    print("\n========== CUTOFF LOCATION ==========")
    print(f"[CUTOFF] Cutoff is in file index: {cutoff_file_idx}")
    print(f"[CUTOFF] Cutoff file name: {cutoff_file.name}")
    print(f"[CUTOFF] Cutoff line within that file (data rows only, 1-based): {cutoff_line_in_file}")

    # Prepare output dirs in parent of input folder
    parent_dir = input_folder.parent
    train_dir = parent_dir / TRAIN_DIR_NAME
    val_dir = parent_dir / VAL_DIR_NAME
    ensure_clean_dir(train_dir)
    ensure_clean_dir(val_dir)

    # Create split CSVs (in input folder first, then transferred)
    before_name = cutoff_file.stem + "_before_cutoff.csv"
    after_name = cutoff_file.stem + "_at_or_after_cutoff.csv"
    before_path = input_folder / before_name
    after_path = input_folder / after_name

    before_rows, after_rows = split_cutoff_file(
        cutoff_file=cutoff_file,
        out_before=before_path,
        out_after=after_path,
        assume_header=ASSUME_HEADER,
        cutoff_line_in_file=cutoff_line_in_file,
    )

    print("\n========== FILE SPLIT OUTPUT ==========")
    print(f"[SPLIT] Wrote: {before_path.name} (data rows: {before_rows})")
    print(f"[SPLIT] Wrote: {after_path.name} (data rows: {after_rows})")

    # Decide which original files go where (excluding cutoff_file itself)
    originals_before = [info.path for info in infos[:cutoff_file_idx]]
    originals_after = [info.path for info in infos[cutoff_file_idx + 1 :]]

    # Transfer
    print("\n========== TRANSFER ==========")
    mode = "MOVE" if MOVE_FILES else "COPY"
    print(f"[TRANSFER] Mode: {mode}")
    print(f"[TRANSFER] Training dir: {train_dir}")
    print(f"[TRANSFER] Validation dir: {val_dir}")

    training_sent: List[str] = []
    validation_sent: List[str] = []

    # Transfer originals before cutoff file
    for f in originals_before:
        transfer_file(f, train_dir, MOVE_FILES)
        training_sent.append(f.name)

    # Transfer originals after cutoff file
    for f in originals_after:
        transfer_file(f, val_dir, MOVE_FILES)
        validation_sent.append(f.name)

    # Transfer generated split CSVs
    transfer_file(before_path, train_dir, MOVE_FILES)
    training_sent.append(before_path.name)

    transfer_file(after_path, val_dir, MOVE_FILES)
    validation_sent.append(after_path.name)

    # If we copied, keep generated split CSVs in input folder too unless you prefer to delete them.
    # If we moved, they're no longer in input folder.

    print("\n========== SUMMARY ==========")
    print(f"[SUMMARY] Total data lines: {total_lines}")
    print(f"[SUMMARY] Cutoff (1-based): {cutoff}")
    print(f"[SUMMARY] Cutoff file: {cutoff_file.name}")
    print(f"[SUMMARY] Cutoff line in cutoff file (data rows, 1-based): {cutoff_line_in_file}")

    print("\n[SUMMARY] Files sent to TRAINING:")
    for name in training_sent:
        print(f"  - {name}")

    print("\n[SUMMARY] Files sent to VALIDATION:")
    for name in validation_sent:
        print(f"  - {name}")

    print("\n[DONE] Program complete.")


if __name__ == "__main__":
    main()