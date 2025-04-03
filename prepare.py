import json
import pandas as pd
import os
import shutil
import argparse
from pathlib import Path
import zipfile

# Define constants for paths/filenames to avoid typos
IMAGE_SOURCE_DIR = "OCRBench_Images"
MINI_IMAGE_DIR = Path("OCRBench_Images_mini")
# --- CHANGE: Use JSON extension ---
MINI_JSON_FILE = Path("OCRBench_mini.json")
OUTPUT_ZIP_BASE = "output"  # Zip file will be output.zip


def zip_directory_and_file(directory: Path, extra_file: Path, output_zip_base: str):
    """
    Zips the specified directory and an additional file.
    The directory's contents will be placed inside a folder named after the directory
    within the zip file. The extra file will be at the root of the zip.
    """
    output_zip_path = Path(f"{output_zip_base}.zip")
    print(f"Creating zip file: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add the directory contents under a prefix folder named after the directory
        if directory.is_dir():
            zip_prefix = directory.name
            print(
                f"Adding directory '{directory}' contents under prefix '{zip_prefix}' in zip..."
            )
            for foldername, _, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = Path(foldername) / filename
                    relative_path = filepath.relative_to(directory)
                    arcname = Path(zip_prefix) / relative_path
                    zipf.write(filepath, arcname)
            print(
                f"Finished adding directory '{directory}' under prefix '{zip_prefix}'."
            )
        elif directory.exists():
            print(
                f"Warning: '{directory}' exists but is not a directory. Skipping directory add."
            )
        else:
            print(f"Info: Directory '{directory}' not found. Skipping directory add.")

        # Add the extra file to the root of the zip
        if extra_file.is_file():
            print(f"Adding file '{extra_file}' to zip root...")
            zipf.write(extra_file, extra_file.name)
            print(f"Finished adding file '{extra_file}'.")
        else:
            print(f"Warning: File '{extra_file}' not found. Skipping file add.")
    print(f"Zip file '{output_zip_path}' created successfully.")


def create_mini_dataset(
    df: pd.DataFrame,
    cat_counts: dict,
    save_images: bool,
    image_source_dir: str,
    mini_image_dir: Path,
    # --- CHANGE: Accept JSON file path ---
    mini_json_file: Path,
) -> pd.DataFrame:
    """
    Creates a mini dataset by selecting samples based on user-defined sizes,
    saves metadata to JSON, and optionally copies the images.
    """
    print("Creating mini dataset sample...")
    # Ensure source dir exists if we need to copy images
    if save_images and not Path(image_source_dir).is_dir():
        raise FileNotFoundError(
            f"Image source directory '{image_source_dir}' not found. Cannot copy images."
        )

    # Filter and sample data for the mini dataset
    mini_df_list = []
    for cat, count in cat_counts.items():
        cat_df = df[df["type"] == cat]
        if not cat_df.empty:
            n_samples = min(count, len(cat_df))
            if n_samples > 0:
                print(f"  Sampling {n_samples} items for category '{cat}'")
                mini_df_list.append(cat_df.sample(n=n_samples, random_state=42))
            else:
                print(
                    f"  Skipping category '{cat}' (no samples requested or available)."
                )
        else:
            print(f"  Category '{cat}' not found in the dataset.")

    if not mini_df_list:
        print("Warning: No samples selected for the mini dataset.")
        mini_df = pd.DataFrame(columns=df.columns)
    else:
        mini_df = pd.concat(mini_df_list).reset_index(drop=True)

    # --- CHANGE: Save the sampled dataframe to JSON ---
    print(f"Saving mini dataset metadata to '{mini_json_file}'...")
    # Use orient='records' for a list of JSON objects, indent for readability
    mini_df.to_json(mini_json_file, orient="records", indent=4, force_ascii=False)
    print("JSON saved.")
    # --- END CHANGE ---

    # Copy images if requested
    if save_images:
        print(f"Copying selected images to '{mini_image_dir}'...")
        if mini_image_dir.exists():
            print(f"  Target directory '{mini_image_dir}' already exists. Clearing it.")
            shutil.rmtree(mini_image_dir)
        mini_image_dir.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        missing_count = 0
        for _, row in mini_df.iterrows():
            src = Path(image_source_dir) / row["image_path"]
            dst = mini_image_dir / row["image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.exists():
                shutil.copy(str(src), str(dst))
                copied_count += 1
            else:
                print(f"  Warning: Source image not found: {src}")
                missing_count += 1

        print(
            f"Image copying complete. Copied {copied_count} files"
            f"{missing_count} source files missing."
        )
    else:
        print("Skipping image copying as requested.")

    return mini_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a mini OCRBench dataset (JSON output)."
    )
    parser.add_argument(
        "--nozip",
        dest="zip",
        action="store_false",
        help="Disable zip file creation. Creates local JSON and image folder (if not --only).",
    )
    parser.set_defaults(zip=True)
    parser.add_argument(
        "--only",
        action="store_true",
        help="Only keep the final zip file.",
    )
    args = parser.parse_args()

    # --- Parameter Definitions ---
    json_file = Path("OCRBench.json")

    cat_num = {
        "Scene Text-centric VQA": 100,
        "Doc-oriented VQA": 100,
        "Key Information Extraction": ,
        "Handwritten Mathematical Expression Recognition": 1,
        "Irregular Text Recognition": 1,
        "Regular Text Recognition": 1,
        "Non-Semantic Text Recognition": 1,
        "Digit String Recognition": 1,
        "Handwriting Recognition": 1,
        "Artistic Text Recognition": 1,
    }
    # --- End Parameter Definitions ---

    # --- Input Validation ---
    if not json_file.is_file():
        print(f"Error: Input JSON file '{json_file}' not found.")
        return

    # --- Load and Prepare Data ---
    print(f"Loading data from '{json_file}'...")
    with open(json_file, "r", encoding="utf-8") as file:  # Added encoding
        data = json.load(file)
    df = pd.json_normalize(data)

    expected_columns = {
        "dataset_name": "string",
        "image_path": "string",
        "type": "string",
        "question": "string",
        "answers": "object",
    }
    for col in expected_columns:
        if col not in df.columns:
            print(
                f"Warning: Column '{col}' not found in JSON data. It will be missing."
            )
    actual_types = {
        col: dtype
        for col, dtype in expected_columns.items()
        if col in df.columns and col != "answers"
    }
    df = df.astype(actual_types)
    if "answers" in df.columns:
        df["answers"] = df["answers"].apply(
            lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else [])
        )
    else:
        print("Warning: 'answers' column not found.")
    print("Data loaded and prepared.")
    # --- End Load and Prepare Data ---

    # --- Determine Actions ---
    should_create_local_images = args.zip or not args.only

    # --- Create Mini Dataset (JSON and optionally Images) ---
    try:
        create_mini_dataset(
            df,
            cat_num,
            save_images=should_create_local_images,
            image_source_dir=IMAGE_SOURCE_DIR,
            mini_image_dir=MINI_IMAGE_DIR,
            # --- CHANGE: Pass JSON file path ---
            mini_json_file=MINI_JSON_FILE,
        )
    except FileNotFoundError as e:
        print(f"Error during dataset creation: {e}")
        return

    # --- Zip Files if Requested ---
    if args.zip:
        # --- CHANGE: Pass JSON file to zip ---
        zip_directory_and_file(MINI_IMAGE_DIR, MINI_JSON_FILE, OUTPUT_ZIP_BASE)

        # --- Cleanup if --only is specified ---
        if args.only:
            print("Cleaning up intermediate files (--only specified)...")
            if MINI_IMAGE_DIR.exists():
                try:
                    shutil.rmtree(MINI_IMAGE_DIR)
                    print(f"Removed directory: {MINI_IMAGE_DIR}")
                except OSError as e:
                    print(f"Error removing directory {MINI_IMAGE_DIR}: {e}")
            # --- CHANGE: Remove JSON file ---
            if MINI_JSON_FILE.exists():
                try:
                    os.remove(MINI_JSON_FILE)
                    print(f"Removed file: {MINI_JSON_FILE}")
                except OSError as e:
                    print(f"Error removing file {MINI_JSON_FILE}: {e}")
            print("Cleanup complete.")
    else:
        print("Skipping zip file creation as requested (--nozip).")

    print("Script finished.")


if __name__ == "__main__":
    main()

"""
Usage Scenarios & Expected Outcomes: (JSON output)

1. python prepare_json.py
   - Creates: OCRBench_mini.json, OCRBench_Images_mini/ (with images)
   - Zips: Creates output.zip containing OCRBench_mini.json (at root)
     and an OCRBench_Images_mini/ folder (containing images).
   - Cleanup: None. Keeps the JSON, image folder, and zip file.

2. python prepare_json.py --nozip
   - Creates: OCRBench_mini.json, OCRBench_Images_mini/ (with images)
   - Zips: No zip file is created.
   - Cleanup: None. Keeps the JSON and image folder.

3. python prepare_json.py --only
   - Creates: OCRBench_mini.json, OCRBench_Images_mini/ (with images)
   - Zips: Creates output.zip containing OCRBench_mini.json (at root) and an
     OCRBench_Images_mini/ folder (containing images).
   - Cleanup: Deletes OCRBench_mini.json and OCRBench_Images_mini/ folder *after* zipping.

4. python prepare_json.py --only --nozip
   - Creates: OCRBench_mini.json ONLY. (No images folder is created).
   - Zips: No zip file is created.
   - Cleanup: None. (The cleanup only happens after successful zipping). Keeps the JSON file.
"""
