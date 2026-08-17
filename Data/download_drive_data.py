"""
download_drive_data.py

Automated Google Drive Downloader for Datasets and Model Checkpoints.
"""

import os
import sys
import time
import json
import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1bmBIccwiHu3yUCBgTqCKlSPnKdro-c0-"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MANIFEST_FILE = os.path.join(SCRIPT_DIR, "gdrive_manifest.json")
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, "data")
MAX_RETRIES = 5
RETRY_DELAY = 3


def get_or_create_manifest():
    if os.path.exists(MANIFEST_FILE):
        print(f"Loading existing folder structure from '{MANIFEST_FILE}'...")
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Indexing Google Drive folder structure (this may take 30-60 seconds)...")
    try:
        raw_files = gdown.download_folder(url=FOLDER_URL, skip_download=True)
    except Exception as e:
        print(f"Error fetching folder structure: {e}")
        sys.exit(1)

    file_list = []
    for f in raw_files:
        file_list.append({
            "id": f.id,
            "path": f.path,
            "local_path": f.local_path
        })

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(file_list, f, indent=2)

    print(f"Saved manifest with {len(file_list)} items to '{MANIFEST_FILE}'.")
    return file_list


def download_file_with_retry(file_id, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) == 0:
        try:
            os.remove(target_path)
        except Exception:
            pass

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return True, "already_exists"

    url = f"https://drive.google.com/uc?id={file_id}"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"      Downloading {os.path.basename(target_path)} (Attempt {attempt}/{MAX_RETRIES})...")
            result = gdown.download(
                url=url,
                output=target_path,
                quiet=False,
                resume=True
            )
            if result and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                return True, "downloaded"
        except Exception as err:
            print(f"      Attempt {attempt} failed: {err}")
            if os.path.exists(target_path) and os.path.getsize(target_path) == 0:
                try:
                    os.remove(target_path)
                except Exception:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return False, "failed"


def main():
    start_subfolder_num = 12
    if len(sys.argv) > 1:
        try:
            start_subfolder_num = int(sys.argv[1])
        except ValueError:
            print(f"Invalid start folder argument '{sys.argv[1]}'. Using default {start_subfolder_num}.")

    manifest = get_or_create_manifest()
    
    subfolders = {}
    for item in manifest:
        path = item["path"]
        parts = path.split(os.sep) if os.sep in path else path.split("/")
        
        if len(parts) > 1:
            sf_name = parts[0]
        else:
            sf_name = "root"
            
        if sf_name not in subfolders:
            subfolders[sf_name] = []
        subfolders[sf_name].append(item)

    sorted_sf_names = sorted(subfolders.keys(), key=lambda x: (int(x.split('_')[1]) if x.startswith('5K_') and x.split('_')[1].isdigit() else 999, x))
    
    total_subfolders = len(sorted_sf_names)
    print(f"\nFound {len(manifest)} total files across {total_subfolders} subfolders.")
    print(f"Starting download from subfolder index: 5K_{start_subfolder_num} and above.")
    print(f"Target directory: {OUTPUT_BASE_DIR}\n")

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for idx, sf_name in enumerate(sorted_sf_names, 1):
        sf_num = -1
        if sf_name.startswith("5K_") and sf_name.split("_")[1].isdigit():
            sf_num = int(sf_name.split("_")[1])

        if sf_num >= 0 and sf_num < start_subfolder_num:
            print(f"[{idx}/{total_subfolders}] Skipping subfolder '{sf_name}' (before start threshold {start_subfolder_num}).")
            continue

        items = subfolders[sf_name]
        print(f"[{idx}/{total_subfolders}] Processing subfolder '{sf_name}' ({len(items)} files)...")

        sf_downloaded = 0
        sf_skipped = 0
        sf_failed = 0

        for item in items:
            rel_path = item["path"]
            target_path = os.path.join(OUTPUT_BASE_DIR, "train_data_10_sensors_zsectors_ptx_model_preprocessing", rel_path)

            if item["id"] is None:
                os.makedirs(target_path, exist_ok=True)
                continue

            success, status = download_file_with_retry(item["id"], target_path)
            if success:
                if status == "already_exists":
                    sf_skipped += 1
                    total_skipped += 1
                else:
                    sf_downloaded += 1
                    total_downloaded += 1
            else:
                sf_failed += 1
                total_failed += 1
                print(f"   [ERROR] Failed to download {rel_path} after {MAX_RETRIES} attempts.")

        print(f"   Done: {sf_downloaded} downloaded, {sf_skipped} skipped, {sf_failed} failed.\n")

    print("=" * 60)
    print(f"Batch Download Finished!")
    print(f"Summary: {total_downloaded} downloaded, {total_skipped} skipped, {total_failed} failed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
