# src/utils.py

import os
import shutil
import datetime
import csv
import unicodedata

# --- Helper Functions for Backup and Cleanup ---


def _normalize_choice(choice: str) -> str:
    """Convert user input to lowercase ASCII for comparisons."""
    return unicodedata.normalize('NFKD', choice).encode('ascii', 'ignore').decode('ascii')


def backup_pngs(archive_path: str, patient_folders: list, project_root: str):
    """Create a backup of existing PNG files and the annotations CSV."""
    print("\n--- Starting PNG and annotation CSV backup ---")
    backup_base_path = os.path.join(project_root, "png_backups")
    os.makedirs(backup_base_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_target_dir = os.path.join(backup_base_path, f"backup_{timestamp}")

    print(f"Creating backup at: {backup_target_dir}")
    try:
        os.makedirs(backup_target_dir)
    except OSError as e:
        print(f"Error creating backup directory '{backup_target_dir}': {e}")
        return

    annotations_csv_source_path = os.path.join(project_root, "annotations.csv")
    if os.path.exists(annotations_csv_source_path):
        annotations_csv_dest_path = os.path.join(backup_target_dir, "annotations.csv")
        try:
            shutil.copy2(annotations_csv_source_path, annotations_csv_dest_path)
            print(f"File 'annotations.csv' copied to '{backup_target_dir}'.")
        except Exception as e:
            print(f"  Error copying 'annotations.csv' to the backup: {e}")
    else:
        print("File 'annotations.csv' not found at the project root. Annotation CSV was not backed up.")

    total_files_backed_up = 0
    backed_up_in_folders = 0

    for folder_name in patient_folders:
        source_folder = os.path.join(archive_path, folder_name)
        try:
            png_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.png')]
        except FileNotFoundError:
            continue

        if png_files:
            backup_subfolder = os.path.join(backup_target_dir, folder_name)
            try:
                os.makedirs(backup_subfolder, exist_ok=True)
                copied_count_this_folder = 0
                for png_file in png_files:
                    source_file_path = os.path.join(source_folder, png_file)
                    dest_file_path = os.path.join(backup_subfolder, png_file)
                    try:
                        shutil.copy2(source_file_path, dest_file_path)
                        copied_count_this_folder += 1
                    except Exception as e:
                        print(f"  Error copying '{png_file}' from folder '{folder_name}': {e}")

                if copied_count_this_folder > 0:
                    total_files_backed_up += copied_count_this_folder
                    backed_up_in_folders += 1
            except Exception as e:
                print(f"Error processing backup for folder '{folder_name}': {e}")

    if total_files_backed_up > 0:
        print(f"PNG backup complete: {total_files_backed_up} PNG file(s) copied from {backed_up_in_folders} folder(s).")
    elif not os.path.exists(annotations_csv_source_path):
        print("No PNG files found and 'annotations.csv' does not exist to back up.")
        if not os.listdir(backup_target_dir):
            try:
                os.rmdir(backup_target_dir)
            except OSError:
                pass
    elif os.path.exists(annotations_csv_source_path) and total_files_backed_up == 0:
        print("No PNG files found for backup, but 'annotations.csv' was copied.")
    else:
        print("No PNG files found to back up.")
        if not os.listdir(backup_target_dir):
            try:
                os.rmdir(backup_target_dir)
            except OSError:
                pass


def cleanup_pngs(archive_path: str, patient_folders: list, project_root: str):
    """Remove all .png files from patient folders and reset annotations.csv."""
    print("\n--- Starting PNG cleanup and annotation CSV reset ---")
    total_files_removed = 0
    cleaned_folders = 0

    confirm = _normalize_choice(
        input("!! WARNING !! Remove ALL existing .png files in patient folders AND RESET 'annotations.csv'? (y/n): ").strip().lower()
    )
    if confirm not in ['y', 'yes']:
        print("Cleanup cancelled.")
        return False

    print("Removing PNG files...")
    for folder_name in patient_folders:
        current_folder_path = os.path.join(archive_path, folder_name)
        removed_count_this_folder = 0
        try:
            items = os.listdir(current_folder_path)
            for item in items:
                if item.lower().endswith('.png'):
                    file_path = os.path.join(current_folder_path, item)
                    try:
                        os.remove(file_path)
                        removed_count_this_folder += 1
                    except OSError as e:
                        print(f"  Error deleting '{item}' inside '{folder_name}': {e}")

            if removed_count_this_folder > 0:
                cleaned_folders += 1
                total_files_removed += removed_count_this_folder
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error during cleanup for folder '{folder_name}': {e}")

    if total_files_removed > 0:
        print(f"PNG cleanup complete: {total_files_removed} PNG file(s) removed from {cleaned_folders} folder(s).")
    else:
        print("No PNG files were found to remove.")

    annotations_csv_path = os.path.join(project_root, "annotations.csv")
    if os.path.exists(annotations_csv_path):
        try:
            header = [
                "AccessionNumber",
                "DCM_Filename",
                "Adjusted_ROI_Center_X",
                "Adjusted_ROI_Center_Y",
                "ROI_Size",
                "Saved_PNG_Filename",
            ]
            with open(annotations_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
            print(f"File '{os.path.basename(annotations_csv_path)}' was reset with the header.")
        except IOError as e:
            print(f"  Error resetting file '{os.path.basename(annotations_csv_path)}': {e}")
    else:
        print(
            f"File '{os.path.basename(annotations_csv_path)}' not found. No CSV cleanup was needed (expected if it was never created)."
        )

    return True
