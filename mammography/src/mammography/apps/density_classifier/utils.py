#
# utils.py
# mammography-pipelines-py
#
# Utility helpers for backing up classification CSVs within the density classifier UI flow.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Small utility helpers used by the density classifier UI."""

import os
import shutil
import datetime

# --- Backup Helper Function ---

def backup_classification_csv(project_root: str):
    """
    Create a backup of 'classification.csv' (and handle the legacy name if present).

    Args:
        project_root (str): Path to the project root directory.
    """
    print("\n--- Starting classification CSV backup ---")
    
    candidate_files = ["classification.csv", "classificacao.csv"]
    classification_csv_path = None
    for candidate in candidate_files:
        candidate_path = os.path.join(project_root, candidate)
        if os.path.exists(candidate_path):
            classification_csv_path = candidate_path
            break

    # 1. Verify the source file exists
    if classification_csv_path is None:
        print("File 'classification.csv' not found. Nothing to back up.")
        return

    source_name = os.path.basename(classification_csv_path)

    # 2. Create the backup directory
    backup_base_path = os.path.join(project_root, "backups")
    os.makedirs(backup_base_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_target_dir = os.path.join(backup_base_path, f"backup_classification_{timestamp}")
    
    try:
        os.makedirs(backup_target_dir)
    except OSError as e:
        print(f"Error creating backup directory '{backup_target_dir}': {e}")
        return

    # 3. Copy the file
    try:
        backup_file_path = os.path.join(backup_target_dir, source_name)
        shutil.copy2(classification_csv_path, backup_file_path)
        print("Backup completed successfully!")
        print(f"File '{source_name}' copied to: {backup_target_dir}")
    except Exception as e:
        print(f"An error occurred while copying '{source_name}' to the backup: {e}")
        # Attempt to remove the empty backup directory if the copy failed
        try:
            if not os.listdir(backup_target_dir):
                os.rmdir(backup_target_dir)
        except OSError:
            pass
