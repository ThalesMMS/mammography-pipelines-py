#
# main.py
# mammography-pipelines
#
# Entry point for the patch marking UI, guiding backups, cleanup, filters, and GUI launch.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Entry point for the patch marking UI; guides backups, cleanup, and browsing."""
import os
import traceback 
import unicodedata
import sys

from .data_manager import DataManager
from .ui_viewer import ImageViewerUI
from .utils import backup_pngs, cleanup_pngs 

def normalize_answer(answer: str) -> str:
    """Normalize yes/no answers by stripping accents for comparison."""
    return unicodedata.normalize('NFKD', answer).encode('ascii', 'ignore').decode('ascii')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    # Heuristic for project root
    project_root = os.path.abspath(os.path.join(script_dir, "../../../../../.."))
    default_archive_path = os.path.join(project_root, "archive")
    if not os.path.isdir(default_archive_path):
         project_root = os.getcwd()
         default_archive_path = os.path.join(project_root, "archive")

    print(f"Looking for the 'archive' folder at: {default_archive_path}")

    if not os.path.isdir(default_archive_path):
        print(f"ERROR: 'archive' folder not found at '{default_archive_path}'.")
        return

    try:
        dm = DataManager(archive_dir_path=default_archive_path) 
        initial_valid_folders = dm._all_valid_patient_folders 
        
        if not initial_valid_folders:
            print("No valid patient folders were found at startup. Exiting.")
            return

        # --- BACKUP STEP ---
        while True:
            answer_backup = normalize_answer(input("\nDo you want to BACK UP existing PNG files? (y/n): ").strip().lower())
            if answer_backup in ['y', 'yes', 'n', 'no']: break
            print("Invalid response.")
        if answer_backup.startswith('y'):
            backup_pngs(default_archive_path, initial_valid_folders, project_root)

        # --- CLEANUP STEP ---
        cleanup_done = False
        while True:
            answer_cleanup = normalize_answer(input("\nDo you want to DELETE all existing PNG files? (y/n): ").strip().lower())
            if answer_cleanup in ['y', 'yes', 'n', 'no']: break
            print("Invalid response.")
        if answer_cleanup.startswith('y'):
            cleanup_done = cleanup_pngs(default_archive_path, initial_valid_folders, project_root)

        # --- FILTERS FOR BROWSING ---

        # Question 1: Filter by target = 0
        target_filter_value = None # Default: no filtering by target
        while True:
            answer_target = normalize_answer(input("\nDo you want to show ONLY studies with target = 0 (normal)? (y/n): ").strip().lower())
            if answer_target in ['y', 'yes', 'n', 'no']: break
            print("Invalid response.")
        if answer_target.startswith('y'):
            target_filter_value = 0 # Apply the filter value
            print("Target filter: Only target = 0 will be shown.")
        else:
            print("Target filter: All targets (0 and 1) will be shown.")

        # Question 2: Filter by existing PNG files
        png_filter_active = False # Default: do not skip folders with PNGs
        # Only ask about skipping PNGs if cleanup did NOT take place
        if not cleanup_done: 
            while True:
                answer_filter = normalize_answer(input("\nWhile browsing, do you want to skip folders that already contain PNG annotations? (y/n): ").strip().lower())
                if answer_filter in ['y', 'yes', 'n', 'no']: break
                print("Invalid response.")
            if answer_filter.startswith('y'):
                png_filter_active = True
                print("PNG filter: Folders with existing PNGs will be skipped.")
            else:
                print("PNG filter: All folders will be shown (even if PNGs exist).")
        else:
            print("PNG filter: Disabled (cleanup was performed).")

        # Apply combined filters in the DataManager
        dm.filter_folders(
            only_folders_without_pngs=png_filter_active, 
            target_value_filter=target_filter_value
        )
        
        # 4. Launch the UI if there are folders left to browse
        if dm.get_total_navigable_folders() == 0: 
            print("\nNo folders match the selected criteria. Exiting.")
        else:
            print("\n--- Launching Graphical Interface ---")
            viewer_app = ImageViewerUI(data_manager=dm) 
            viewer_app.show()

    # ... exception handling mirrors the previous structure ...
    except SystemExit as e:
        print(f"Application terminated: {e}")
    except FileNotFoundError as e: 
        print(f"File error: {e}")
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.") 
    except Exception as e:
        print(f"Unexpected error while running the application: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
