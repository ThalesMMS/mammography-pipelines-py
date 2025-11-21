# src/main.py
import os
import traceback
import unicodedata
import sys
from .data_manager import DataManager
from .ui_viewer import ImageViewerUI
from .utils import backup_classification_csv

def main():
    def normalize_choice(text: str) -> str:
        normalized = unicodedata.normalize('NFKD', text)
        return ''.join(ch for ch in normalized.lower() if not unicodedata.combining(ch)).strip()

    # Assuming run from repo root or installed package
    # If run as module, __file__ is inside src/mammography/apps/density_classifier/
    # Project root (where archive is) is ../../../../.. if structure is mammography/src/mammography/apps/density_classifier
    # But let's allow passing project root or inferring it.
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    # heuristic: finding 'mammography-pipelines-py' or just going up until 'archive' is found
    project_root = os.path.abspath(os.path.join(script_dir, "../../../../.."))
    
    archive_path_check = os.path.join(project_root, "archive")
    if not os.path.isdir(archive_path_check):
         # Try current dir
         project_root = os.getcwd()
         archive_path_check = os.path.join(project_root, "archive")

    print(f"Checking for the 'archive' directory at: {archive_path_check}")

    dm = None 
    if not os.path.isdir(archive_path_check):
        print(f"ERROR: 'archive' directory not found at '{archive_path_check}'.")
        return

    try:
        dm = DataManager(project_root=project_root) 
        
        if not dm._all_valid_patient_folders:
            print("No valid patient folders were found. Exiting.")
            return

        # Configuration
        while True:
            answer_backup_raw = input("\nWould you like to BACK UP the classification file? (y/n): ")
            answer_backup = normalize_choice(answer_backup_raw)
            if answer_backup in ['y', 'yes', 'n', 'no', 's', 'sim', 'nao']: break
            print("Invalid response.")
        if answer_backup.startswith(('y', 's')):
            backup_classification_csv(project_root)

        while True:
            answer_filter_raw = input("\nShow ONLY exams that have not been classified yet? (y/n): ")
            answer_filter = normalize_choice(answer_filter_raw)
            if answer_filter in ['y', 'yes', 'n', 'no', 's', 'sim', 'nao']: break
            print("Invalid response.")

        show_only_unclassified = answer_filter.startswith(('y', 's'))
        dm.filter_folders(only_unclassified=show_only_unclassified)

        if dm.get_total_navigable_folders() > 0:
            print("\n--- Launching Graphical Interface with Background Buffer ---")
            dm.start_loader() # Start background loading
            viewer_app = ImageViewerUI(data_manager=dm) 
            viewer_app.show()
        else:
            print("\nNo folders available to display. Exiting.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if dm:
            print("Shutting down loading processes...")
            dm.shutdown_loader()

if __name__ == '__main__':
    main()
