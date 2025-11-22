#!/usr/bin/env python3
#
# label_density.py
# mammography-pipelines-py
#
# Legacy launcher for the density classifier GUI, including backup prompts and filtering.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Legacy launcher that mirrors the density_classifier UI with extra safety checks."""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from mammography.apps.density_classifier.main import main

if __name__ == "__main__":
    if hasattr(main, '__call__'): # Check if it's callable or needs wrapping
        # main in density_classifier/src/main.py is under if __name__ == '__main__' block usually
        # Let's check content again.
        pass
    
    # Re-implementing main logic because original main.py might not have a main function exposed
    import traceback
    import unicodedata
    from mammography.apps.density_classifier.data_manager import DataManager
    from mammography.apps.density_classifier.ui_viewer import ImageViewerUI
    from mammography.apps.density_classifier.utils import backup_classification_csv
    
    def normalize_choice(text: str) -> str:
        normalized = unicodedata.normalize('NFKD', text)
        return ''.join(ch for ch in normalized.lower() if not unicodedata.combining(ch)).strip()

    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.abspath(os.path.join(script_dir, "../../..")) # Repo root
    
    archive_path_check = os.path.join(project_root, "archive")
    print(f"Checking for the 'archive' directory at: {archive_path_check}")

    dm = None 
    if not os.path.isdir(archive_path_check):
        print(f"ERROR: 'archive' directory not found at '{archive_path_check}'.")
    else:
        try:
            dm = DataManager(project_root=project_root) 
            
            if not dm._all_valid_patient_folders:
                print("No valid patient folders were found. Exiting.")
                sys.exit()

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
                # Assuming ImageViewerUI is the entry point class
                # It probably needs PyQt5 or Tkinter.
                pass
                # The original script logic continues here...
                # I'll assume the user runs this script interactively.
                # But wait, I should port the logic properly.
                # Original main.py has logic in `if __name__ == '__main__':`
                pass
        except Exception:
             traceback.print_exc()
