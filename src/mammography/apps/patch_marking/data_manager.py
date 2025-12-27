#
# data_manager.py
# mammography-pipelines
#
# Wraps the archive folder to index patient studies, apply filters, and track navigation for patch marking.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Convenience wrapper around the archive folder to navigate studies and metadata."""

import os
import pandas as pd

class DataManager:
    """Track available patient folders, filters, and navigation state for patch marking."""
    def __init__(self, archive_dir_path: str):
        """
        Initialize the DataManager.

        Args:
            archive_dir_path (str): Path to the 'archive' directory.
        """
        self.archive_dir = archive_dir_path
        self.train_csv_path = os.path.join(self.archive_dir, "train.csv")
        
        self.patient_data_df = None
        self._all_valid_patient_folders = [] # All valid folders that have labels
        
        self.navigable_folders = [] # Folders available for navigation (can be filtered)
        self.current_folder_index = -1

        if not os.path.isdir(self.archive_dir):
            raise FileNotFoundError(f"The 'archive' directory was not found at: {self.archive_dir}")
        
        if not os.path.isfile(self.train_csv_path):
            raise FileNotFoundError(f"The 'train.csv' file was not found at: {self.train_csv_path}")

        self._load_train_data()
        self._scan_patient_folders()
        
        # Initially, every valid folder is available for navigation
        self.navigable_folders = list(self._all_valid_patient_folders)
        if self.navigable_folders:
            self.current_folder_index = 0

    def _load_train_data(self):
        """
        Load train.csv into a pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.train_csv_path, dtype={'AccessionNumber': str, 'PatientID': str})
            required_columns = ['AccessionNumber', 'PatientID', 'Laterality', 'target']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing columns in train.csv: {missing_cols}")
            df.set_index('AccessionNumber', inplace=True)
            self.patient_data_df = df
            print(f"train.csv loaded successfully. {len(self.patient_data_df)} records found.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{self.train_csv_path}' is empty.")
            self.patient_data_df = pd.DataFrame()
        except ValueError as ve:
            print(f"Value error while processing train.csv: {ve}")
            self.patient_data_df = pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error while loading '{self.train_csv_path}': {e}")
            self.patient_data_df = pd.DataFrame()

    def _scan_patient_folders(self):
        """
        Scan the 'archive' directory for valid patient folders.
        """
        if self.patient_data_df is None or self.patient_data_df.empty:
            print("Unable to inspect patient folders: train.csv data is not loaded.")
            return

        found_folders = []
        for item_name in sorted(os.listdir(self.archive_dir)):
            item_path = os.path.join(self.archive_dir, item_name)
            if os.path.isdir(item_path) and item_name in self.patient_data_df.index:
                found_folders.append(item_name)
        
        self._all_valid_patient_folders = found_folders
        if not self._all_valid_patient_folders:
            print("Warning: No valid patient folders were found.")
        else:
            print(f"Found {len(self._all_valid_patient_folders)} valid patient folders with labels.")

    def get_files_in_folder(self, accession_number: str, extension: str) -> list:
        """
        List files with a given extension inside a patient folder.

        Args:
            accession_number (str): The patient's AccessionNumber (folder name).
            extension (str): File extension (e.g., '.dcm', '.png'). Must include the dot.

        Returns:
            list: A sorted list of file names.
        """
        folder_path = os.path.join(self.archive_dir, accession_number)
        if not os.path.isdir(folder_path):
            return []
        
        files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(extension.lower())]
        return files

    def get_dicom_files(self, accession_number: str) -> list:
        """Return the list of .dcm files for a patient."""
        return self.get_files_in_folder(accession_number, '.dcm')

    def get_png_files(self, accession_number: str) -> list:
        """Return the list of .png files for a patient."""
        return self.get_files_in_folder(accession_number, '.png')

    def filter_folders(self, 
                    only_folders_without_pngs: bool, 
                    target_value_filter: int | None = None):
        """
        Filter navigable folders based on the provided criteria.

        Args:
            only_folders_without_pngs (bool): If True, keep only folders without .png files.
            target_value_filter (int | None): If not None (e.g., 0 or 1), keep only folders with the
                                              matching 'target' value. Defaults to None (no target filter).
        """
        print("\nApplying navigation filters...")
        
        # Start with every valid folder that has labels
        candidate_folders = list(self._all_valid_patient_folders)
        
        # 1. PNG filter (if requested)
        if only_folders_without_pngs:
            candidate_folders = [
                folder_name for folder_name in candidate_folders
                if not self.get_png_files(folder_name) 
            ]
            print(f"- PNG filter: {len(candidate_folders)} folder(s) remaining after checking PNGs.")
        else:
            print("- PNG filter: Disabled.")

        # 2. Target filter (if requested)
        if target_value_filter is not None:
            if self.patient_data_df is None:
                print("Warning: Cannot filter by target because train.csv data is not loaded.")
            else:
                # Ensure the 'target' column exists and is numeric
                if 'target' in self.patient_data_df.columns and pd.api.types.is_numeric_dtype(self.patient_data_df['target']):
                    original_count = len(candidate_folders)
                    candidate_folders = [
                        folder_name for folder_name in candidate_folders
                        # Access the DataFrame indexed by AccessionNumber (folder_name)
                        # and compare the value in the 'target' column
                        if folder_name in self.patient_data_df.index and 
                            int(self.patient_data_df.loc[folder_name, 'target']) == target_value_filter 
                    ]
                    print(f"- Target filter={target_value_filter}: {len(candidate_folders)} folder(s) remaining (from {original_count}).")
                else:
                    print("Warning: 'target' column not found or not numeric in the DataFrame. Target filter ignored.")
                    target_value_filter = None # Reset to avoid confusion
        else:
            print("- Target filter: Disabled.")

        # Final navigable folder list
        self.navigable_folders = candidate_folders
        
        # Reset the index after filtering
        if self.navigable_folders:
            self.current_folder_index = 0
            print(f"Total navigable folders after filtering: {len(self.navigable_folders)}")
        else:
            self.current_folder_index = -1
            print("Warning: No folders match the selected filter criteria.")

            
    def get_patient_info_from_df(self, accession_number: str) -> pd.Series | None:
        """
        Return the data (Target, Laterality, etc.) from the DataFrame for an AccessionNumber.
        """
        if self.patient_data_df is not None and accession_number in self.patient_data_df.index:
            return self.patient_data_df.loc[accession_number]
        return None

    def get_current_folder_details(self) -> dict | None:
        """
        Return a dictionary with details about the current folder.
        """
        if not self.navigable_folders or self.current_folder_index < 0 or self.current_folder_index >= len(self.navigable_folders):
            return None
            
        accession_number = self.navigable_folders[self.current_folder_index]
        folder_path = os.path.join(self.archive_dir, accession_number)
        metadata = self.get_patient_info_from_df(accession_number)
        
        if metadata is None:  # Safety check; unlikely if _scan_patient_folders worked
            print(f"Warning: Metadata not found for {accession_number} in train.csv.")
            return {
                "accession_number": accession_number,
                "folder_path": folder_path,
                "target": "N/A",
                "laterality": "N/A",
                "patient_id": "N/A",
                "dicom_files": self.get_dicom_files(accession_number),
                "png_files": self.get_png_files(accession_number)
            }

        return {
            "accession_number": accession_number,
            "folder_path": folder_path,
            "target": metadata['target'],
            "laterality": metadata['Laterality'],
            "patient_id": metadata['PatientID'],
            "dicom_files": self.get_dicom_files(accession_number),
            "png_files": self.get_png_files(accession_number)
        }

    def move_to_next_folder(self) -> bool:
        """
        Move to the next folder in the navigable list.

        Returns:
            bool: True if the move was successful, False if already at the last folder.
        """
        if not self.navigable_folders or self.current_folder_index >= len(self.navigable_folders) - 1:
            return False # Already at the last folder or none available
        self.current_folder_index += 1
        return True

    def move_to_previous_folder(self) -> bool:
        """
        Move back to the previous folder in the navigable list.

        Returns:
            bool: True if the move was successful, False if already at the first folder.
        """
        if not self.navigable_folders or self.current_folder_index <= 0:
            return False # Already at the first folder or none available
        self.current_folder_index -= 1
        return True

    def get_total_navigable_folders(self) -> int:
        return len(self.navigable_folders)

    def get_current_folder_index_display(self) -> int:
        """Return the current display index (1-based) or 0 if no folders are available."""
        if self.current_folder_index == -1:
            return 0
        return self.current_folder_index + 1

# Example usage (test only):
if __name__ == '__main__':
    project_root = os.path.dirname(os.getcwd()) 
    archive_path = os.path.join(project_root, "archive")
    print(f"Attempting to access the archive directory at: {archive_path}")

    try:
        data_manager = DataManager(archive_dir_path=archive_path)
        
        print(f"\nTotal valid folders initially: {len(data_manager._all_valid_patient_folders)}")
        print(f"Total navigable folders initially: {data_manager.get_total_navigable_folders()}")

        # Test filter (simulate user choice)
        print("\n--- Testing filter: Only folders without PNGs ---")
        data_manager.filter_folders(only_folders_without_pngs=True)
        print(f"Navigable folders after filter: {data_manager.get_total_navigable_folders()}")
        
        current_details = data_manager.get_current_folder_details()
        if current_details:
            print(f"\nCurrent Folder Details ({data_manager.get_current_folder_index_display()}/{data_manager.get_total_navigable_folders()}): {current_details['accession_number']}")
            print(f"  Target: {current_details['target']}, Laterality: {current_details['laterality']}")
            print(f"  DICOMs: {current_details['dicom_files']}")
            print(f"  PNGs: {current_details['png_files']}")
        else:
            print("\nNo folders to display after filtering.")

        # Test navigation
        if data_manager.get_total_navigable_folders() > 1:
            print("\n--- Testing navigation ---")
            if data_manager.move_to_next_folder():
                next_details = data_manager.get_current_folder_details()
                print(f"Next Folder Details ({data_manager.get_current_folder_index_display()}/{data_manager.get_total_navigable_folders()}): {next_details['accession_number']}")
            
            if data_manager.move_to_previous_folder():
                prev_details = data_manager.get_current_folder_details()
                print(f"Previous Folder Details ({data_manager.get_current_folder_index_display()}/{data_manager.get_total_navigable_folders()}): {prev_details['accession_number']}")

        # Restore to all folders for another test
        print("\n--- Testing filter: All valid folders ---")
        data_manager.filter_folders(only_folders_without_pngs=False)
        print(f"Total navigable folders: {data_manager.get_total_navigable_folders()}")
        current_details_all = data_manager.get_current_folder_details()
        if current_details_all:
            print(f"\nCurrent Folder Details ({data_manager.get_current_folder_index_display()}/{data_manager.get_total_navigable_folders()}): {current_details_all['accession_number']}")


    except FileNotFoundError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"A general error occurred: {e}")
