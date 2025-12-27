#
# data_manager.py
# mammography-pipelines
#
# Manages exam metadata, background DICOM loading, and shared buffers for the density classifier UI.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Data and caching helpers for the density classifier desktop UI."""

import os
import pandas as pd
import threading
import time
from datetime import datetime
import multiprocessing
import sys
import numpy as np
from .dicom_loader import load_dicom_task

class DataManager:
    """Manage exam metadata and prefetch DICOM renders for the density UI."""
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.archive_dir = os.path.join(self.project_root, "archive")
        self.train_csv_path = os.path.join(self.archive_dir, "train.csv")
        self.classification_csv_path = os.path.join(self.project_root, "classification.csv")
        legacy_classification_path = os.path.join(self.project_root, "classificacao.csv")
        if not os.path.exists(self.classification_csv_path) and os.path.exists(legacy_classification_path):
            try:
                os.rename(legacy_classification_path, self.classification_csv_path)
                print("Renamed legacy 'classificacao.csv' to 'classification.csv'.")
            except OSError as exc:
                print(f"Unable to rename legacy file: {exc}. Continuing with the original name.")
                self.classification_csv_path = legacy_classification_path

        self.patient_data_df = None
        self.classifications_df = None
        self._all_valid_patient_folders = []
        self.navigable_folders = []
        self.current_folder_index = -1

        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.manager = multiprocessing.Manager()
        self.image_buffer = self.manager.dict()
        num_workers = max(1, os.cpu_count() - 2)
        self.pool = multiprocessing.Pool(processes=num_workers)
        print(f"Created worker pool with {num_workers} processes.")
        
        self.pending_jobs = {}
        self.control_thread_stop_event = threading.Event()
        self.control_thread = threading.Thread(target=self._manage_buffer_jobs, daemon=True)

        if not os.path.isdir(self.archive_dir):
            raise FileNotFoundError(f"The 'archive' directory was not found at: {self.archive_dir}")
        if not os.path.isfile(self.train_csv_path):
            raise FileNotFoundError(f"The 'train.csv' file was not found at: {self.train_csv_path}")

        self._load_train_data()
        self._scan_patient_folders()
        self._load_classifications()

        self.navigable_folders = list(self._all_valid_patient_folders)
        if self.navigable_folders:
            self.current_folder_index = 0

    def start_loader(self):
        if not self.control_thread.is_alive():
            print("Starting parallel buffer manager...")
            self.control_thread.start()

    def shutdown_loader(self):
        print("Shutting down process pool...")
        self.control_thread_stop_event.set()
        self.control_thread.join(timeout=2)
        self.pool.close()
        self.pool.join()
        self.manager.shutdown()

    def _manage_buffer_jobs(self):
        """Preload nearby exams into a multiprocessing-backed buffer."""
        while not self.control_thread_stop_event.is_set():
            if self.current_folder_index < 0:
                time.sleep(0.1)
                continue

            buffer_size = 8
            indices_to_load = {self.current_folder_index + i for i in range(buffer_size) if 0 <= self.current_folder_index + i < len(self.navigable_folders)}
            accessions_to_load = {self.navigable_folders[i] for i in indices_to_load}

            for acc, job in list(self.pending_jobs.items()):
                if job.ready():
                    del self.pending_jobs[acc]

            for accession in accessions_to_load:
                if accession not in self.image_buffer and accession not in self.pending_jobs:
                    dicom_files = self.get_dicom_files(accession)
                    if dicom_files:
                        file_path = os.path.join(self.archive_dir, accession, dicom_files[0])
                        job = self.pool.apply_async(load_dicom_task, args=(file_path, accession, self.image_buffer))
                        self.pending_jobs[accession] = job
                    else:
                        self.image_buffer[accession] = None

            for accession in list(self.image_buffer.keys()):
                if accession not in accessions_to_load and accession not in self.pending_jobs:
                    del self.image_buffer[accession]
            
            time.sleep(0.1)

    def _load_train_data(self):
        try:
            df = pd.read_csv(self.train_csv_path, dtype={'AccessionNumber': str})
            df.set_index('AccessionNumber', inplace=True)
            self.patient_data_df = df
        except Exception as e:
            print(f"Unexpected error while loading '{self.train_csv_path}': {e}")
            self.patient_data_df = pd.DataFrame()

    def _scan_patient_folders(self):
        if self.patient_data_df is None or self.patient_data_df.empty:
            return
        self._all_valid_patient_folders = [
            item for item in sorted(os.listdir(self.archive_dir))
            if os.path.isdir(os.path.join(self.archive_dir, item)) and item in self.patient_data_df.index
        ]
        print(f"Found {len(self._all_valid_patient_folders)} valid patient folders with labels.")

    def _load_classifications(self):
        csv_name = os.path.basename(self.classification_csv_path)
        try:
            self.classifications_df = pd.read_csv(self.classification_csv_path, dtype={'AccessionNumber': str})
            self.classifications_df.set_index('AccessionNumber', inplace=True)
            print(f"Loaded '{csv_name}'. {len(self.classifications_df)} exams already classified.")
        except FileNotFoundError:
            print(f"File '{csv_name}' not found. Creating a new DataFrame.")
            self.classifications_df = pd.DataFrame(columns=['Classification', 'ClassificationDate'])
            self.classifications_df.index.name = 'AccessionNumber'
        except Exception as e:
            print(f"Error while loading '{csv_name}': {e}")
            self.classifications_df = pd.DataFrame()

    def save_classification(self, accession_number: str, classification: int):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.classifications_df.loc[accession_number] = {'Classification': classification, 'ClassificationDate': now}
        try:
            self.classifications_df.to_csv(self.classification_csv_path)
            print(f"Classification for {accession_number} saved as '{classification}'.")
        except IOError as e:
            print(f"ERROR: Failed to save to '{self.classification_csv_path}': {e}")

    def get_classification(self, accession_number: str) -> int | None:
        if accession_number in self.classifications_df.index:
            return int(self.classifications_df.loc[accession_number, 'Classification'])
        return None

    def filter_folders(self, only_unclassified: bool):
        print("\nApplying navigation filters...")
        self.navigable_folders = list(self._all_valid_patient_folders)
        if only_unclassified:
            classified_exams = set(self.classifications_df.index)
            self.navigable_folders = [f for f in self.navigable_folders if f not in classified_exams]
            print(f"- Filter 'Only unclassified': {len(self.navigable_folders)} folders remaining.")
        else:
            print("- Filter 'Only unclassified': Disabled.")

        # --- REMOVED PREVIOUS 1000-FOLDER LIMIT ---

        if self.navigable_folders:
            self.current_folder_index = 0
        else:
            self.current_folder_index = -1
        print(f"Total folders in this session: {len(self.navigable_folders)}")

    def get_dicom_files(self, accession_number: str) -> list:
        folder_path = os.path.join(self.archive_dir, accession_number)
        if not os.path.isdir(folder_path):
            return []
        return [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith('.dcm')]

    def get_current_folder_details(self) -> dict | None:
        if not self.navigable_folders or self.current_folder_index < 0:
            return None
        return {"accession_number": self.navigable_folders[self.current_folder_index]}

    def get_exam_data_from_buffer(self, accession_number: str) -> np.ndarray | None:
        timeout, start_time = 10, time.time()
        while accession_number not in self.image_buffer:
            if time.time() - start_time > timeout:
                print(f"ERROR: Timed out while waiting for exam '{accession_number}' in the buffer.")
                return None
            time.sleep(0.05)
        return self.image_buffer.get(accession_number)

    def move_to_next_folder(self) -> bool:
        if not self.navigable_folders or self.current_folder_index >= len(self.navigable_folders) - 1:
            return False
        self.current_folder_index += 1
        return True

    def move_to_previous_folder(self) -> bool:
        if not self.navigable_folders or self.current_folder_index <= 0:
            return False
        self.current_folder_index -= 1
        return True
    
    def get_total_navigable_folders(self) -> int:
        return len(self.navigable_folders)

    def get_current_folder_index_display(self) -> int:
        if self.current_folder_index == -1:
            return 0
        return self.current_folder_index + 1
