# Breast Density Classification Tool

This Python application streamlines the mammography density classification workflow. It displays every DICOM image from an exam in a single grid, allowing the user to classify the study quickly from the keyboard. Each decision is recorded in `classification.csv`.

The app prioritizes responsiveness by preloading upcoming exams in the background to eliminate waiting time between cases.

## How to Run

PowerShell

```
# Example using PowerShell
PS D:\dicom_workplace> .\.venv\Scripts\activate
(.venv) PS D:\dicom_workplace> python .\src\main.py
```

## Key Features

- **Full Exam Viewer:** Loads every DICOM image from a study into a dynamic grid.
- **Keyboard-Based Classification:** Press the number keys (1–4) to assign the BI-RADS density category.
- **Fast Navigation:** Automatically advances to the next exam after classification, with manual navigation available through the Up/Down arrows.
- **Classification Log:** Stores each decision in `classification.csv` with the exam identifier (`AccessionNumber`), the selected class, and a timestamp.
- **Optimized Performance:** Keeps the next exams in memory to deliver instant transitions.
- **Visual Feedback:** Shows the current exam identifier and highlights whether it was already classified.
- **Startup Options (console prompts):**
  1. **Classification Backup:** Optionally back up the existing `classification.csv` file.
  2. **Navigation Filter:** Optionally show only exams that still lack a classification.

## Project Structure

```
dicom_workplace/
│
├── archive/                     # DICOM studies and train.csv
│   ├── 002000/
│   │   ├── image1.dcm
│   │   └── ...
│   ├── ...
│   └── train.csv
│
├── backups/                     # Created for classification.csv backups
│   └── backup_classification_YYYYMMDD_HHMMSS/
│       └── classification.csv
│
├── src/                         # Application source code
│   ├── main.py                  # Entry point
│   ├── ui_viewer.py             # UI (grid viewer)
│   ├── data_manager.py          # Data logic, classification, and preload buffer
│   ├── dicom_loader.py          # DICOM loading and processing
│   └── utils.py                 # Utility helpers (backup)
│
├── .gitignore
├── requirements.txt
├── classification.csv
└── README.md
```

## Prerequisites

- Python (recommended: 3.9 or later)
- Dependencies listed in `requirements.txt`, notably:
  - `pandas`
  - `pydicom`
  - `matplotlib`
  - `numpy`
- For decompressing certain DICOM files (e.g., JPEG Lossless):
  - `pylibjpeg`
  - `pylibjpeg-libjpeg`

## Environment Setup

1. **Clone the repository (if hosted on Git):**
   ```bash
   git clone <repository_url>
   cd <project_name>
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment:**
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   To ensure support for compressed DICOM objects, also install:
   ```bash
   pip install pylibjpeg pylibjpeg-libjpeg
   ```
5. **Prepare the data:**
   - Confirm that the `archive/` folder exists at the project root.
   - Inside `archive/`, add one subfolder per exam (named with the `AccessionNumber`).
   - Ensure `train.csv` is available inside `archive/` to determine valid exams.

## Running the Application

1. Activate the virtual environment.
2. From the project root, run:
   ```bash
   python src/main.py
   ```
3. Answer the startup prompts in the console.

## Usage (Keyboard Shortcuts)

- **Keys `1`, `2`, `3`, `4`:** Classify the current exam and move to the next one.
  - `1`: Fatty
  - `2`: Mostly Fatty
  - `3`: Mostly Dense
  - `4`: Dense
- **Up/Down arrows:** Manually navigate between exams to review or adjust previous classifications.

## Troubleshooting

- **"Unable to decompress..." when loading DICOMs:** Install the optional dependencies listed in step 4 of _Environment Setup_.
- **`FileNotFoundError` for `archive` or `train.csv`:** Verify that the directory structure matches the layout described in _Project Structure_.

## Potential Improvements

- Add zoom and pan support for individual images in the grid.
- Allow manual windowing adjustments (Window Center/Width).
- Provide a graphical interface for the startup options.
- Persist the last viewed exam to resume the session.
- Display additional DICOM metadata (e.g., projection type: MLO, CC) in each subplot.
- Introduce a "review" mode where classification does not advance automatically.
