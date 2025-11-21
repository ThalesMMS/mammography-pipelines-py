import pydicom

# Load the DICOM file
ds = pydicom.dcmread("src/img.dcm", force=True)

# Access image dimensions
rows = ds.Rows        # (0028,0010)
columns = ds.Columns  # (0028,0011)

print(f"Image dimensions: {columns} x {rows} pixels")
