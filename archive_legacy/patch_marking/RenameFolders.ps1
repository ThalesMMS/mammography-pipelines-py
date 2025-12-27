# Set the path to the archive folder
$archivePath = ".\archive" # Assuming the script sits alongside the 'archive' folder
                          # If the script is INSIDE the 'archive' folder, use $archivePath = "."

# Set the path to the train.csv file
$trainCsvPath = Join-Path -Path $archivePath -ChildPath "train.csv"

Write-Host "Starting folder verification and renaming..."

# Ensure the archive directory exists
If (-not (Test-Path -Path $archivePath -PathType Container)) {
    Write-Error "Error: Directory '$archivePath' was not found."
    Exit
}

# Ensure train.csv exists
If (-not (Test-Path -Path $trainCsvPath -PathType Leaf)) {
    Write-Error "Error: File '$trainCsvPath' was not found."
    Exit
}

Try {
    # 1. Read train.csv and obtain AccessionNumbers with labels
    # Import-Csv keeps leading zeros by treating values as strings when not purely numeric
    $trainData = Import-Csv -Path $trainCsvPath
    if ($null -eq $trainData) {
        Write-Warning "train.csv is empty or could not be read properly."
        Exit
    }
    $accessionNumbersWithTarget = $trainData | Select-Object -ExpandProperty AccessionNumber | Sort-Object -Unique
    
    $countAccessionNumbers = $accessionNumbersWithTarget.Count
    Write-Host "Found $countAccessionNumbers AccessionNumbers with labels in train.csv."

    # 2. List every folder in the archive directory
    $foldersInArchive = Get-ChildItem -Path $archivePath -Directory

    $renamedCount = 0
    $skippedCount = 0

    Write-Host "Checking and renaming folders..."
    foreach ($folder in $foldersInArchive) {
        $folderName = $folder.Name
        
        # Skip folders already prefixed with "_" to avoid double-renaming
        if ($folderName.StartsWith("_")) {
            continue
        }

        # Only handle folders whose names are purely numeric
        if ($folderName -notmatch "^\d+$") {
            continue
        }

        # 3. Check if the folder name (AccessionNumber) is in the labeled list
        if ($accessionNumbersWithTarget -contains $folderName) {
        } else {
            $newFolderName = "_$($folderName)"
            $newFolderPath = Join-Path -Path $archivePath -ChildPath $newFolderName
            
            # Ensure the new name does not already exist
            if (Test-Path -Path $newFolderPath) {
                Write-Warning "Warning: Folder '$newFolderName' already exists under '$archivePath'. Skipping '$folderName'."
                $skippedCount++
                continue
            }

            try {
                Rename-Item -Path $folder.FullName -NewName $newFolderName -ErrorAction Stop
                Write-Host "Folder '$folderName' renamed to '$newFolderName'."
                $renamedCount++
            } catch {
                Write-Error "Error renaming folder '$folderName': $($_.Exception.Message)"
                $skippedCount++
            }
        }
    }

    Write-Host ""
    Write-Host "Process completed."
    Write-Host "Total folders renamed: $renamedCount"
    Write-Host "Total folders skipped (labeled, already prefixed, or error): $($foldersInArchive.Count - $renamedCount)"
    Write-Host "Unlabeled folders are now prefixed with '_'."

} Catch {
    Write-Error "A general error occurred: $($_.Exception.Message)"
}
