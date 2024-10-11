import os
import urllib.request
import subprocess
import shutil
import zipfile
import tarfile

# Step 1: Create directories if they don't exist
def create_folders():
    folders = {
        "Model_Data": ["Test", "Train", "Val"],
        "Models": ["PikeBot_Models", "Pretrained", "Training_Results"],
        "Preprocessed_Data": ["Test", "Val", "Train"],
        "Generators": [],
        "Data": ["Train", "Test", "Val"],
        "stockfish": []  # Added stockfish folder
    }

    for folder, subfolders in folders.items():
        # Create main folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Create subfolders if they don't exist
        for subfolder in subfolders:
            path = os.path.join(folder, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

# Step 2: Download data files if they don't exist
def download_data():
    files_to_download = {
        "./Data/Train": "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst",
        "./Data/Val": "https://database.lichess.org/standard/lichess_db_standard_rated_2024-02.pgn.zst",
        "./Data/Test": "https://database.lichess.org/standard/lichess_db_standard_rated_2024-03.pgn.zst"
    }

    for folder, url in files_to_download.items():
        # Check if folder is empty
        if not os.listdir(folder):
            file_name = os.path.join(folder, url.split("/")[-1])
            print(f"Downloading {url} to {file_name}...")
            urllib.request.urlretrieve(url, file_name)
            print(f"Download complete: {file_name}")

# Step 3: Download and unpack stockfish binaries if not already present
def download_stockfish():
    stockfish_folder = './stockfish'
    stockfish_zip_url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip"
    stockfish_tar_url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar"

    # Check if the stockfish folder is empty
    if not os.listdir(stockfish_folder):
        # Download the zip and tar files
        stockfish_zip = os.path.join(stockfish_folder, "stockfish-windows-x86-64-avx2.zip")
        stockfish_tar = os.path.join(stockfish_folder, "stockfish-ubuntu-x86-64-avx2.tar")

        print(f"Downloading {stockfish_zip_url} to {stockfish_zip}...")
        urllib.request.urlretrieve(stockfish_zip_url, stockfish_zip)
        print(f"Download complete: {stockfish_zip}")

        print(f"Downloading {stockfish_tar_url} to {stockfish_tar}...")
        urllib.request.urlretrieve(stockfish_tar_url, stockfish_tar)
        print(f"Download complete: {stockfish_tar}")

        # Unpack zip file
        if zipfile.is_zipfile(stockfish_zip):
            print(f"Unzipping {stockfish_zip}...")
            with zipfile.ZipFile(stockfish_zip, 'r') as zip_ref:
                zip_ref.extractall(stockfish_folder)
            print(f"Unzipped {stockfish_zip}")

            # Rename the extracted folder to 'stockfish_windows'
            extracted_windows_dir = os.path.join(stockfish_folder, 'stockfish')
            renamed_windows_dir = os.path.join(stockfish_folder, 'stockfish_windows')
            if os.path.exists(extracted_windows_dir):
                os.rename(extracted_windows_dir, renamed_windows_dir)
                print(f"Renamed {extracted_windows_dir} to {renamed_windows_dir}")

        # Unpack tar file
        if tarfile.is_tarfile(stockfish_tar):
            print(f"Untarring {stockfish_tar}...")
            with tarfile.open(stockfish_tar, 'r') as tar_ref:
                tar_ref.extractall(stockfish_folder)
            print(f"Untarred {stockfish_tar}")

            # Rename the extracted folder to 'stockfish_linux'
            extracted_linux_dir = os.path.join(stockfish_folder, 'stockfish')
            renamed_linux_dir = os.path.join(stockfish_folder, 'stockfish_linux')
            if os.path.exists(extracted_linux_dir):
                os.rename(extracted_linux_dir, renamed_linux_dir)
                print(f"Renamed {extracted_linux_dir} to {renamed_linux_dir}")

        # Remove the zip and tar files after unpacking
        os.remove(stockfish_zip)
        os.remove(stockfish_tar)
        print(f"Deleted {stockfish_zip} and {stockfish_tar}")

# Step 4: Run Jupyter notebooks in sequence and overwrite them
def run_notebooks():
    notebooks = [
        "Data_reading.ipynb",
        "Preprocessing.ipynb",
        "Generator.ipynb",
        "Training.ipynb",
        "Nan_count.ipynb",
        "EDA.ipynb"
    ]

    for notebook in notebooks:
        print(f"Running {notebook} and overwriting the original file...")
        result = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", notebook], check=True)
        if result.returncode == 0:
            print(f"{notebook} completed successfully.")
        else:
            print(f"Error running {notebook}.")
            break

if __name__ == "__main__":
    create_folders()
    download_data()
    download_stockfish()  # Step to handle Stockfish download and unpacking
    run_notebooks()
