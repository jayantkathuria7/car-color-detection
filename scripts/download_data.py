import kagglehub
import shutil
import os

OUTPUT_PATH = 'data/input'
# Download latest version
path = kagglehub.dataset_download("jayantkathuria/car-color-dataset", output_dir=OUTPUT_PATH)

#Deleting the helper files
folder_path = os.path.join(OUTPUT_PATH,'.complete')
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    shutil.rmtree(folder_path)
    print("Helper Folder and all its contents deleted successfully.")
else:
    print("The specified folder does not exist.")

print(f"Path to dataset files: '{path}'")