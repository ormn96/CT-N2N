from tqdm import tqdm
from zipfile import ZipFile
import os


def finish(output_path, output_zip_path, download_in_colab=False):
    create_output_zip(output_path, output_zip_path)
    if download_in_colab:
        from google.colab import files
        files.download(output_zip_path)


def create_output_zip(output_path, output_zip_path):
    print("starting to create archive of output dir")
    with ZipFile(output_zip_path, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(output_path):
            for filename in tqdm(desc=f"Packing({folderName})", iterable=filenames, unit="Files"):
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, filePath)
    print("finish creating archive")
