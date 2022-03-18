from tqdm import tqdm
from zipfile import ZipFile
import shutil
import os


def finish(output_path, output_zip_path, copy_path=None, download_in_colab=False):
    create_output_zip(output_path, output_zip_path)
    if copy_path:
        try:
            shutil.copy(output_zip_path, copy_path)
        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
        # For other errors
        except:
            print("Error occurred while copying file.")
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
                file_path = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(file_path, file_path)
    print("finish creating archive")
