import os
import shutil


def remove_pycache(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
        for file in files:
            if file.endswith(":Zone.Identifier"):
                zone_path = os.path.join(root, file)
                os.remove(zone_path)
                print(f"Removed: {zone_path}")

if __name__ == "__main__":
    current_directory = os.getcwd()
    print(current_directory)
    remove_pycache(current_directory)