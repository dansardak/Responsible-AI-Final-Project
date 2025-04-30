import os
import re

def count_dalle_files_in_images(images_root='images'):
    """
    Counts files containing 'dalle' in their name within subdirectories
    of the specified images root directory, filtering for directories named
    in the format 'OBJECT in REGION'.

    Args:
        images_root (str): The path to the main images directory.

    Returns:
        dict: A dictionary where keys are subdirectory paths matching the
              pattern and values are the counts of files containing 'dalle'.
    """
    dalle_counts = {}

    # Check if the root directory exists
    if not os.path.isdir(images_root):
        print(f"Error: Directory '{images_root}' not found.")
        return dalle_counts

    # Iterate through items in the images root directory
    for item_name in os.listdir(images_root):
        item_path = os.path.join(images_root, item_name)

        # Check if it's a directory AND matches the "OBJECT in REGION" pattern
        if os.path.isdir(item_path) and " in " in item_name : # Added check for " in "
            subdir_path = item_path
            dalle_count = 0
            try:
                # List files in the subdirectory
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    # Check if it's a file and 'dalle' is in the filename (case-insensitive)
                    if os.path.isfile(file_path) and 'dalle' in filename.lower():
                        dalle_count += 1
            except OSError as e:
                print(f"Could not access files in {subdir_path}: {e}")
                continue # Skip this directory if we can't read it

            dalle_counts[subdir_path] = dalle_count

    return dalle_counts

# --- Main execution ---
if __name__ == "__main__":
    results = count_dalle_files_in_images('images')

    if results:
        # Find directories with 0 DALLE files
        zero_dalle_dirs = [subdir for subdir, count in results.items() if count == 0]
        
        if zero_dalle_dirs:
            print(f"{len(zero_dalle_dirs)} Directories with 0 DALLE files:")
            for subdir in zero_dalle_dirs:
                print(f"- {subdir}")
        else:
            print("All directories have at least one DALLE file.")
    else:
        print("No subdirectories matching the 'OBJECT in REGION' pattern found or processed in 'images'.")