from pathlib import Path
from PIL import Image
import imagehash

DATA_DIR = Path("data")

total_duplicates = 0

for folder in DATA_DIR.iterdir():

    if not folder.is_dir():
        continue

    print(f"\nChecking {folder.name}...")

    hashes = {}
    duplicates = []

    for img_path in folder.glob("*"):

        if not img_path.is_file():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            current_hash = imagehash.phash(img)

            is_duplicate = False

            for existing_hash in hashes:
                if current_hash - existing_hash <= 5:
                    is_duplicate = True
                    break

            if is_duplicate:
                duplicates.append(img_path)
            else:
                hashes[current_hash] = img_path

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    for dup in duplicates:
        try:
            dup.unlink()
        except Exception as e:
            print(f"Could not delete {dup}: {e}")

    total_duplicates += len(duplicates)

    print(
        f"{folder.name}: Removed {len(duplicates)} duplicates, "
        f"Remaining {len(hashes)} images"
    )

print(f"\nTotal duplicates removed: {total_duplicates}")