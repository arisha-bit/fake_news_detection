from pathlib import Path
from PIL import Image

DATASET_DIR = Path(
    r"backend\datasets\image_dataset\archive (1)"
)

folders = [
    DATASET_DIR / "train" / "FAKE",
    DATASET_DIR / "train" / "REAL",
    DATASET_DIR / "test" / "FAKE",
    DATASET_DIR / "test" / "REAL",
]

total = 0
corrupt = []

for folder in folders:
    for image_path in folder.iterdir():

        if not image_path.is_file():
            continue

        total += 1

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupt.append(str(image_path))

print("Total images checked:", total)
print("Corrupt images:", len(corrupt))

if corrupt:
    print("\nCorrupt files:")
    for file in corrupt[:20]:
        print(file)
else:
    print("All images are valid.")