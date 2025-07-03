import os
import shutil
import pandas as pd

CSV_PATH = "classification_results.csv"
SOURCE_DIR = "/home/hyomin/workspace/k-digital/data"
DEST_ROOT = "sorted_images"

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    filename = row["filename"]
    predicted_class = row["predicted_class"]

    src_path = os.path.join(SOURCE_DIR, filename)
    dest_dir = os.path.join(DEST_ROOT, predicted_class)
    dest_path = os.path.join(dest_dir, os.path.basename(filename))

    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(src_path):
        continue

    shutil.copy2(src_path, dest_path)
