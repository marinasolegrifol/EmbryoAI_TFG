import os, shutil, re, random
import pandas as pd

XLSX_PATH = r"C:\Users\oriol\Downloads\dataset.xlsx"
ROOT_IMAGES = r"C:\Users\oriol\Downloads\embryo_dataset\embryo_dataset"
OUT_ROOT = r"C:\Users\oriol\Desktop\excels"

TRAIN_DIR = os.path.join(OUT_ROOT, "train")
VAL_DIR   = os.path.join(OUT_ROOT, "val")
TEST_DIR  = os.path.join(OUT_ROOT, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

df = pd.read_excel(XLSX_PATH, header=None)
df = df.iloc[1:]
df.columns = ["image_path", "label"]

def parse_image(path):
    embryo = os.path.basename(os.path.dirname(path))

    match = re.search(r'run(\d+)', path.lower())
    run = int(match.group(1)) if match else -1

    return embryo, run

embryos = df["image_path"].apply(lambda x: os.path.basename(os.path.dirname(x))).unique()

embryos = list(embryos)
random.shuffle(embryos)

n = len(embryos)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_emb = embryos[:train_end]
val_emb   = embryos[train_end:val_end]
test_emb  = embryos[val_end:]

train_data = []
val_data = []
test_data = []

def extract_run(filename):
    import re
    match = re.search(r'run\s*(\d+)', filename.lower())
    if match:
        return int(match.group(1))

    match = re.search(r'run(\d+)', filename.lower())
    if match:
        return int(match.group(1))

    return None

def process_split(split_embryos, out_dir, dataset_list):

    for emb in split_embryos:

        emb_df = df[df["image_path"].str.contains(emb)]

        for _, row in emb_df.iterrows():

            old_path = row["image_path"]
            label = row["label"]

            run = extract_run(old_path)

            new_name = f"{emb}_RUN{run}.jpg"
            new_path = os.path.join(out_dir, new_name)

            shutil.copy(old_path, new_path)

            dataset_list.append([new_path, label])

process_split(train_emb, TRAIN_DIR, train_data)
process_split(val_emb, VAL_DIR, val_data)
process_split(test_emb, TEST_DIR, test_data)

pd.DataFrame(train_data, columns=["image_path","label"]).to_excel(os.path.join(OUT_ROOT,"train.xlsx"), index=False)
pd.DataFrame(val_data, columns=["image_path","label"]).to_excel(os.path.join(OUT_ROOT,"val.xlsx"), index=False)
pd.DataFrame(test_data, columns=["image_path","label"]).to_excel(os.path.join(OUT_ROOT,"test.xlsx"), index=False)

print("\n SPLIT + REFACTOR COMPLET")
print("Train:", len(train_data))
print("Val:", len(val_data))
print("Test:", len(test_data))