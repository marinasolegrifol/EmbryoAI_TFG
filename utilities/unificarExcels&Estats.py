import os, re
import pandas as pd

ROOT_IMAGES = r"C:\Users\oriol\Downloads\embryo_dataset\embryo_dataset"
ROOT_CSV = r"C:\Users\oriol\Downloads\embryo_dataset_annotations\embryo_dataset_annotations"
OUTPUT_PATH = r"C:\Users\oriol\Downloads\dataset.xlsx" #Excel resultant de l'execució

samples = []

def extract_run(filename):
    match = re.search(r'run(\d+)', filename.lower())
    return int(match.group(1)) if match else None

# Fa servir el número de run per ordenar
def sort_key(filename):
    run = extract_run(filename)
    return run if run is not None else 0

for embryo_folder in os.listdir(ROOT_IMAGES):

    img_folder = os.path.join(ROOT_IMAGES, embryo_folder)

    if not os.path.isdir(img_folder):
        continue

    print(f"\n Embrió: {embryo_folder}")

    csv_name = embryo_folder + "_phases.csv"
    csv_path = os.path.join(ROOT_CSV, csv_name)

    if not os.path.exists(csv_path):
        print(f"No CSV: {csv_name}")
        continue

    df = pd.read_csv(csv_path, header=None)

    intervals = []
    for _, row in df.iterrows():
        intervals.append({
            "state": row[0],
            "start": int(row[1]),
            "end": int(row[2])
        })

    max_end = max(it["end"] for it in intervals)

    imgs = sorted(
        [
            f for f in os.listdir(img_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ],
        key=sort_key
    )

    print(f"imágenes: {len(imgs)}")

    for img in imgs:

        run_num = extract_run(img)

        if run_num is None:
            continue

        label = None

        for it in intervals:

            if it["start"] <= run_num <= it["end"]:
                label = it["state"]
                break

        # pre / post
        if run_num < intervals[0]["start"]:
            label = "pre"

        elif run_num > max_end:
            label = "post"

        else:
            if label is None:
                label = "unknown"

        img_path = os.path.join(img_folder, img)

        samples.append([img_path, label])

df_out = pd.DataFrame(samples, columns=["image_path", "label"])

df_out.to_excel(OUTPUT_PATH, index=False)

print("\nDATASET FINAL GENERAT")
print("Total imatges:", len(df_out))
print("\nDistribució de classes:")
print(df_out["label"].value_counts())