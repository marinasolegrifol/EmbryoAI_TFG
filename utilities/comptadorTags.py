import pandas as pd

TRAIN_PATH = r"C:\Users\oriol\Desktop\TFG_Marina\train.xlsx"
VAL_PATH   = r"C:\Users\oriol\Desktop\TFG_Marina\val.xlsx"
TEST_PATH  = r"C:\Users\oriol\Desktop\TFG_Marina\test.xlsx"

OUTPUT_PATH = r"C:\Users\oriol\Desktop\TFG_Marina\distribution.xlsx"

BIO_ORDER = [
    "pre",
    "tPB2",
    "tPNa",
    "tPNf",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "t9+",
    "tM",
    "tSB",
    "tB",
    "tEB",
    "tHB",
    "post"
]

df_train = pd.read_excel(TRAIN_PATH)
df_val   = pd.read_excel(VAL_PATH)
df_test  = pd.read_excel(TEST_PATH)

train_counts = df_train["label"].value_counts()
val_counts   = df_val["label"].value_counts()
test_counts  = df_test["label"].value_counts()

table = pd.DataFrame({
    "tag": BIO_ORDER,
    "train": [train_counts.get(c, 0) for c in BIO_ORDER],
    "val":   [val_counts.get(c, 0) for c in BIO_ORDER],
    "test":  [test_counts.get(c, 0) for c in BIO_ORDER],
})

table.to_excel(OUTPUT_PATH, index=False)

print(f"\nExcel exportat correctament a: {OUTPUT_PATH}")
