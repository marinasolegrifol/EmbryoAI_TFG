import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import timm


# -------------------------
# CONFIG
# -------------------------
TRAIN_XLSX = r"C:\Users\oriol\Desktop\excels\train.xlsx"   # 🔥 NECESARIO para reconstruir el mapeo
TEST_XLSX = r"C:\Users\oriol\Desktop\excels\test.xlsx"
MODEL_PATH = r"C:\Users\oriol\PycharmProjects\EmbryoAI\resnet\model_rn.pth"
OUTPUT_DIR = r"C:\Users\oriol\Desktop\test_results\ResNet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 32


# -------------------------
# CLASSES (FIX CRÍTICO)
# El modelo aprendió las clases en el orden sorted(unique) del TRAIN,
# que es exactamente lo que hace dataset.py. Hay que decodificar igual.
# NO usar una lista fija en orden biológico: ese era el bug del 0,03.
# -------------------------
classes = sorted(pd.read_excel(TRAIN_XLSX)["label"].astype(str).unique())
label2idx = {c: i for i, c in enumerate(classes)}
print("Clases (orden de entrenamiento):", classes)
print("Nombre de classes:", len(classes))


# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -------------------------
# DATASET
# -------------------------
class EmbryoDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_excel(path)
        self.images = self.df["image_path"].values
        self.labels = self.df["label"].astype(str).values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = transform(img)

        # 🔥 mismo mapeo que en entrenamiento
        label = label2idx[self.labels[idx]]

        return img, label, self.images[idx]


dataset = EmbryoDataset(TEST_XLSX)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# -------------------------
# MODEL
# -------------------------
model = timm.create_model(
    "resnetv2_50",
    pretrained=False,
    num_classes=len(classes)
).to(DEVICE)


# -------------------------
# LOAD CHECKPOINT
# -------------------------
state = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

new_state = {}
for k, v in state.items():
    k = k.replace("module.", "")
    k = k.replace("model.", "")
    new_state[k] = v

missing, unexpected = model.load_state_dict(new_state, strict=False)

print("Estats que falten:", len(missing))
print("Estats inesperats:", len(unexpected))

model.eval()

y_true = []
y_pred = []
img_paths = []

with torch.no_grad():
    for imgs, labels, paths in loader:
        imgs = imgs.to(DEVICE)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        img_paths.extend(paths)

acc = accuracy_score(y_true, y_pred)

all_idx = list(range(len(classes)))

report = classification_report(
    y_true,
    y_pred,
    labels=all_idx,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred, labels=all_idx)

print("\n📊 FINAL RESULTS")
print("Accuracy:", round(acc, 4))


# -------------------------
# SAVE RESULTS  (xlsx REAL con to_excel, no csv)
# Requiere openpyxl:  pip install openpyxl
# -------------------------
pd.DataFrame({
    "image": img_paths,
    "true": [classes[i] for i in y_true],
    "pred": [classes[i] for i in y_pred],
}).to_excel(os.path.join(OUTPUT_DIR, "predictions_rn.xlsx"), index=False)


pd.DataFrame(report).transpose().to_excel(
    os.path.join(OUTPUT_DIR, "metrics_rn.xlsx")
)

pd.DataFrame(cm, index=classes, columns=classes).to_excel(
    os.path.join(OUTPUT_DIR, "confusion_matrix_rn.xlsx")
)

print("\nFET → Guardat a :", OUTPUT_DIR)