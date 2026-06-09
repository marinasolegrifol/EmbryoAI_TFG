import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import timm

TRAIN_XLSX = r"C:\Users\oriol\Desktop\excels\train.xlsx"
TEST_XLSX = r"C:\Users\oriol\Desktop\excels\test.xlsx"
MODEL_PATH = r"C:\Users\oriol\PycharmProjects\EmbryoAI\mobilenet\model_mn.pth"
OUTPUT_DIR = r"C:\Users\oriol\Desktop\test_results\MobileNet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 32

classes = sorted(pd.read_excel(TRAIN_XLSX)["label"].astype(str).unique())
label2idx = {c: i for i, c in enumerate(classes)}
print("Classes (ordre d'entrenament):", classes)
print("Nombre de classes:", len(classes))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

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

        label = label2idx[self.labels[idx]]

        return img, label, self.images[idx]

dataset = EmbryoDataset(TEST_XLSX)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = timm.create_model(
    "mobilenetv3_large_100",
    pretrained=False,
    num_classes=len(classes)
).to(DEVICE)

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

print("\nRESULTAT FINAL")
print("Accuracy:", round(acc, 4))

pd.DataFrame({
    "image": img_paths,
    "true": [classes[i] for i in y_true],
    "pred": [classes[i] for i in y_pred],
}).to_excel(os.path.join(OUTPUT_DIR, "predictions_mn.xlsx"), index=False)

pd.DataFrame(report).transpose().to_excel(
    os.path.join(OUTPUT_DIR, "metrics_mn.xlsx")
)

pd.DataFrame(cm, index=classes, columns=classes).to_excel(
    os.path.join(OUTPUT_DIR, "confusion_matrix_mn.xlsx")
)

print("\nFET → Guardat a :", OUTPUT_DIR)