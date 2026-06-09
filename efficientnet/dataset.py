from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class EmbryoDataset(Dataset):

    def __init__(self, xlsx_path, transform=None):
        self.df = pd.read_excel(xlsx_path)

        self.classes = sorted(self.df["label"].unique())
        self.label2idx = {c: i for i, c in enumerate(self.classes)}

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = self.label2idx[row["label"]]

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image, label