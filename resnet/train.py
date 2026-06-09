import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
import config
from dataset import EmbryoDataset
from model import ResNetV2


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.ToTensor()
])

def get_loader(ds, shuffle=False):

    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2
    )

def main():

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    train_ds = EmbryoDataset(config.TRAIN_XLSX, transform=train_transform)
    val_ds   = EmbryoDataset(config.VAL_XLSX, transform=None)

    train_loader = get_loader(train_ds, shuffle=True)
    val_loader   = get_loader(val_ds, shuffle=False)

    num_classes = len(train_ds.classes)
    model = ResNetV2(num_classes).to(config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=1e-2
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS
    )

    scaler = GradScaler(enabled=True)

    best_acc = 0.0

    for epoch in range(config.EPOCHS):

        print(f"\nEpoch {epoch+1}")

        model.train()
        train_loss = 0

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            images = F.interpolate(
                images,
                size=(config.IMG_SIZE, config.IMG_SIZE),
                mode="bilinear",
                align_corners=False
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if i % 100 == 0:
                print(f"Batch {i}/{len(train_loader)}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True)

                images = F.interpolate(
                    images,
                    size=(config.IMG_SIZE, config.IMG_SIZE),
                    mode="bilinear",
                    align_corners=False
                )

                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model_rn.pth")
            print("Millor model guardat")

        torch.save(model.state_dict(), f"epoch_{epoch+1}.pth")

        scheduler.step()

        print(f"""
Epoch {epoch+1}
Train Loss: {train_loss/len(train_loader):.4f}
Val Acc:    {acc:.4f}
Best Acc:   {best_acc:.4f}
LR:         {scheduler.get_last_lr()[0]:.6f}
""")

if __name__ == "__main__":
    main()