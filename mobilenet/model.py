import timm
import torch.nn as nn

class MobileNetV3(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.model = timm.create_model(
            "mobilenetv3_large_100",  # base estable en timm
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)