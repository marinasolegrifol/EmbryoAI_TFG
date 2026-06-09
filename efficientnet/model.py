import torch.nn as nn
import timm

class EfficientNetB4(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=True
        )

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)