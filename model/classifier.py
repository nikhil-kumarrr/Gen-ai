import torch
import torch.nn as nn

class PneumoniaClassifier(nn.Module):
    """
    CNN that classifies chest X-rays as NORMAL or PNEUMONIA.
    Input:  [batch, 1, 64, 64] grayscale image
    Output: [batch, 1] probability (>0.5 = PNEUMONIA, <0.5 = NORMAL)
    """
    def __init__(self):
        super(PneumoniaClassifier, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 64x64 → 32x32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 32x32 → 16x16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 16x16 → 8x8

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # 8x8 → 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),              # prevent overfitting
            nn.Linear(512, 1),
            nn.Sigmoid()                  # output between 0 and 1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x