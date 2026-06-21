import torch
import torch.nn as nn
import torch.optim as optim
from model.classifier import PneumoniaClassifier
from utils.data_loader import get_data_loaders

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def train_classifier(progress_callback=None, epochs_override=None):
    EPOCHS = epochs_override if epochs_override else 10

    train_loader, val_loader, _ = get_data_loaders()

    model     = PneumoniaClassifier().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    train_losses, val_accuracies = [], []

    for epoch in range(EPOCHS):

        # ── Training ──
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            # labels from ImageFolder: 0=NORMAL, 1=PNEUMONIA
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ── Validation ──
        model.eval()
        correct = 0
        total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(DEVICE)
                labels  = labels.float().unsqueeze(1).to(DEVICE)
                outputs = model(images)
                preds   = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total * 100
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1f}%")

        if progress_callback:
            progress_callback((epoch + 1) / EPOCHS)

    torch.save(model.state_dict(), "classifier.pth")
    print("Classifier saved as classifier.pth")

    return model, train_losses, val_accuracies


def predict_single_image(model, image_tensor):
    """
    Takes a single image tensor, returns prediction label and confidence.
    """
    model.eval()
    image_tensor = image_tensor.squeeze()
    while image_tensor.dim() < 4:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        output     = model(image_tensor)
        confidence = output.item()
        prediction = "PNEUMONIA" if confidence > 0.5 else "NORMAL"

    return prediction, confidence

def load_classifier():
    """Load a previously trained classifier from disk."""
    import os
    if not os.path.exists("classifier.pth"):
        return None
    model = PneumoniaClassifier().to(DEVICE)
    model.load_state_dict(
        torch.load("classifier.pth", map_location=DEVICE)
    )
    model.eval()
    return model