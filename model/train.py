import torch
import torch.nn as nn
import torch.optim as optim
from model.gan import Generator, Discriminator
from utils.data_loader import get_data_loaders

NOISE_DIM = 100

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

def train_gan(progress_callback=None, epochs_override=None):
    EPOCHS = epochs_override if epochs_override else 10

    train_loader, _, _ = get_data_loaders()  # now returns 3 loaders

    generator     = Generator(noise_dim=NOISE_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    opt_g = optim.Adam(generator.parameters(),     lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []

    for epoch in range(EPOCHS):
        for real_imgs, _ in train_loader:
            real_imgs  = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

            # Train Discriminator
            noise     = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            fake_imgs = generator(noise).detach()

            d_loss = (
                criterion(discriminator(real_imgs), real_labels) +
                criterion(discriminator(fake_imgs), fake_labels)
            )
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train Generator
            noise     = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            fake_imgs = generator(noise)
            g_loss    = criterion(discriminator(fake_imgs), real_labels)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        print(f"Epoch [{epoch+1}/{EPOCHS}] | G: {g_loss.item():.4f} | D: {d_loss.item():.4f}")

        if progress_callback:
            progress_callback((epoch + 1) / EPOCHS)

    torch.save(generator.state_dict(),     "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")

    return generator, discriminator, g_losses, d_losses

def load_generator():
    """Load a previously trained generator from disk."""
    import os
    if not os.path.exists("generator.pth"):
        return None
    model = Generator(noise_dim=NOISE_DIM).to(DEVICE)
    model.load_state_dict(
        torch.load("generator.pth", map_location=DEVICE)
    )
    model.eval()
    return model

def load_discriminator():
    """Load a previously trained discriminator from disk."""
    import os
    if not os.path.exists("discriminator.pth"):
        return None
    model = Discriminator().to(DEVICE)
    model.load_state_dict(
        torch.load("discriminator.pth", map_location=DEVICE)
    )
    model.eval()
    return model