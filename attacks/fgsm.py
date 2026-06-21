import torch
from model.train import DEVICE

def fgsm_attack(model, image, label, epsilon=0.1):
    image = image.squeeze()
    while image.dim() < 4:
        image = image.unsqueeze(0)
    image = image.to(DEVICE).clone().detach().requires_grad_(True)
    label = label.to(DEVICE).clone().detach()

    output = model(image)
    loss = torch.nn.functional.binary_cross_entropy(output, label)

    model.zero_grad()
    loss.backward()

    perturbation = epsilon * image.grad.sign()
    adversarial_image = image + perturbation
    adversarial_image = torch.clamp(adversarial_image, -1, 1)
    return adversarial_image.detach()