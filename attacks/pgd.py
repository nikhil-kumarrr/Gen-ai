import torch
from model.train import DEVICE

def pgd_attack(model, image, label, epsilon=0.1, alpha=0.01, num_steps=40):
    image = image.squeeze()
    while image.dim() < 4:
        image = image.unsqueeze(0)
    image = image.to(DEVICE).clone().detach()
    label = label.to(DEVICE).clone().detach()

    original_image = image.clone()
    perturbed_image = image.clone()

    for _ in range(num_steps):
        perturbed_image.requires_grad_(True)
        output = model(perturbed_image)
        loss = torch.nn.functional.binary_cross_entropy(output, label)

        model.zero_grad()
        loss.backward()

        step = alpha * perturbed_image.grad.sign()
        perturbed_image = perturbed_image.detach() + step

        delta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
        perturbed_image = torch.clamp(original_image + delta, -1, 1).detach()

    return perturbed_image