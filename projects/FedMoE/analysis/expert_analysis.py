import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_expert_class_matrix(model, dataloader, num_experts, num_classes, device):
    model.eval()

    matrix = torch.zeros(num_experts, num_classes)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            features = model.feature_extractor(images)
            gate_weights = model.gating(features)

            # select top-1 expert
            selected_experts = torch.argmax(gate_weights, dim=1)

            for i in range(len(labels)):
                expert_id = selected_experts[i].item()
                class_id = labels[i].item()
                matrix[expert_id, class_id] += 1

    return matrix


def plot_expert_heatmap(matrix):
    matrix = matrix.numpy()

    plt.figure()
    plt.imshow(matrix)
    plt.colorbar()

    plt.xlabel("Class")
    plt.ylabel("Expert")
    plt.title("Expert vs Class Distribution")

    plt.xticks(np.arange(matrix.shape[1]))
    plt.yticks(np.arange(matrix.shape[0]))

    plt.show()
