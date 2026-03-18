import torch
from data_loader.dataset import load_fashion_mnist
from experiments.cnn_baseline import train_cnn_baseline
from experiments.moe_experiment import train_moe_model
from models.moe import MoE
from conf.config import NUM_EXPERTS
from analysis.expert_analysis import compute_expert_class_matrix, plot_expert_heatmap


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"device: {device}")

    train_loader, test_loader, class_names = load_fashion_mnist()
    # print(f"Lenght of train_loader: {len(train_loader)}.")
    # print(f"Lenght of test_loader: {len(test_loader)}.")
    # print(f"class names: {class_names}")

    moe_model = MoE(
        input_shape=1,
        hidden_units=10,
        output_shape=len(class_names),
        num_experts=NUM_EXPERTS,
    )

    moe_results = train_moe_model(
        model=moe_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    print(f"moe_results: {moe_results}")

    matrix = compute_expert_class_matrix(
        model=moe_model,
        dataloader=test_loader,
        num_experts=NUM_EXPERTS,
        num_classes=len(class_names),
        device=device,
    )

    print(matrix)
    matrix = matrix / matrix.sum(dim=1, keepdim=True)

    plot_expert_heatmap(matrix=matrix)


"""  cnn_baseline_result = train_cnn_baseline(
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    print(f"cnn_baseline_result: {cnn_baseline_result}")
    
"""

if __name__ == "__main__":
    main()
