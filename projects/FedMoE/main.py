import torch
from data_loader.dataset import load_fashion_mnist
from experiments.baseline import run_baseline


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader, class_names = load_fashion_mnist()
    print(f"Lenght of train_loader: {len(train_loader)}.")
    print(f"Lenght of test_loader: {len(test_loader)}.")
    print(f"class names: {class_names}")

    cnn_baseline_result = run_baseline(
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    print(f"cnn_baseline_result: {cnn_baseline_result}")


if __name__ == "__main__":
    main()
