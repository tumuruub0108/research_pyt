from data_loader.dataset import load_fashion_mnist


def main():
    train_loader, test_loader = load_fashion_mnist()
    for images, labels in train_loader:
        print(images.shape)
        print(labels.shape)
        break


if __name__ == "__main__":
    main()
