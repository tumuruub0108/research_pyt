import numpy as np
import matplotlib.pyplot as plt

N = 200
H = 2
EPOCHS = 300
LR = 0.01


# generate 2D Gaussian Data x= input, mu= mean, C = covariance matrix
def gaussian_2d(x, mean, C):
    difference = x - mean
    inv_C = np.linalg.inv(C)
    exponent = -0.5 * np.sum(difference @ inv_C * difference, axis=1)
    return np.exp(exponent)


def generate_data(N):
    x = np.random.uniform(-4, 4, (N, 2))
    mean = np.array([0, 0])
    C = np.array([[3.0, 0.5], [0.5, 2.0]])
    y = gaussian_2d(x, mean, C)
    return x, y.reshape(-1, 1)


# prepare datasets
X_train, y_train = generate_data(N)
X_test, y_test = generate_data(100)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


class MLP:
    def __init__(self, input_size, hidden_size, lr=0.01):
        self.lr = lr

        self.w1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))

        self.w2 = np.random.randn(1, hidden_size) * 0.1
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        self.x = x.reshape(-1, 1)

        self.z1 = self.w1 @ self.x + self.b1
        self.out1 = sigmoid(self.z1)

        self.z2 = self.w2 @ self.out1 + self.b2
        self.y_pred = self.z2  # linear activation

        return self.y_pred

    def backward(self, y_true):
        y_true = y_true.reshape(1, 1)

        # output layer error
        dz2 = self.y_pred - y_true
        dw2 = dz2 @ self.out1.T
        db2 = dz2

        # Hidden layer error
        dz1 = (self.w2.T @ dz2) * sigmoid_derivative(self.z1)
        dw1 = dz1 @ self.x.T
        db1 = dz1

        # update weights
        self.w2 = self.w2 - self.lr * dw2
        self.b2 = self.b2 - self.lr * db2

        self.w1 = self.w1 - self.lr * dw1
        self.b1 = self.b1 - self.lr * db1


def train(model, X_train, y_train, epochs):
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for x_i, y_i in zip(X_train, y_train):
            y_pred = model.forward(x_i)

            loss = 0.5 * (y_pred - y_i) ** 2
            total_loss = total_loss + loss.item()

            model.backward(y_i)

        losses.append(total_loss / len(X_train))

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]}")

    return losses


def predict(model, X):
    preds = []

    for x in X:
        y_pred = model.forward(x)
        preds.append(y_pred[0][0])

    return np.array(preds)


mlp = MLP(input_size=2, hidden_size=H, lr=LR)

losses = train(model=mlp, X_train=X_train, y_train=y_train, epochs=EPOCHS)

y_pred = predict(model=mlp, X=X_test)
mse = np.mean((y_pred - y_test.flatten()) ** 2)
print("Test MSE:", mse)


# Epochs (E*)
def experiment_epochs():

    epoch_list = [50, 100, 200, 300, 500]
    test_errors = []

    for E in epoch_list:
        mlp = MLP(input_size=2, hidden_size=5, lr=0.01)

        train(mlp, X_train, y_train, epochs=E)

        y_pred = predict(mlp, X_test)
        mse = np.mean((y_pred - y_test.flatten()) ** 2)

        test_errors.append(mse)

    plt.figure()
    plt.plot(epoch_list, test_errors, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Test MSE")
    plt.title("Effect of Epochs (E)")
    plt.show()

    return epoch_list, test_errors


# Training Size (N*)
def experiment_N():

    N_list = [50, 100, 200, 500]
    test_errors = []

    for N in N_list:
        X_train, y_train = generate_data(N)

        mlp = MLP(input_size=2, hidden_size=5, lr=0.01)

        train(mlp, X_train, y_train, epochs=200)  # use E*

        y_pred = predict(mlp, X_test)
        mse = np.mean((y_pred - y_test.flatten()) ** 2)

        test_errors.append(mse)

    plt.figure()
    plt.plot(N_list, test_errors, marker="o")
    plt.xlabel("Training Size (N)")
    plt.ylabel("Test MSE")
    plt.title("Effect of Training Size (N)")
    plt.show()

    return N_list, test_errors


# Hidden Nodes (H*)
def experiment_H():

    H_list = [2, 5, 10, 20]
    test_errors = []

    for H in H_list:
        mlp = MLP(input_size=2, hidden_size=H, lr=0.01)

        train(mlp, X_train, y_train, epochs=200)  # use E*

        y_pred = predict(mlp, X_test)
        mse = np.mean((y_pred - y_test.flatten()) ** 2)

        test_errors.append(mse)

    plt.figure()
    plt.plot(H_list, test_errors, marker="o")
    plt.xlabel("Hidden Nodes (H)")
    plt.ylabel("Test MSE")
    plt.title("Effect of Hidden Nodes (H)")
    plt.show()

    return H_list, test_errors


experiment_epochs()
experiment_N()
experiment_H()

# 3D Visualization
grid_size = 30
x1 = np.linspace(-4, 4, grid_size)
x2 = np.linspace(-4, 4, grid_size)
X1, X2 = np.meshgrid(x1, x2)

grid_points = np.c_[X1.ravel(), X2.ravel()]

Z_pred = predict(mlp, grid_points).reshape(grid_size, grid_size)
Z_true = gaussian_2d(
    grid_points, np.array([0, 0]), np.array([[3.0, 0.5], [0.5, 2.0]])
).reshape(grid_size, grid_size)

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(X1, X2, Z_true)
ax.set_title("True Gaussian")

ax = fig.add_subplot(122, projection="3d")
ax.plot_surface(X1, X2, Z_pred)
ax.set_title("MLP Prediction")

plt.show()
