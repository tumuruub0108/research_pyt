import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. Generate 2D Gaussian Data
# -----------------------------
def gaussian_2d(x, mu, C):
    diff = x - mu
    inv_C = np.linalg.inv(C)
    exponent = -0.5 * np.sum(diff @ inv_C * diff, axis=1)
    return np.exp(exponent)


def generate_data(N):
    x = np.random.uniform(-4, 4, (N, 2))
    mu = np.array([0, 0])
    C = np.array([[3.0, 0.5], [0.5, 2.0]])
    y = gaussian_2d(x, mu, C)
    return x, y.reshape(-1, 1)


# -----------------------------
# 2. Activation Functions
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# -----------------------------
# 3. MLP Class (2 → H → 1)
# -----------------------------
class MLP:
    def __init__(self, input_size, hidden_size, lr=0.01):
        self.lr = lr

        # Initialize weights
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = np.random.randn(1, hidden_size) * 0.1
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        self.x = x.reshape(-1, 1)  # (2,1)

        self.z1 = self.W1 @ self.x + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.y_hat = self.z2  # regression (no activation)

        return self.y_hat

    def backward(self, y):
        y = y.reshape(1, 1)

        # Output layer error
        dz2 = self.y_hat - y  # (1,1)

        dW2 = dz2 @ self.a1.T
        db2 = dz2

        # Hidden layer error
        dz1 = (self.W2.T @ dz2) * sigmoid_derivative(self.z1)

        dW1 = dz1 @ self.x.T
        db1 = dz1

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, Y, epochs):
        losses = []

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(X, Y):
                y_hat = self.forward(x)
                loss = 0.5 * (y_hat - y) ** 2
                total_loss += loss

                self.backward(y)

            losses.append(total_loss / len(X))

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1][0]}")

        return losses

    def predict(self, X):
        preds = []
        for x in X:
            preds.append(self.forward(x)[0][0])
        return np.array(preds)


# -----------------------------
# 4. Training
# -----------------------------
N = 200
H = 5
EPOCHS = 300
LR = 0.01

X_train, Y_train = generate_data(N)
X_test, Y_test = generate_data(100)

mlp = MLP(input_size=2, hidden_size=H, lr=LR)

losses = mlp.train(X_train, Y_train, EPOCHS)

# -----------------------------
# 5. Evaluation
# -----------------------------
Y_pred = mlp.predict(X_test)

mse = np.mean((Y_pred - Y_test.flatten()) ** 2)
print("Test MSE:", mse)

# -----------------------------
# 6. Visualization
# -----------------------------

# Loss curve
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Compare prediction vs true
plt.figure()
plt.scatter(Y_test, Y_pred)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Prediction vs True")
plt.show()

# 3D surface visualization
grid_size = 30
x1 = np.linspace(-4, 4, grid_size)
x2 = np.linspace(-4, 4, grid_size)
X1, X2 = np.meshgrid(x1, x2)

grid_points = np.c_[X1.ravel(), X2.ravel()]
Z_pred = mlp.predict(grid_points).reshape(grid_size, grid_size)
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
