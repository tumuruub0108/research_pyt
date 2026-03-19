from pathlib import Path

# ===== Training =====
BATCH_SIZE = 32
EPOCHS = 10
DATA_PATH = Path(r"C:\Users\trt\Desktop\research_pyt\datasets")
LEARNING_RATE = 0.1


# ===== CNN =====
KERNEL_SIZE = 3
PADDING = 1
STRIDE = 1

# ===== Federated Learning =====
NUM_CLIENTS = 5
CLIENT_ID = 0

# ===== CNN + MoE =====
NUM_EXPERTS = 6
