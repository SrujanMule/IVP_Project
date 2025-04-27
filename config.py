# config.py
import torch 

# Image dimensions for resizing
resize_x = 128
resize_y = 128
input_channels = 3

# Training hyperparameters
batchsize = 32
epochs = 50
learning_rate = 1e-3

#Grouping based on questions in each class
group_size = [3, 2, 2, 2, 4, 2, 3, 7, 3, 3, 6]

# BatchNorm parameters
batch_norm_momentum = 0.01  # Momentum for BatchNorm layers
batch_norm_eps = 1e-3       # Small epsilon for BatchNorm stability

# Loss function hyperparameters
epsilon = 1e-12  # Epsilon for stability in group-wise normalization

# Dataset and weight paths (relative to the submission root directory)
data_dir = "data"  # Should contain the 10 example images
csv_path = "data/data.csv"

# Checkpoint and model saving
weights_dir = "checkpoints"
weights_path = "checkpoints/final_weights.pth"

# Device selection (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
