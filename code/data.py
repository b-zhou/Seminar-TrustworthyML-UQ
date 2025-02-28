from sklearn.datasets import make_moons
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def get_data_classif(n_samples=100, noise=0.1, seed=1453):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X, y

def regr_true_func(x):
    return np.sin(x)

def get_dataloader(X, y, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ensure y is a column vector
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
