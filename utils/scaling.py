import torch
import numpy as np

def scale_numeric_features(features: list[torch.Tensor], columns: list[int]):
    col_data = [[0 for _ in range(len(features))] for _ in range(len(columns))]
    for row_index, tensor in enumerate(features):
        for col_index in columns:
            col_data[col_index][row_index] = tensor[col_index]

    for i, data in enumerate(col_data):
        col_data[i] = (np.mean(data), np.std(data))

    for r in range(len(features)):
        for c in range(len(columns)):
            features[r][c] = features[r][c]