import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import numpy as np
import json

from training import train_model
from model import BertModel
from dataset import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.loss_curve import plot_loss_curve

def make_sure_dir_exists(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

BATCH_SIZE = 32

torch.manual_seed(42)
np.random.seed(42)

def main(large: bool = False, augmented: bool = False):

    AUGMENTED = "_augmented" if augmented else ""
    DATASET_NAME = "roberta_large" if large else "roberta"
    MODEL_NAME = "roberta-large" if large else "roberta-base"
    LAYERS_TO_FREEZE = 16 if large else 6

    # load dataset
    train_dataset = torch.load(f"encoded_data/{DATASET_NAME}_train{AUGMENTED}.pt", weights_only=False)
    val_dataset = torch.load(f"encoded_data/{DATASET_NAME}_val.pt", weights_only=False)

    # initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)

    model = BertModel(model_name=MODEL_NAME, layers_to_freeze=LAYERS_TO_FREEZE)

    # start train/validation pipeline
    train_losses, val_losses = train_model(model, train_loader, val_loader, device="mps", epochs=3, clip_grad_norm=False)
    
    # # save model
    make_sure_dir_exists("models")
    model_name = MODEL_NAME.replace('-', '_')
    torch.save(model.state_dict(), "models/"+model_name+".pth")

    make_plot = input("Generate loss curve? ")
    if make_plot == "y":
        plot_loss_curve(train_losses, val_losses, DATASET_NAME)
        with open("losses.json", "w") as f:
            result = {"train": train_losses, "val": val_losses}
            f.write(json.dumps(result))

if __name__ == "__main__":
    main(large=True,augmented=False)
