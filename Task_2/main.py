import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import *
from models import T5BaseModel
from tokenizer import Tokenizer
from training import train_model

BATCH_SIZE = 16

torch.manual_seed(42)
np.random.seed(42)

def collate_fn(batch):
    """
    Custom collate function for Task2Dataset that dynamically pads input_ids, attention_mask, and labels.
    This was written by GPT-4.1
    """
    # Unpack the batch into separate lists
    input_ids_list, attention_masks_list, _, labels_list = zip(*batch)

    # Convert to tensors
    input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids_list]
    attention_masks = [torch.tensor(x, dtype=torch.long) for x in attention_masks_list]
    labels = [torch.tensor(x, dtype=torch.long) for x in labels_list]

    # Pad to the length of the longest sequence in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return input_ids_padded, attention_mask_padded, labels_padded

# tuples (i,j) - freeze i encoder layers, j decoder layers
frozen_layers = {
    "small": (0, 0),
    "base": (6, 0),
    "large": (0, 0)
}

def main(tokenizer_name: str = "google/flan-t5-small", loaded_model_name: str = None, model_save_name: str = "t5-small-squad.pth", squad: bool = False):
    tokenizer = Tokenizer(tokenizer_name)

    for s in ["small", "base", "large"]:
        if s in tokenizer_name:
            size = s

    if squad:
        DATASET_NAME = f"squad_v1_weighted_{size}"
    else:
        DATASET_NAME = tokenizer_name.split('/')[-1]
        
    loaded_model_name = loaded_model_name or tokenizer_name

    # load dataset
    train_dataset = torch.load(f"encoded_data/{DATASET_NAME}_train.pt", weights_only=False)
    val_dataset = torch.load(f"encoded_data/{DATASET_NAME}_val.pt", weights_only=False)

    # initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, collate_fn=collate_fn)

    for size, vals in frozen_layers.items():
        if size in tokenizer_name:
            enc, dec = vals

    model = T5BaseModel(loaded_model_name, encoder_layers_freeze=enc, decoder_layers_freeze=dec)
    train_model(model, train_loader, val_loader, tokenizer.tokenizer, device="cuda", learning_rate=5e-5, epochs=2)
    model.transformer.save_pretrained(model_save_name)

if __name__ == "__main__":
    # Finetune on SQuAD
    main(tokenizer_name="google/flan-t5-small", loaded_model_name="t5-small-squad-pretrain", model_save_name="t5-small-squad")
    
    # Finetune on clickbait dataset
    main(tokenizer_name="google/flan-t5-small", model_save_name="t5-small-squad-pretrain", squad=True)