import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from model import BertModel

dead_neuron_counts = None  # to sum zeros per neuron across the whole val set
num_samples = 0

def check_dead_neurons(module, input, output):
    global dead_neuron_counts, num_samples
    batch_zeros = (output == 0).sum(dim=0).cpu()  # shape: [num_neurons]
    if dead_neuron_counts is None:
        dead_neuron_counts = batch_zeros.clone()
    else:
        dead_neuron_counts += batch_zeros
    num_samples += output.shape[0]

def train_model(
    model: BertModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    bert_epochs: int = 3,
    learning_rate: float = 2e-5,
    eps: float = 1e-8,
    l2: float = 0,
    device: str = "cpu",
    class_weights: torch.Tensor = None,
    clip_grad_norm: bool = False
) -> tuple[list[float], list[float]]:
    
    if class_weights is not None:
        class_weights.to(device)
    
    # 1. Set up loss function (CrossEntropyLoss)
    loss_fn = nn.CrossEntropyLoss()

    # 2. Set up optimizer (AdamW with L2 regularization via weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=eps, weight_decay=l2)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
        num_warmup_steps = 0, # Default value in run_glue.py
        num_training_steps = total_steps
    )

    train_losses = []
    val_losses = []

    model.to(device)

    print("🚀 Starting training!")
    
    for epoch in range(epochs):
        
        # stop updating weights of transformer past `bert_epochs` epochs
        if epoch >= bert_epochs:
            model.stop_transformer_backprop()

        # training loop
        model.train()
        train_loss = 0
        i = 0
        for batch in train_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_features = batch[2].to(device)
            b_labels = batch[3].to(device)
            optimizer.zero_grad()
            
            logits = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask, features=b_features)
            
            # 4. Track training and validation losses/accuracies
            loss = loss_fn(logits, b_labels)
            train_loss += loss.item()
            loss.backward()

            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print(f"Epoch {epoch+1}{' '*int(epoch+1 < 10)} - train loss: {train_loss:.6f}", end=" | ")

        # validation loop
        model.eval()
        val_loss = 0
        y_pred = []
        y_true = []
        dead_neuron_counts = None
        num_samples = 0

        layer_to_check = model.feature_block[2]
        hook_handle = layer_to_check.register_forward_hook(check_dead_neurons)

        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_features = batch[2].to(device)
                b_labels = batch[3]
                logits = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask, features=b_features)
                
                # put logits through softmax
                pred = torch.softmax(logits, dim=1)

                # get predicted category
                pred_class = torch.argmax(pred, dim=1)

                # Use raw logits to compute loss
                val_loss += loss_fn(logits, b_labels.to(device)).item()
                
                # record predictions and labels
                y_pred.append(pred_class.cpu())
                y_true.append(b_labels)

        if dead_neuron_counts is not None:
            # Dead if zero for all samples in val set
            dead_per_unit = (dead_neuron_counts == num_samples)
            dead_count = dead_per_unit.sum().item()
        else:
            dead_count = 0

        hook_handle.remove()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"val loss: {val_loss:.6f} | F1: {f1:.6f} | avg dead neurons: {dead_count:.5f}")

    return train_losses, val_losses
