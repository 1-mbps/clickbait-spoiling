import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from nltk.translate.meteor_score import meteor_score
import numpy as np

from models import T5BaseModel

def train_model(
    model: T5BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: AutoTokenizer,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    eps: float = 1e-8,
    l2: float = 0,
    device: str = "cpu",
    class_weights: torch.Tensor = None,
    clip_grad_norm: bool = False,
    grad_checkpoint: bool = True
) -> tuple[list[float], list[float]]:
    
    if class_weights is not None:
        class_weights.to(device)
    if grad_checkpoint:
        model.transformer.gradient_checkpointing_enable()
    

    # Set up optimizer (AdamW with L2 regularization via weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=eps, weight_decay=l2)

    # set up learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
        num_warmup_steps=0, num_training_steps=total_steps
    )

    train_losses = []
    val_losses = []

    model.to(device)

    print("🚀 Starting training!")
    
    for epoch in range(epochs):

        # training loop
        model.train()
        train_loss = 0
        i = 0
        for batch in train_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            l_input_ids = batch[2].to(device)
            optimizer.zero_grad()
            
            output = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=l_input_ids)
            
            # 4. Track training and validation losses/accuracies
            loss = output.loss
            loss.backward()
            train_loss += loss.item()

            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            # if i % 20 == 0:
            #     print(f"TRAIN LOSS: {loss.item()}")
            # i += 1
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print(f"Epoch {epoch+1}{' '*int(epoch+1 < 10)} - train loss: {train_loss:.6f}", end=" | ")

        # validation loop
        model.eval()
        val_loss = 0
        meteor_scores = []

        j = 0
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                labels = batch[2]

                raw_out = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=labels.to(device))
                outputs = model.transformer.generate(input_ids=b_input_ids, attention_mask=b_input_mask)
                for i in range(len(outputs)):
                    output = tokenizer.decode(outputs[i], skip_special_tokens=True)
                    target = tokenizer.decode([token for token in labels[i] if token != -100], skip_special_tokens=True)
                    m_score = meteor_score([target.split()], output.split())
                    meteor_scores.append(m_score)

                val_loss += raw_out.loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"val loss: {val_loss:.6f} | avg meteor score: {float(np.mean(meteor_scores)):.6f}")

    return train_losses, val_losses
