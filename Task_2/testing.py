import torch
import json
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import *
from models import *

BATCH_SIZE = 16

def test_model(model: nn.Module, test_loader: DataLoader, input_file: str, tokenizer: AutoTokenizer, device: str = "cuda"):
    
    ids = []
    with open(input_file, "r") as f:
        for i in f:
            i = json.loads(i)
            ids.append(i['id'])

    i = 0
    model.to(device)
    model.eval()

    with torch.no_grad():
        with open("spoiler_result.jsonl", 'w') as out:
            for batch in test_loader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
    
                outputs = model.generate(input_ids=b_input_ids, attention_mask=b_input_mask)
    
                for p in outputs:
                    prediction = tokenizer.decode(p, skip_special_tokens=True)
                    if not prediction:
                        prediction = "empty string"
                    out.write(json.dumps({"id": ids[i], "spoiler": prediction})+'\n')
                    i += 1
    
    # Create DataFrame
    df = pd.read_json('spoiler_result.jsonl', lines=True)
    
    # Save as CSV
    df.to_csv('task2_submission.csv', index=False)
    
    print(f"Converted predictions to CSV")
    print("First few rows:")
    print(df.head())
    
def main(tokenizer_name: str = "mrm8488/t5-base-finetuned-squadv2", model_name: str = "t5-base"):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    DATASET_NAME = tokenizer_name.split('/')[-1]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    test_dataset = torch.load(f"encoded_data/{DATASET_NAME}_test.pt", weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)
    test_model(model, test_loader, "../data/test.jsonl", tokenizer)

if __name__ == "__main__":
    main(tokenizer_name="google/flan-t5-small", model_name="t5-small-squad")

