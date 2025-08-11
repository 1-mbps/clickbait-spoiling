import torch
from torch.utils.data import TensorDataset
import json
from sklearn.preprocessing import StandardScaler
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.cleaning import clean_text
from utils.feature_eng import check_patterns
from tokenizer import Tokenizer

class Task2Dataset(TensorDataset):
    def __init__(self, input_file: str, tokenizer: Tokenizer, test: bool = False, scaler = None):
        self.test = test
        input_ids = []
        attention_masks = []
        spoil_input_ids = []
        spoil_attention_masks = []
        num_text_features = []
        cat_text_features = []
        with open(input_file, 'r') as inp:
            for i, line in enumerate(inp):
                line = json.loads(line)

                # extract title, paragraphs, and label
                title = clean_text(line["postText"][0])
                paragraphs = line["targetParagraphs"]

                title_words = title.lower().split()

                num_features, cat_features = check_patterns(title_words, title, paragraphs)
                num_text_features.append(num_features)
                cat_text_features.append(cat_features)              

                # encode using tokenizer
                encoded_dict = tokenizer.encode_features(title, '\n'.join(paragraphs), MAX_LEN=1000)

                # Add the encoded sentence to the list.    
                input_ids.append(encoded_dict['input_ids'])
                
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_dict['attention_mask'])

                if not self.test:
                    spoiler = line["spoiler"]
                    # we added the special token <SEP> to the tokenizer to separate spoilers
                    spoiler_txt = ' <SEP> '.join(spoiler)
                    encoded_dict = tokenizer.encode_spoiler(spoiler_txt)
                    spoil_input_ids.append(encoded_dict["input_ids"])
                    spoil_attention_masks.append(encoded_dict["attention_mask"])
                    

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        if not scaler:
            scaler = StandardScaler()
            scaler.fit(num_text_features)

        num_scaled = scaler.transform(num_text_features)
        self.scaler = scaler

        # Create text feature tensor
        text_features = torch.stack([
            torch.cat((torch.tensor(num_scaled[index], dtype=torch.float32), cat_text_features[index]))
            for index in range(len(num_scaled))
        ])

        if self.test:
            super().__init__(input_ids, attention_masks, text_features)
        else:
            spoil_input_ids = torch.cat(spoil_input_ids, dim=0)
            spoil_attention_masks = torch.cat(spoil_attention_masks, dim=0)
            super().__init__(input_ids, attention_masks, text_features, spoil_input_ids, spoil_attention_masks)

def produce_clickbait_datasets(tokenizer_name: str):
    tokenizer = Tokenizer(tokenizer_name)

    for s in ["small", "base", "large"]:
        if s in tokenizer_name:
            size = s
    
    train_dataset = Task2Dataset("../data/train.jsonl", tokenizer)
    name = tokenizer_name.split('/')[-1]

    torch.save(train_dataset, f"encoded_data/{name}_train.pt")
    print("✅ Produced training dataset")
    val_dataset = Task2Dataset('../data/val.jsonl', tokenizer, scaler=train_dataset.scaler)
    torch.save(val_dataset, f"encoded_data/{name}_val.pt")
    print("✅ Produced validation dataset")

    test_dataset = Task2Dataset("../data/test.jsonl", tokenizer, scaler=train_dataset.scaler, test=True)
    torch.save(test_dataset, f"encoded_data/{name}_test.pt")
    print("✅ Produced test dataset")

def produce_squad_datasets(tokenizer_name: str):
    tokenizer = Tokenizer(tokenizer_name)

    for s in ["small", "base", "large"]:
        if s in tokenizer_name:
            size = s
    
    train_dataset = Task2Dataset("../data/squad_v1_weighted_train.jsonl", tokenizer)
    name = tokenizer_name.split('/')[-1]

    torch.save(train_dataset, f"encoded_data/squad_v1_weighted_{size}_train.pt")
    print("✅ Produced training dataset")

    val_dataset = Task2Dataset('../data/squad_v1_weighted_val.jsonl', tokenizer, scaler=train_dataset.scaler)
    torch.save(val_dataset, f"encoded_data/squad_v1_weighted_{size}_val.pt")
    print("✅ Produced validation dataset")

if __name__ == "__main__":
    produce_squad_datasets("google/flan-t5-small")
    produce_clickbait_datasets("google/flan-t5-small")