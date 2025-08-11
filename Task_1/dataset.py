import torch
from torch.utils.data import TensorDataset
import json
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.preprocessing import StandardScaler
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.cleaning import clean_text
from utils.feature_eng import check_patterns

tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)

def encode(title: str, MAX_LEN: int = 24):
    encoded_dict = tokenizer.encode_plus(
        title,                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = MAX_LEN,           # Pad & truncate all sentences.
        padding = "max_length",
        truncation=True,
        return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
    )
    return encoded_dict

class BertDataset(TensorDataset):
    def __init__(self, input_file: str, test: bool = False, scaler = None):
        label_encodings = {"multi": 0, "passage": 1, "phrase": 2}
        self.test = test
        input_ids = []
        attention_masks = []
        num_text_features = []
        cat_text_features = []
        labels = []
        MAX_LEN = 24
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
                encoded_dict = encode(title)

                # Add the encoded sentence to the list.    
                input_ids.append(encoded_dict['input_ids'])
                
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_dict['attention_mask'])

                if not self.test:
                    label = label_encodings[line["tags"][0]]
                    labels.append(label)

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
            labels = torch.tensor(labels)
            super().__init__(input_ids, attention_masks, text_features, labels)


if __name__ == "__main__":
    train_dataset = BertDataset("../data/train.jsonl")
    torch.save(train_dataset, "encoded_data/roberta_large_train.pt")
    print("✅ Produced training dataset")

    val_dataset = BertDataset("../data/val.jsonl", scaler=train_dataset.scaler)
    torch.save(val_dataset, "encoded_data/roberta_large_val.pt")
    print("✅ Produced validation dataset")

    test_dataset = BertDataset("../data/test.jsonl", test=True, scaler=train_dataset.scaler)
    torch.save(test_dataset, "encoded_data/roberta_large_test.pt")
    print("✅ Produced testing dataset")