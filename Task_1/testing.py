import torch
import json
from torch import nn
from torch.utils.data import DataLoader

from dataset import *
from model import *

def test_model(model: nn.Module, test_loader: DataLoader, input_file: str, outfile_path: str = "classification_result.jsonl", device: str = "mps", bert: bool = True):
    
    ids = []
    with open(input_file, "r") as f:
        for i in f:
            i = json.loads(i)
            ids.append(i['id'])

    i = 0
    label_encodings = ["multi", "passage", "phrase"]

    model.to(device)
    model.eval()

    with torch.no_grad():
        with open(outfile_path, 'w') as out:
            for tpl in test_loader:
                batch_features = [f.to(device) for f in tpl]
                batch_features.insert(1, None)
                raw_pred = model(*batch_features)
                pred = torch.softmax(raw_pred, dim=1)
                pred_class = torch.argmax(pred, dim=1)

                for p in pred_class.tolist():
                    prediction = {"id": ids[i], "spoilerType": label_encodings[int(p)]}
                    out.write(json.dumps(prediction)+'\n')
                    i += 1

def main(large: bool = False):

    DATASET = "roberta_large_test" if large else "roberta_test"
    SAVED_MODEL_NAME = "roberta_large" if large else "roberta_base"
    MODEL_NAME = SAVED_MODEL_NAME.replace('_', '-')
    LAYERS_TO_FREEZE = 16 if large else 6

    state_dict = torch.load(f"models/{SAVED_MODEL_NAME}.pth", map_location="cpu")
    model = BertModel(model_name=MODEL_NAME, layers_to_freeze=LAYERS_TO_FREEZE)
    model.load_state_dict(state_dict)

    test_dataset = torch.load(f"encoded_data/{DATASET}.pt", weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=32)
    test_model(model, test_loader, "../data/test.jsonl")

if __name__ == "__main__":
    main(large=True)