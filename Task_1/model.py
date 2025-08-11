import torch
from torch import nn

from transformers import AutoModel, BertModel as BertBase

class BertModel(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_features: int = 6, features_out: int = 8, dropout_rate: float = 0.2, layers_to_freeze: int = 6, class_dist: list[int] = [559, 1367, 1274]):
        super().__init__()

        self.transformer: BertBase = AutoModel.from_pretrained(model_name, output_attentions=False, output_hidden_states=False)

        if model_name == "roberta-large":
            HIDDEN_SIZE = 1024
        else:
            HIDDEN_SIZE = 768

        for i in range(layers_to_freeze):
            for param in self.transformer.encoder.layer[i].parameters():
                param.requires_grad = False

        self.transformer_block = nn.Sequential(
            nn.Dropout(dropout_rate),
        )

        self.feature_block = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=features_out),
            nn.ReLU()
        )

        final_layer = nn.Linear(in_features=features_out+HIDDEN_SIZE, out_features=3)

        # Apply Kaiming uniform initialization to all linear layers
        # self.feature_block.apply(init_weights)
        # final_layer.apply(init_weights)

        # counts = torch.tensor(class_dist, dtype=torch.float)
        # probs = counts / counts.sum()
        # bias_init = torch.log(probs)

        # final_layer.bias.data = bias_init

        self.fc = nn.Sequential(
            final_layer
        )

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None, attention_mask: torch.Tensor, features: torch.Tensor):
        t_raw = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = t_raw.pooler_output
        t_out = self.transformer_block(logits)

        feature_out = self.feature_block(features)

        out = torch.cat((t_out, feature_out), dim=1)

        return self.fc(out)
    
    def stop_transformer_backprop(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def set_feature_block_backprop(self, on: bool):
        for param in self.feature_block.parameters():
            param.requires_grad = on

    def set_final_layer_backprop(self, on: bool):
        for param in self.fc.parameters():
            param.requires_grad = on

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)