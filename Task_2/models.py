import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, T5Model

class T5BaseModel(nn.Module):
    def __init__(self, model_name: str = "mrm8488/t5-base-finetuned-squadv2", encoder_layers_freeze: int = 0, decoder_layers_freeze: int = 0):
        super().__init__()
        self.transformer: T5Model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        for i in range(encoder_layers_freeze):
            for param in self.transformer.encoder.block[i].parameters():
                param.requires_grad = False
        for i in range(decoder_layers_freeze):
            for param in self.transformer.decoder.block[i].parameters():
                param.requires_grad = False

    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        token_id_outs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return token_id_outs

    def stop_transformer_backprop(self):
        for param in self.transformer.parameters():
            param.requires_grad = False


