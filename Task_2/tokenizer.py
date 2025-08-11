from transformers import AutoTokenizer, PreTrainedTokenizerBase

class Tokenizer:
    def __init__(self, tokenizer_name: str):
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_tokens("<SEP>", special_tokens=True)

        if tokenizer_name == "mrm8488/t5-base-finetuned-squadv2" or tokenizer_name.startswith("google/flan-t5"):
            # this is the input template for T5 models
            self.input_template = "question: {question}  context: {context}"
        else:
            raise ValueError("Invalid tokenizer name.")
    
    def encode_features(self, title: str, paragraphs: str, MAX_LEN = 2000):
        input_text = self.input_template.format(question=title, context=paragraphs)
        features = self.tokenizer.encode_plus(
            input_text,                      # Sentence to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = MAX_LEN,           # Pad & truncate all sentences.
            padding = "max_length",
            truncation=True,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
        )
        return features
    
    def encode_spoiler(self, spoiler: str, MAX_LEN = 64):
        encoded = self.tokenizer.encode_plus(
            spoiler,                      # Sentence to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = MAX_LEN,           # Pad & truncate all sentences.
            padding = "max_length",
            truncation=True,
            return_attention_mask = False,
            return_tensors = 'pt',     # Return pytorch tensors.
        )
        input_ids = encoded["input_ids"].clone()
        # replace padding tokens with -100 for labels to match T5's pretraining
        input_ids[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids}
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)