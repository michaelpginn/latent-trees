from transformers import PreTrainedTokenizer
from collections import Counter
import json

class WhitespaceTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab=None, max_length=1024, **kwargs):
        super().__init__(**kwargs)
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.model_max_length = max_length
        self.vocab = vocab if vocab else {self.unk_token: 0, self.cls_token: 1, self.sep_token: 2}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text, **kwargs):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_ids_to_tokens(self, ids, **kwargs):
        id_to_token = {id: token for token, id in self.vocab.items()}
        return [id_to_token.get(id, self.unk_token) for id in ids]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.vocab[self.cls_token]] + token_ids_0 + [self.vocab[self.sep_token]]
        else:
            return [self.vocab[self.cls_token]] + token_ids_0 + [self.vocab[self.sep_token]] + token_ids_1 + [self.vocab[self.sep_token]]

    def learn_vocab(self, sentences, min_freq=1):
        """Learn a vocabulary from a list of sentences."""
        token_freqs = Counter()
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            token_freqs.update(tokens)

        # Remove tokens that don't meet the minimum frequency threshold
        token_freqs = {k: v for k, v in token_freqs.items() if v >= min_freq}

        # Assign IDs to each token
        next_id = max(self.vocab.values()) + 1  # Start after the last special token ID
        for token, freq in token_freqs.items():
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1

    def tokenize_batch(self, batch):
        """Tokenize a batch of text using the custom tokenizer"""
        inputs = [self.tokenize(text) for text in batch["text"]]
        input_ids = [self.convert_tokens_to_ids(tokens) for tokens in inputs]
        input_ids = [self.build_inputs_with_special_tokens(ids) for ids in input_ids]

        input_ids = [ids[:self.model_max_length] + [0] * (self.model_max_length - len(ids)) for ids in input_ids]
        attention_mask = [[1] * min(len(ids), self.model_max_length) + [0] * (self.model_max_length - len(ids)) for ids in
                          input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # Saving
    def save_vocabulary(self, save_directory):
        vocab_file = f"{save_directory}/vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        vocab_file = f"{pretrained_model_name_or_path}/vocab.json"
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        return cls(vocab, *args, **kwargs)