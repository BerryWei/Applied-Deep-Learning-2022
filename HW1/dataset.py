from typing import List, Dict
import numpy as np
from torch.utils.data import Dataset
import torch
from utils import Vocab
import torch.nn.functional as F

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    )re
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn

       

        text_list = [instance['text'].split() for instance in samples ]
        encode_batch_X = self.vocab.encode_batch(text_list, self.max_len)
        tensor_encode_X = torch.tensor(encode_batch_X)

        
        
        targets = [ self.label2idx(ins['intent']) for ins in samples ]
        tensor_targets = torch.tensor(targets)
        return tensor_encode_X, tensor_targets
    
    def collatePred_fn(self, samples: List[Dict]) -> Dict:
        text_list = [instance['text'].split() for instance in samples ]
        encode_batch_X = self.vocab.encode_batch(text_list, self.max_len)
        tensor_encode_X = torch.tensor(encode_batch_X)

        return tensor_encode_X

        

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples: List[Dict]) -> Dict:
        self.ignore_idx = -100
        text_list = [instance["tokens"] for instance in samples ]
        encode_batch_X = self.vocab.encode_batch(text_list, self.max_len)
        tensor_encode_X = torch.tensor(encode_batch_X)

        batch_size = len(samples)
        
        tensor_targets = torch.empty(batch_size, self.max_len, dtype=torch.long).fill_(self.ignore_idx)
        for i, instance in enumerate(samples):
            for j, word in enumerate(instance['tags']):
                tensor_targets[i][j] = self.label2idx(word)
        return tensor_encode_X, tensor_targets
    
    def collatePred_fn(self, samples: List[Dict]) -> Dict:
        text_list = [instance["tokens"] for instance in samples ]
        encode_batch_X = self.vocab.encode_batch(text_list, self.max_len)
        tensor_encode_X = torch.tensor(encode_batch_X)

        return tensor_encode_X
