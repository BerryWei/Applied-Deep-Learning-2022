from typing import Dict
import torch.nn.functional as F
import torch
from torch.nn import Embedding
from torch import nn, optim
from torch.autograd import Variable


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()

        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        # TODO: model architecture
        self.rnn= nn.GRU(input_size=self.embed.embedding_dim , hidden_size=self.hidden_size, num_layers=self.num_layers,dropout=self.dropout,bidirectional=self.bidirectional, batch_first=True)
        # x --> batch, seq, feature
        self.drop = nn.Dropout(p=self.dropout)
        if (self.bidirectional == False):
            self.linear1=nn.Linear(self.hidden_size, self.num_class)
        else:
            self.linear1=nn.Linear(self.hidden_size * 2, self.num_class)
        
        #self.softmax = nn.Softmax(dim=1)
        

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) :

        output = self.embed(batch)
        output = self.drop(output)
        output, (h_n) = self.rnn(output)
        output = output[:,-1,:]         
        output = self.linear1(output)
        
        return output
        
class SlotTagging(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SlotTagging, self).__init__()

        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        # TODO: model architecture
        self.rnn= nn.LSTM(input_size=self.embed.embedding_dim , hidden_size=self.hidden_size, num_layers=self.num_layers,dropout=self.dropout,bidirectional=self.bidirectional, batch_first=True)
        # x --> batch, seq, feature
        self.drop = nn.Dropout(p=self.dropout)
        if (self.bidirectional == False):
            self.linear1=nn.Linear(self.hidden_size, self.num_class)
        else:
            self.linear1=nn.Linear(self.hidden_size * 2, self.num_class)
        
        #self.softmax = nn.Softmax(dim=1)
        

    @property
    def encoder_output_size(self) -> int:
        raise NotImplementedError

    def forward(self, batch) :
        
        output = self.embed(batch)
        output = self.drop(output)
        # print(output.size()) [batch_size, seq_len, embed_features]
        output, (h_n, c_n) = self.rnn(output)   
        output = self.linear1(output)
        # torch.Size([16, 50, 9]) [batch_size, seq_len, output_dim]
        output = output.permute(0, 2, 1)
        # torch.Size([16, 9, 50]) [batch_size, output_dim, seq_len]

        
        return output
        