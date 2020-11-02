# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class MyClassifier1A(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_dim,
                 num_layers, device, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device=device

        self.embedded = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                          #dropout=dropout, 
                          batch_first=True
                         )
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x, hs=None):
        batch = x.shape[0]
        print("x.shape", x.shape)
        if hs is None:
            hs = Variable(
                torch.zeros(self.num_layers, batch, self.hidden_dim))
            hs = hs.to(self.device)
            print("hs", hs.shape)
        word_embedd = self.embedded(x.long())
        print("word_embed", word_embedd.shape)
        word_embedd = torch.sum(word_embedd, dim=2)  # (batch, len, embed)
        print("word_embed", word_embedd.shape)
        #word_embed = word_embed.permute(1, 0, 2)  # (len, batch, embed)
        out, h0 = self.gru(word_embedd, hs)  # (len, batch, hidden)
        le, ba, hd = out.shape
        print("GRUoutput.shape",out.shape)
        out = out.reshape(le * ba, hd)
        print("GRUoutput.shape",out.shape)
        out = self.linear(out)
        print("linear.shape",out.shape)
        out = out.reshape(le, ba, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)
        return out.view(-1, out.shape[2]), h0

class MyClassifier1B(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_dim,
                 num_layers, device, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device=device
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, 
                          #dropout=dropout, 
                          batch_first=True
                         )
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax= nn.LogSoftmax(dim=2)  

    def forward(self, x, hs=None):
        batch = x.shape[0]
        out = self.linear1(x)
        #print("x.shape and dtype:", x.shape, x.dtype)
        if hs is None:
            hs = Variable(
                torch.zeros(self.num_layers, batch, self.hidden_dim, dtype=torch.float))
            hs = hs.to(self.device)
            #print("hs.shape and dtype", hs.shape, hs.dtype)

        out, h0 = self.gru(out, hs)  # (len, batch, hidden)
        #le, ba, hd = out.shape
        #print("GRUoutput.shape",out.shape)
        #out = out.reshape(le * ba, hd)
        #print("GRUoutput.shape",out.shape)
        out = self.linear(out)
        #print("linear.shape",out.shape)
        #out = out.reshape(le, ba, -1)
        #out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)
        #out = out.view(-1, out.shape[2])
        out = self.softmax(out)
        return out #, h0

    
class MyClassifier2A(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_dim, num_layers, device, dropout=0.5):
        self.num_layers=num_layers
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.device=device
        
        super().__init__()
        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True, 
                            bidirectional=True,
                           )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        embeds = torch.sum(self.embeddings(x.long()),dim=2)
        #embeds = self.embeddings(x.view(-1, x.size(2)))
        #embeds1 = torch.flatten(self.embeddings(x.long()), start_dim=2)
        print("embeds.shape", embeds.shape)

        h0 = torch.randn(self.num_layers*2, x.shape[0], self.hidden_dim)
        h0=h0.to(self.device)
        c0 = torch.randn(self.num_layers*2, x.shape[0], self.hidden_dim)
        c0=c0.to(self.device)
        lstm_out, (hn,cn) = self.lstm(embeds, (h0,c0))
        print (lstm_out.shape)
       
        dropout_out = self.dropout(lstm_out)
        print(len(dropout_out), dropout_out.shape)
        output = self.linear(dropout_out)
        print("linear_output.shape:", output.shape)
        #output = F.log_softmax(output, dim=1)
        #print("len(softmax_out)", len(softmax_out), "output.shape", output.shape)
        return output

class MyClassifier2B(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_dim, num_layers, device, dropout=0.5):
        self.num_layers=num_layers
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.device=device
        
        super().__init__()
        #self.lin1 = nn.Linear(input_size, hidden_dim)
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout
                           )
        #self.linear = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        #self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_dim*2, output_dim)
        #self.softmax = nn.LogSoftmax(dim=2) 

    def forward(self, x):
        #first = self.lin1(x)
        h0 = torch.randn(self.num_layers*2, x.shape[0], self.hidden_dim)
        h0=h0.to(self.device)
        c0 = torch.randn(self.num_layers*2, x.shape[0], self.hidden_dim)
        c0=c0.to(self.device)
        
        lstm_out, _ = self.lstm(x)
        #dropout_out = self.dropout(lstm_out)
        output = self.linear(lstm_out)
        #output = self.softmax(output)
        return output
    
class MyClassifier3A(nn.Module):
    def __init__(   self, input_size, embedding_size, hidden_dim, output_size, num_layers, device,
                    dropout=0.5,
                ):

        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN( embedding_size, 
                           hidden_dim, 
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout = dropout if num_layers > 1 else 0,
                        )
        self.linear = nn.Linear(hidden_dim*2, output_size)

    def forward(self, x):
        # embedding shape: (batch_size, seq_length, embedding_size)
        embedding = torch.sum(self.dropout(self.embedding(x.long())),dim=2)
        print("embedding.shape:",embedding.shape)
        h0 = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_dim)
        h0 = h0.to(self.device)
        # outputs shape: (num_batch, seq_length, hidden_size*2)
        rnn_outputs, _ = self.rnn(embedding, h0)
        print("rnn_outputs.shape:", rnn_outputs.shape)
        # outputs shape: (num_batch, seq_length, output_size)
        output = self.linear(self.dropout(rnn_outputs))
        print("utput.shape:", output.shape)
        return output
    
class MyClassifier3B(nn.Module):
    def __init__(   self, input_size, embedding_size, hidden_dim, output_size, num_layers, device,
                    dropout=0.5,
                ):

        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.rnn = nn.RNN( hidden_dim, 
                           hidden_dim, 
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout = dropout if num_layers > 1 else 0,
                        )
        self.linear = nn.Linear(hidden_dim*2, output_size)
        self.softmax = nn.LogSoftmax(dim=2) 

    def forward(self, x):
        hidden = self.linear1(x)
        h0 = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_dim)
        h0 = h0.to(self.device)
        rnn_outputs, _ = self.rnn(hidden, h0)
        #print("rnn_outputs.shape:", rnn_outputs.shape)
        output = self.linear(rnn_outputs)
        output = self.softmax(output)
        #print("output.shape:", output.shape)
        return output