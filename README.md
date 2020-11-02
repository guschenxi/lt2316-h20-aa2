## LT2316 H20 Assignment A2 : Ner Classification

Name: *fill in your real name here* (if you don't want to use your real name with your current GitHub account, you will have to make another GitHub account)

## Notes on Part 1.

*fill in notes and documentation for part 1 as mentioned in the assignment description*

1A:
MyClassifier1A(
  (embedded): Embedding(1827, 100)
  (rnn): GRU(100, 50, num_layers=3, batch_first=True)
  (linear): Linear(in_features=50, out_features=5, bias=False)
)
when using dropout, got error: RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
when running on CPU, got error: RuntimeError: index out of range: Tried to access index -1...
when running Linear, got error: cublas runtime error : resource allocation failed at /pytorch/aten/src/THC/THCGeneral.cpp:216


1B:
MyClassifier1B(
  (rnn): GRU(5, 10, num_layers=3, batch_first=True)
  (linear): Linear(in_features=10, out_features=5, bias=False)
)


2A:
MyClassifier2A(
  (embeddings): Embedding(5, 100)
  (lstm): LSTM(100, 10, num_layers=3, batch_first=True, bidirectional=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (linear): Linear(in_features=10, out_features=5, bias=True)
)
when running
        h0 = torch.randn(self.num_layers, x.shape[0], self.hidden_dim)
  -->   h0=h0.to(self.device)
        c0 = torch.randn(self.num_layers, x.shape[0], self.hidden_dim)
  -->   c0=c0.to(self.device)
got error: RuntimeError: CUDA error: device-side assert triggered
but when taking away   (embeddings): Embedding(5, 100), runs fine as in MyClassifier2B showed below.


2B:
MyClassifier2B(
  (lin1): Linear(in_features=5, out_features=3, bias=True)
  (lstm): LSTM(3, 3, num_layers=15, batch_first=True, dropout=0.2, bidirectional=True)
  (linear): Linear(in_features=6, out_features=6, bias=True)
  (softmax): LogSoftmax()
)


3A:
MyClassifier3A(
  (dropout): Dropout(p=0.2, inplace=False)
  (embedding): Embedding(5, 100)
  (rnn): LSTM(100, 10, num_layers=3, batch_first=True, dropout=0.2)
  (linear): Linear(in_features=10, out_features=5, bias=True)
)
got error "RuntimeError: CUDA error: device-side assert triggered" at h0 = h0.to(self.device)


MyClassifier3B(
  (dropout): Dropout(p=0.2, inplace=False)
  (linear1): Linear(in_features=5, out_features=10, bias=True)
  (rnn): RNN(10, 10, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
  (linear): Linear(in_features=20, out_features=6, bias=True)
  (softmax): LogSoftmax(dim=2)
)


Does both work?
loss = criterion(pred.permute(0,2,1), label.long())
loss = criterion(pred.view(pred.shape[0]*pred.shape[1],-1), label.view(-1).long())
## Notes on Part 2.

*fill in notes and documentation for part 2 as mentioned in the assignment description*

## Notes on Part 3.

*fill in notes and documentation for part 3 as mentioned in the assignment description*

## Notes on Part 4.

*fill in notes and documentation for part 4 as mentioned in the assignment description*


## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*
