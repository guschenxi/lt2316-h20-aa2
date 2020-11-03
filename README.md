## LT2316 H20 Assignment A2 : Ner Classification

Name: *XI CHEN* (if you don't want to use your real name with your current GitHub account, you will have to make another GitHub account)

## Notes on Part 1.

*fill in notes and documentation for part 1 as mentioned in the assignment description*

I have tried to set up many different models CNN, LSTM, GRU. When I used embedding, then they didn't work, and I couldn't figure out why. I documented the errors below.

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


Which is right? Or does both work?
loss = criterion(pred.permute(0,2,1), label.long())
loss = criterion(pred.view(pred.shape[0]*pred.shape[1],-1), label.view(-1).long())

## Notes on Part 2.

*fill in notes and documentation for part 2 as mentioned in the assignment description*

I tried to use both Nllloss together with Log_softmax and CrossEntropy without softmax function. The scores looked fine, however, I'm not very confident about that everything I made was in the correct way.

Due to that there are still small bugs in my AA1 part, I tried my models on a classmates' AA1 part, and some of the results looked better, but still not perfect in my opinion.

## Notes on Part 3.

*fill in notes and documentation for part 3 as mentioned in the assignment description*

The results varied every time I run it, because the batches were randomly created. However, it seemed that lower learning rate, such as 0.001 and 0.0025, got better results.

I got an error when running the code for parallel coordination plot:
TypeError: comparison not implemented
It was because there was a non-numeric column in the pd frame.


## Notes on Part 4.

*fill in notes and documentation for part 4 as mentioned in the assignment description*

Model: model1 Accuracy: 0.9650851266096275 Precision: 0.946543022025391 Recall: 0.9650851266096275 F1_score: 0.9746243047001547


## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*
