
import os
import torch

import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


class Trainer:


    def __init__(self, dump_folder="/tmp/aa2_models/"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self, model_path):
        # Finish this function so that it loads a model and return the appropriate variables
        
        checkpoint = torch.load(model_path)
        
        return checkpoint


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        # Finish this function so that it set up model then trains and saves it.
        hyperparamaters = hyperparamaters
        lr = hyperparamaters["learning_rate"]
        num_layers = hyperparamaters["number_layers"]
        dropout = hyperparamaters["dropout"]
        batch_size = hyperparamaters["batch_size"]
        epochs = hyperparamaters["epochs"]
        model_name = hyperparamaters["model_name"]
        hidden_dim = hyperparamaters["hidden_size"]
        embedding_dim = hyperparamaters["embedding_dim"]
        device = hyperparamaters["device"]
        #optimizer = hyperparamaters["optimizer"]
        
                
        #print("trainx_shape:", train_X.shape)
        #input_size = int(train_X.max()) + 1 
        
        input_size = 7 # number of features

        output_dim = 6 # number of ner labels
        
        self.device = device
        b = Batcher(train_X, train_y, self.device, batch_size=batch_size, max_iter=epochs)
        model = model_class(input_size, embedding_dim, hidden_dim, output_dim, num_layers, self.device, dropout)
        model = model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss()
        
        print(model)
        print("Training... \n")
        
        model.train()
        epoch = 0
        for split in b:
            
            tot_loss = 0
            for sent, label in split:
                optimizer.zero_grad()
                pred = model(sent.float().to(self.device))
                loss = criterion(pred.permute(0,2,1), label.long())
                #loss = criterion(pred.view(pred.shape[0]*pred.shape[1],-1), label.view(-1).long())
                tot_loss += loss
                loss.backward()
                optimizer.step()
            print("Total loss in epoch {} is {}.".format(epoch+1, tot_loss))
            epoch += 1
           
        print(f'Training Finished\n')
        
        print("Evaluating... \n")

        model.eval()
        y_label = []
        y_pred = []
        test_batches = Batcher(val_X, val_y, self.device, batch_size=batch_size, max_iter=1)
        for split in test_batches:
            for sent, label in split:
                
                with torch.no_grad():
                    pred = model(sent.float().to(self.device))
                    #print(pred)
                    for i in range(pred.shape[0]):
                        pred_sent = pred[i]
                        label_sent = label[i]
                        for j in range(len(pred_sent)):
                            #print(pred_sent[j])
                            pred_value = int(torch.argmax(pred_sent[j]))
                            label_true = int(label_sent[j])
                            
                            y_pred.append(pred_value)
                            y_label.append(label_true)

        scores = {}
        print(max(y_label))
        print(max(y_pred))
        print(y_label[:50])
        print(y_pred[:50])
        accuracy = accuracy_score(y_label, y_pred, normalize=True)
        scores['accuracy'] = accuracy
        recall = recall_score(y_label, y_pred, average='weighted')
        scores['recall'] = recall
        precision = precision_score(y_label, y_pred, average='weighted')
        scores['precision'] = precision
        f1 = f1_score(y_pred, y_label, average='weighted')
        scores['f1_score'] = f1
        scores['loss'] = int(tot_loss)


        print('model:', model_name, 'Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F1_score:', f1)

        #print(f'Evaluation Finished\n')
        #self.save_model(epochs, model, optimizer, tot_loss, scores, hyperparamaters, model_name)
        
        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        pass

    
class Batcher:
    def __init__(self, X, y, device, batch_size=50, max_iter=None):
        self.X = X
        self.y = y
        self.device = device
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.curr_iter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_iter == self.max_iter:
            raise StopIteration
        permutation = torch.randperm(self.X.size()[0], device=self.device)
        permX = self.X[permutation]
        permy = self.y[permutation]
        splitX = torch.split(permX, self.batch_size)
        splity = torch.split(permy, self.batch_size)
        
        self.curr_iter += 1
        return zip(splitX, splity)