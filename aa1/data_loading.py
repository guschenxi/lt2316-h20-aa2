
#basics
import random
import pandas as pd
import torch
import glob
import xml.etree.ElementTree as et
import numpy as np
import matplotlib.pyplot as plt
import string


import os

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)


    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        data_df_cols = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"]
        data_df_rows = []

        ner_df_cols = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]
        ner_df_rows = []

        vocab={}
        ner_dict={'O': 0, 'drug': 1, 'group': 2, 'brand': 3, 'drug_n': 4}

        #parse xml
        path=[['TRAIN','*/*/*.xml'],['TEST','*/Test for DrugNER task/*/*.xml']]
        for path in path:
            for file in glob.iglob(os.path.join(data_dir, path[1])):
                with open(file) as f:
                    xtree = et.parse(f)
                    xroot = xtree.getroot()

                for senten in xroot:
                    sentence_id = senten.attrib.get("id")
                    sentence = senten.attrib.get("text")
                    for token in sentence.lower().strip().split():
                        token=''.join((char for char in token if char not in string.punctuation)).strip()
                        if token.strip() == "":
                            continue
                        if token not in vocab:
                            vocab[token]=len(vocab)
                        token_id = vocab[token]
                        char_start_id = sentence.lower().find(token)
                        data_df_rows.append({"sentence_id": sentence_id, "token_id": token_id, 
                                    "char_start_id": char_start_id, "char_end_id": char_start_id+len(token)-1,
                                    "split": path[0]})

                    for node in senten:
                        if node.tag == 'entity':
                            type_name = node.attrib.get("type")
                            if type_name in ner_dict: ner_id = ner_dict[type_name]
                            else: ner_id = 0
                            entity_name = node.attrib.get("text")
                            if len(entity_name.split(" ")) == 1: #entity name consists of only one word
                                ner_id = ner_dict[node.attrib.get("type")]
                                if ';' in node.attrib.get("charOffset"): # Deal with special format in Train/DrugBank/Eszopiclone_ddi.xml
                                    charOffset = [entity_name.find(token), entity_name.find(token) + len(token)-1]
                                else:
                                    charOffset = node.attrib.get("charOffset").split("-")
                                ner_df_rows.append({"sentence_id": sentence_id, "ner_id": ner_id, 
                                        "char_start_id": int(charOffset[0]), "char_end_id": int(charOffset[1]),
                                        })
                            else: #entity name consists of more than one word
                                for token in entity_name.split(" "):
                                    ner_id = ner_dict[node.attrib.get("type")]
                                    char_start_id = entity_name.find(token)
                                    ner_df_rows.append({"sentence_id": sentence_id, "ner_id": ner_id, 
                                        "char_start_id": char_start_id, "char_end_id": char_start_id + len(token),
                                        })

        #create dataframe
        self.data_df = pd.DataFrame(data_df_rows, columns = data_df_cols)
        self.ner_df = pd.DataFrame(ner_df_rows, columns = ner_df_cols)
        
        
            
        #divide VAL NEW
        train_set=self.data_df[self.data_df["split"]=="TRAIN"]
        train_ids=train_set["sentence_id"].unique()
        val_ids = np.random.choice(train_ids, size = int(len(train_ids) * 0.3))
        for ids in val_ids:
            self.data_df.loc[(self.data_df['sentence_id'] == ids),'split'] = 'VAL'
        
        self.id2word={value : key for (key, value) in vocab.items()}
        self.id2ner={value : key for (key, value) in ner_dict.items()}
        self.vocab=[[key,value] for key, value in vocab.items()]
        
        
        #create X and Y data
        self.train_sentences= []
        self.train_labels= []
        self.val_sentences= []
        self.val_labels= []
        self.test_sentences= []
        self.test_labels= []
        for sen_id in list(self.data_df["sentence_id"].unique()):
            s_tokens = self.data_df[self.data_df["sentence_id"]==sen_id]
            s_ners = self.ner_df[self.ner_df["sentence_id"]==sen_id]
            sentence=[]
            label = []
            for i,t_row in s_tokens.iterrows():
                sentence.append(t_row["token_id"])
                is_ner = False
                for i, l_row in s_ners.iterrows():
                    if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                        label.append(l_row["ner_id"])
                        is_ner = True
                if not is_ner:
                    label.append(0)

              #Split
            split=s_tokens['split'].unique().tolist()[0]
            if split == "TRAIN":
                self.train_labels.append(label)
                self.train_sentences.append(sentence)
            elif split == "VAL":
                self.val_labels.append(label)
                self.val_sentences.append(sentence)
            elif split == "TEST":
                self.test_labels.append(label)
                self.test_sentences.append(sentence)
        a=max([len(i) for i in self.train_sentences])
        b=max([len(i) for i in self.val_sentences])
        c=max([len(i) for i in self.test_sentences])
        self.max_sample_length = max([a,b,c])
        #print ("self.max_sample_length",a,b,c,self.max_sample_length)
        pass


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        
        
        #OLD why not working?
        #[x.extend((self.max_sample_length-len(x))*[-1]) for x in self.train_labels]
        #[x.extend((self.max_sample_length-len(x))*[-1]) for x in self.val_labels]
        #[x.extend((self.max_sample_length-len(x))*[-1]) for x in self.test_labels]
        x=[(i + [-1] * self.max_sample_length)[:self.max_sample_length] for i in self.train_labels]
        y=[(i + [-1] * self.max_sample_length)[:self.max_sample_length] for i in self.val_labels]
        z=[(i + [-1] * self.max_sample_length)[:self.max_sample_length] for i in self.test_labels]
        train_labels_tensor=torch.Tensor(x).to(device=self.device)
        val_labels_tensor=torch.Tensor(y).to(device=self.device)
        test_labels_tensor=torch.Tensor(z).to(device=self.device)
        output_data=[train_labels_tensor, val_labels_tensor, test_labels_tensor]
        return output_data


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        ner_counts_train = sum([sum(y>0 for y in x) for x in zip(*self.train_labels)])
        ner_counts_val = sum([sum(y>0 for y in x) for x in zip(*self.val_labels)])
        ner_counts_test = sum([sum(y>0 for y in x) for x in zip(*self.test_labels)])

        x=[ner_counts_train, ner_counts_val, ner_counts_test]
        y=["Train","Validation","Test"]
        plt.title("NER label counts for each split")
        plt.xlabel('Splits')
        plt.ylabel('Counts')
        plt.bar(y,x,color='rgb',tick_label=y)
        plt.show()
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        length_train = [len(x) for x in self.train_labels]
        length_val = [len(x) for x in self.val_labels]
        length_test = [len(x) for x in self.test_labels]
        
        x=length_train + length_val + length_test
        fig,ax = plt.subplots(1,1)
        ax.set_title("Distribution of sample lengths")
        ax.set_xlabel('Sample Lengths')
        ax.set_ylabel('Counts')
        plt.hist(x)
        plt.show()
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        Ner_count_train = [sum(y>0 for y in x) for x in self.train_labels]
        Ner_count_val = [sum(y>0 for y in x) for x in self.val_labels]
        Ner_count_test = [sum(y>0 for y in x) for x in self.test_labels]
        
        x = Ner_count_train + Ner_count_val + Ner_count_test
        counts,values = pd.Series(x).value_counts().values, pd.Series(x).value_counts().index
        plt.figure(figsize=(15,5))
        plt.title("Distribution of number of NERs in sentences")
        plt.xlabel('Number of NERs in sentences')
        plt.ylabel('Counts')
        plt.bar(values,counts,tick_label=values)
        plt.show()
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass



