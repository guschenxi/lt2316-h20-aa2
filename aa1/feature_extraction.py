
#basics
import pandas as pd
import torch
import string
alphabet=dict(zip(string.ascii_lowercase, range(1, 27)))

# Feel free to add any new code to this script


def extract_features(data:pd.DataFrame, max_sample_length:int, device, id2word):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    train_sentences= []
    val_sentences= []
    test_sentences= []

    for sen_id in list(data["sentence_id"].unique()):
        sentence=[]
        s_tokens = data[data["sentence_id"]==sen_id]
        
        seq=list(set(s_tokens['token_id']))
                 
        for item in seq:
            #Get token_id
            token_id=item
            word=id2word[item]
            #print(word, len(word))
                 
            #Get word length
            word_len=len(word)
            if word_len==0:
                continue
            
            #Get last character
            if word[-1] in string.ascii_lowercase:
                last_char=alphabet[word[-1]]
            else:
                last_char=0
            
            #Get previous token_id and next token_id
            if seq.index(token_id) > 0:
                 pre_token_id = seq[seq.index(token_id)-1]
            else:
                 pre_token_id = 0
            if seq.index(token_id) < len(seq)-1:
                 next_token_id = seq[seq.index(token_id)+1]
            else:
                 next_token_id = 0
                 
            sentence.append([token_id, 
                             pre_token_id, 
                             next_token_id, 
                             word_len, 
                             last_char
                            ])

           
        split=s_tokens['split'].unique().tolist()[0]
        if split == "TRAIN":
            train_sentences.append(sentence)
        elif split == "VAL":
            val_sentences.append(sentence)
        elif split == "TEST":
            test_sentences.append(sentence)
            
        C=[0,0,0,0,0]
        for x in train_sentences:
            for i in range(max_sample_length-len(x)):
                x.append(C)
        for y in val_sentences:
            for i in range(max_sample_length-len(y)):
                y.append(C)
        for z in test_sentences:
            for i in range(max_sample_length-len(z)):
                z.append(C)
    
    train_sentences_tensor=torch.LongTensor(train_sentences).to(device=device)
    val_sentences_tensor=torch.LongTensor(val_sentences).to(device=device)
    test_sentences_tensor=torch.LongTensor(test_sentences).to(device=device)

    output_data=[train_sentences_tensor, val_sentences_tensor, test_sentences_tensor]
    print("TRAIN Tensor Size:", train_sentences_tensor.size())
    print("VAL Tensor Size:", val_sentences_tensor.size())
    print("TEST Tensor Size:", test_sentences_tensor.size())
    return output_data
    
