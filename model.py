import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
# class LSTMTagger(nn.Module):

#     def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
#         ''' Initialize the layers of this model.'''
#         super(LSTMTagger, self).__init__()
        
#         self.hidden_dim = hidden_dim

#         # embedding layer that turns words into a vector of a specified size
#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

#         # the LSTM takes embedded word vectors (of a specified size) as inputs 
#         # and outputs hidden states of size hidden_dim
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)

#         # the linear layer that maps the hidden state output dimension 
#         # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
#         self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
#         # initialize the hidden state (see code below)
#         self.hidden = self.init_hidden()

        
#     def init_hidden(self):
#         ''' At the start of training, we need to initialize a hidden state;
#            there will be none because the hidden state is formed based on perviously seen data.
#            So, this function defines a hidden state with all zeroes and of a specified size.'''
#         # The axes dimensions are (n_layers, batch_size, hidden_dim)
#         return (torch.zeros(1, 1, self.hidden_dim),
#                 torch.zeros(1, 1, self.hidden_dim))

#     def forward(self, sentence):
#         ''' Define the feedforward behavior of the model.'''
#         # create embedded word vectors for each word in a sentence
#         embeds = self.word_embeddings(sentence)
        
#         # get the output and hidden state by passing the lstm over our word embeddings
#         # the lstm takes in our embeddings and hiddent state
#         lstm_out, self.hidden = self.lstm(
#             embeds.view(len(sentence), 1, -1), self.hidden)
        
#         # get the scores for the most likely tag for a word
#         tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
#         tag_scores = F.log_softmax(tag_outputs, dim=1)
        
#         return tag_scores

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size, num_layers = num_layers)
        # Applies a linear transformation to the data (512, 9955)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        
        # DecoderRNN method __init__ and forward, has been done mainly from logic in exercise materials
        # we have in training course, a lot of reusable logic could have been used in this project
        # few arguments were named differently, but it was only matter of renaming the arguments, 
        # rest of the logic was working 100% 

        #additional resources:
        #https://www.tensorflow.org/tutorials/text/nmt_with_attention
        #https://arxiv.org/pdf/1411.4555.pdf

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeds = self.word_embeddings(captions)
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        lstm_outputs, self.hidden = self.lstm(embeds)
        tag_scores = self.hidden2tag(lstm_outputs)
        
        return tag_scores

    
# Before executing the next code cell, you must write the sample method in the DecoderRNN class in model.py. This method should 
# accept as input a PyTorch tensor features containing the embedded input features corresponding to a single image.

# It should return as output a Python list output, indicating the predicted sentence. output[i] is a nonnegative integer that 
# identifies the predicted i-th token in the sentence. The correspondence between integers and tokens can be explored by 
# examining either data_loader.dataset.vocab.word2idx (or data_loader.dataset.vocab.idx2word).
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentences = []
        while max_len != 0:
            lstm_outputs, states = self.lstm(inputs, states)
            
            lstm_outputs = lstm_outputs.squeeze(1)
            values = self.hidden2tag(lstm_outputs)
            
            ### I've strugled with passing correct values from tensor so I had to google and search in community forums
            ### I've found logic for finding max value in Udacity Knowledge hub : knowledge.udacity.com
            value_to_pass = values.max(1)[1]
            
            # converts np array to scallar and adds it to sentences which will be returned after looping 20 times (max_len) 
            sentences.append(np.asscalar(value_to_pass))
            print('check :',np.asscalar(value_to_pass))       
           
            # now we have to prepare new input, with value_to_pass, which will serve as a new input - like "memory"
            inputs = self.word_embeddings(value_to_pass).unsqueeze(1)
            max_len = max_len - 1
         
        return sentences