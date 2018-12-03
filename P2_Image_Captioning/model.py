import torch
import torch.nn as nn
import torchvision.models as models


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
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        drop_prob=0.4
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.n_hidden = hidden_size
        self.n_layers = num_layers 
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # initialize the weights
        self.init_weights()
        
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())
        
    def forward(self, features, captions):
        
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        outputs, _ = self.lstm(inputs)
        outputs = self.dropout(outputs)        
        outputs = self.fc(outputs)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        sampled_ids = [] 
        for i in range(max_len): 
            hiddens, states = self.lstm(inputs, states) 
            outputs = self.fc(hiddens.squeeze(1)) 
            predicted = outputs.max(1)[1] 
            sampled_ids.append(predicted.data[0].item()) 
            inputs = self.embed(predicted) 
            inputs = inputs.unsqueeze(1) 
        return sampled_ids