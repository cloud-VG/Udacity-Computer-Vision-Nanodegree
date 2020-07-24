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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size) 
        
        # recurrent layer
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.2)
        
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # remove <END> token
        captions = captions[:, :-1]
        
        # embedding captions
        embed = self.embedding(captions)
        
        # concatinate features and captions
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        
        # rnn
        output, _ = self.rnn(embed) 
        
        # linear
        output = self.dropout(output)
        output = self.fc(output)
        
        return output
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        for _ in range(max_len):
            output, states = self.rnn(inputs, states)
            output = self.fc(output.squeeze(1))
            token = output.max(1)[1]
            caption.append(token.item())
            inputs = self.embedding(token).unsqueeze(1)
        return caption