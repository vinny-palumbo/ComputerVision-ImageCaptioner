import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Remove classification head
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
    
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Convert captions of vocab_size to embeddings of embed_size to input to the LSTM along with the image embeddings
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        
        # LSTM taking inputs of embed_size and generating outputs of hidden_size
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first = True)
        
        # Fully-connected layer taking the LSTM output of hidden_size and generating a vector of vocab_size to predict the next word
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)


    def forward(self, features, captions):
    
        # Create embeddings for the captions that match the image embeddings
        captions = captions[:, :-1]
        captions = self.embedding(captions)
        
        # Concatenate embeddings of the image features and the captions
        features = features.unsqueeze(dim=1)
        inputs = torch.cat((features, captions), dim=1)
        
        # Pass embeddings through the decoder
        out, _ = self.lstm(inputs)
        out = self.fc(out)
        
        return out



    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentence_idx = []
        
        for i in range(max_len):
        
            # Pass embeddings through the decoder
            decoder_out, states = self.lstm(inputs, states)
            decoder_out = self.fc(decoder_out)
            
            # Get word_idx with highest probability
            _, word_idx = decoder_out.max(dim=2)
            
            # append word_idx to output list
            sentence_idx.append(word_idx.item())
            
            # break if <END> word idx predicted
            if word_idx.item() == 1: break
            
            # embed predicted word idx and pass as input to decoder for next prediction
            inputs = self.embedding(word_idx)
        
        return sentence_idx
        
        
        
        
        
        
        
        
        
        