import torch
import torch.nn as nn

# This class adds some linear layers on top of BERT and supports added features.
# It is meant to mimic the HappyTransformer implementation at 
# https://huggingface.co/transformers/_modules/transformers/models/distilbert/modeling_distilbert.html#DistilBertForSequenceClassification
class BERTLinear(nn.Module):
  def __init__(self,
               bert,
               output_dim,
               dropout,
               # Manually add features to BERT output,
               added_dim = 0):
    super().__init__()
    self.bert = bert
    # Add the dimension of extra features
    dim = bert.config.to_dict()['dim'] + added_dim # embedding dim of BERT
    self.pre_classifier = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(dim, output_dim)
    self.added_dim =added_dim
    # I noticed this line was in the Happytransformer code.
    # https://github.com/huggingface/transformers/issues/4701
    # Seems like it helps initialize the weights for non-pretrained layers
    # ex our final layers -- we might need to write a _init_weights?
    # self.init_weights() 

  def forward(self, text, added_features=None):
    # forward pass of bert; then take the output of CLS token
    embedded = self.bert(text)[0] # [4, 425, 768] = (bs, seq_len, dim)
    pooled_output = embedded[:,0] # [4, 768] = (bs, dim)
    if added_features is not None:
        assert added_features.shape[1] == self.added_dim
        pooled_output = torch.cat((pooled_output, added_features), 1)
    pooled_output = self.pre_classifier(pooled_output) # [4, 768] = (bs, dim)
    pooled_output = nn.ReLU()(pooled_output) # [4, 768] = (bs, dim)
    pooled_output = self.dropout(pooled_output) # [4, 768] = (bs, dim)
    output = self.classifier(pooled_output) # [4, 1] = (bs, output_dim)
    return output

# This class uses a RNN classifier on top of BERT embeddings.
# TODO: support added features.
class BERTRNN(nn.Module):
  def __init__(self,
               bert,
               output_dim,
               hidden_dim,
               num_layers,
               bidirectional,
               dropout):
    super().__init__()
    self.bert = bert
    dim = bert.config.to_dict()['dim'] # embedding dim of BERT
    self.rnn = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first=True)
    self.pre_classifier = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim * 2 if bidirectional else hidden_dim)
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

  def forward(self, text, added_features=None):
    # forward pass of bert; then take the output of CLS token
    embedded = self.bert(text)[0] # [4, 425, 768] = (bs, seq_len, dim)
    # pooled_output = embedded[:,0] # [4, 768] = (bs, dim)
    _, hidden = self.rnn(embedded) # (n_layers * n_directions, bs, hidden_dim)
    hidden = hidden[0] # self.rnn returns both hidden state and cell state
    # take the final outputs to pass to the next layer of classifier
    # resulting size: (bs, hidden_dim)
    if self.rnn.bidirectional:
      hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
    else:
      hidden = hidden[-1,:,:] 
    pooled_output = self.pre_classifier(hidden) # [4, h] = (bs, hidden_dim)
    pooled_output = nn.ReLU()(pooled_output) # [4, h] = (bs, hidden_dim)
    pooled_output = self.dropout(pooled_output) # [4, h] = (bs, hidden_dim)
    output = self.classifier(pooled_output) # [4, 1] = (bs, output_dim)
    return output
