import torch.nn as nn

# This class adds some linear layers on top of BERT.
# It is meant to mimic the HappyTransformer implementation at 
# https://huggingface.co/transformers/_modules/transformers/models/distilbert/modeling_distilbert.html#DistilBertForSequenceClassification
class BERTLinear(nn.Module):
  def __init__(self,
               bert,
               output_dim,
               dropout):
    super().__init__()
    self.bert = bert
    dim = bert.config.to_dict()['dim'] # embedding dim of BERT
    self.pre_classifier = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(dim, output_dim)
    
    # I noticed this line was in the Happytransformer code.
    # https://github.com/huggingface/transformers/issues/4701
    # Seems like it helps initialize the weights for non-pretrained layers
    # ex our final layers -- we might need to write a _init_weights?
    # self.init_weights() 

  def forward(self, text):
    # forward pass of bert; then take the output of CLS token
    embedded = self.bert(text)[0] # [4, 425, 768] = (bs, seq_len, dim)
    pooled_output = embedded[:,0] # [4, 768] = (bs, dim)
    pooled_output = self.pre_classifier(pooled_output) # [4, 768] = (bs, dim)
    pooled_output = nn.ReLU()(pooled_output) # [4, 768] = (bs, dim)
    pooled_output = self.dropout(pooled_output) # [4, 768] = (bs, dim)
    output = self.classifier(pooled_output) # [4, 1] = (bs, output_dim)
    return output