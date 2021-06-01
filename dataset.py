import pandas as pd
import torch
from torchtext.legacy import data 
from transformers import DistilBertTokenizer

import constants # constants.py

# Read the data for a single study and single metric (e.g., "Novelty_Combined")
# Optionally shuffles the data before returning dataframe
# Returns a df with a column named 'text' and a column named 'label'
def read_data(study, metric, shuffle=True):
    sheet_df = pd.read_excel("Idea Ratings_Berg_2019_OBHDP.xlsx", sheet_name=study-1) 
    sheet_df.dropna(inplace=True)
    data_df = sheet_df[['Final_Idea', metric]].rename(columns={'Final_Idea': 'text', metric: 'label'})

    if shuffle:
        data_df = data_df.sample(frac=1)
    return data_df

# Take a list with the numbers of studies, and a specific metric
# Extract multiple datasets with get_data and concatenate them
def read_multiple_datasets(study_list, metric, shuffle = True):
    dfs = [read_data(study, metric, shuffle) for study in study_list]
    return pd.concat(dfs)

def get_tokenizer():
    return DistilBertTokenizer.from_pretrained(constants.WEIGHTS_NAME)

# Due to efficiency, create a global tokenizer so that we don't have to get
# a new tokenizer every time get_tokens is applied to each sentence.
TOKENIZER = get_tokenizer()
# Apply tokenization and some preprocessing steps to the input sentence.
# Namely, this trims examples down to MAX_INPUT_LENGTH. (There is a -2 
# since the [CLS] and [SEP] tokens will be added) and removes slashes.
def get_tokens(sentence):
    # BERT input can be at most 512 words
    MAX_INPUT_LENGTH = TOKENIZER.max_model_input_sizes[constants.WEIGHTS_NAME]
    sentence = sentence.replace('/', '') # remove slashes
    tokens = TOKENIZER.tokenize(sentence) 
    tokens = tokens[:MAX_INPUT_LENGTH-2]
    return tokens

# text_fields defines preprocessing and handling of the text of an example.
def get_text_fields():
    tokenizer = get_tokenizer()

    return data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = get_tokens,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = tokenizer.cls_token_id, # add [CLS] token
                  eos_token = tokenizer.sep_token_id, # add [SEP] token
                  pad_token = tokenizer.pad_token_id,
                  unk_token = tokenizer.unk_token_id)


# label_fields defines how to handle the label of an example.
# for regression, we do not need to build a vocabulary.
def get_label_fields():
    return data.LabelField(sequential=False, use_vocab=False, dtype = torch.float)
    

# Returns list containing text_fields and label_fields
# Assumes csv data has columns in order of [text, label] 
# If the data has a column for added feature named "add", create a field for it.
def get_all_fields(add= False):
    if add:
        ADD = data.Field(use_vocab = False,
                        dtype = torch.float,
                        sequential=False,
                        batch_first = True)
        return [('text', get_text_fields()), ('label', get_label_fields()), ('add', ADD)]
    else:
        return [('text', get_text_fields()), ('label', get_label_fields())]

# Returns train_dataset, test_dataset
def get_train_test_datasets(train_file, test_file, add=False):
    return data.TabularDataset.splits(
        path='', # path='' because the csvs are in the same directory
        train=train_file, test=test_file, format='csv',
        fields=get_all_fields(add=add))
