import numpy as np
import torch
from torchtext.legacy import data 

# Get train and validation iterators.
# Given train and validation datasets, returns 2 iterators.
# train_data and valid_data should be torchtext datasets
# with a 'text' field.
def get_iterators(train_data, valid_data, batch_size, device):
    return data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size = batch_size,
        device = device,
        # Below are needed to overcome error when calling evaluate():
        # TypeError: '<' not supported between instances of 'Example' and 'Example'
        sort_key = lambda x: len(x.text),
        sort_within_batch = False,
    )

# Get a test iterator.
# (Technically this function can be used to get a single iterator of any dataset.)
def get_iterator(test_data, batch_size, device):
    return data.BucketIterator(
        test_data,
        batch_size = batch_size,
        device = device,
        # Below are needed to overcome error when calling evaluate():
        # TypeError: '<' not supported between instances of 'Example' and 'Example'
        sort_key = lambda x: len(x.text),
        sort_within_batch = False,
    )

# def train(model, iterator, optimizer, criterion):
#     epoch_loss = 0
#     epoch_corr = 0
  
#     model.train() # Put model in training mode.

#     for batch in iterator:
#         optimizer.zero_grad()
#         predictions = model(batch.text).squeeze(1)
#         loss = criterion(predictions, batch.label)
#         # need to use detach() since `predictions` requires gradient
#         # alternative: scipy.stats.pearsonr? (might be more memory efficient,
#         # but not sure which one is more efficient to compute)
#         corr = np.corrcoef(batch.label.cpu().data.numpy(), predictions.detach().cpu().data.numpy())
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()
#         # corr is a (2,2) matrix, so we just get the top right element.
#         # If the correlation is a nan value, replace with 0, which means
#         # no correlation.
#         corr_value = corr[0][1].item()
#         if np.isnan(corr[0][1]):
#             corr_value = 0

#         epoch_corr += corr_value

#     return epoch_loss / len(iterator), epoch_corr / len(iterator)

# # Evaluate the model on a validation or test set.
# # Use debug=True to print more detailed info.
# def evaluate(model, iterator, criterion, debug=False):
#     epoch_loss = 0
#     epoch_corr = 0

#     model.eval()

#     # i = 0
#     with torch.no_grad():
#         for batch in iterator:
#             # print(i)
#             # i += 1
#             predictions = model(batch.text).squeeze(1)
#             if debug:
#                 print('predictions: {}'.format(predictions)) 
#                 print('true labels: {}'.format(batch.label))
#             loss = criterion(predictions, batch.label)
#             corr = np.corrcoef(batch.label.cpu().data.numpy(), predictions.cpu().data.numpy())
#             epoch_loss += loss.item()

#             # If the correlation is a nan value, replace with 0, which means
#             # no correlation.
#             corr_value = corr[0][1].item()
#             if np.isnan(corr[0][1]):
#                 corr_value = 0

#             epoch_corr += corr_value

#     return epoch_loss / len(iterator), epoch_corr / len(iterator)

def train(model, iterator, optimizer, criterion, added_features = None):
    epoch_loss = 0
    epoch_pred = []
    epoch_true = []
    
    model.train() # Put model in training mode.

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text,added_features).squeeze(1)
        loss = criterion(predictions, batch.label)
        # need to use detach() since `predictions` requires gradient
        # alternative: scipy.stats.pearsonr? (might be more memory efficient,
        # but not sure which one is more efficient to compute)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_pred.extend(predictions.tolist())
        epoch_true.extend(batch.label.tolist())

    corr = np.corrcoef(epoch_pred, epoch_true)
    corr_value = corr[0][1].item()
    if np.isnan(corr[0][1]):
            corr_value = 0

    return epoch_loss / len(iterator), corr_value

# Evaluate the model on a validation or test set.
# Use debug=True to print more detailed info.
def evaluate(model, iterator, criterion, debug=False, added_features = None):
    epoch_loss = 0

    model.eval()
    epoch_pred = []
    epoch_true = []

    # i = 0
    with torch.no_grad():
        for batch in iterator:
            # print(i)
            # i += 1
            predictions = model(batch.text,added_features).squeeze(1)
            if debug:
                print('predictions: {}'.format(predictions)) 
                print('true labels: {}'.format(batch.label))
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_pred.extend(predictions.tolist())
            epoch_true.extend(batch.label.tolist())

            # If the correlation is a nan value, replace with 0, which means
            # no correlation.

    corr = np.corrcoef(epoch_pred, epoch_true)
    corr_value = corr[0][1].item()
    if np.isnan(corr[0][1]):
            corr_value = 0

    return epoch_loss / len(iterator), corr_value