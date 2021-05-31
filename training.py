import constants # constants.py
import models # models.py
import dataset # dataset.py
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data 
from transformers import DistilBertModel
from sklearn.model_selection import KFold, ParameterGrid


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

def train(model, iterator, optimizer, criterion, added_feature=None):
    epoch_loss = 0
    epoch_pred = []
    epoch_true = []
    
    model.train() # Put model in training mode.

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text,added_feature).squeeze(1)
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
def evaluate(model, iterator, criterion, debug=False, added_feature=None):
    epoch_loss = 0

    model.eval()
    epoch_pred = []
    epoch_true = []

    # i = 0
    with torch.no_grad():
        for batch in iterator:
            # print(i)
            # i += 1
            predictions = model(batch.text,added_feature).squeeze(1)
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

# This function evaluates a model with a certain set of parameters
# Returns validation correlations (list with a score for each split)
# Optionally saves the weights of the best model from this experiment.
def launch_experiment(eid, train_array, n_splits, params, added_feature=None, save_weights=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss(size_average=False)
    criterion = criterion.to(device)

    valid_corrs = np.empty(n_splits)
    best_valid_loss = float('inf') 
    filename = str(eid) + "_best_valid_loss.pt"
    
    added_dim = 0
    if added_feature is not None:
        added_dim = added_feature.shape[1]
        added_features = added_feature.to(device)
    kf = KFold(n_splits=n_splits)
    fold = 0
    all_fields = dataset.get_all_fields()
    
    for train_index, valid_index in kf.split(train_array):
        print('training on fold {}'.format(fold))
        train_data = data.Dataset(train_array[train_index], all_fields)
        valid_data = data.Dataset(train_array[valid_index], all_fields)
    
        # Initialize a new model each fold.
        # https://ai.stackexchange.com/questions/18221/deep-learning-with-kfold-cross-validation-with-epochs
        # https://stats.stackexchange.com/questions/358380/neural-networks-epochs-with-10-fold-cross-validation-doing-something-wrong
        bert = DistilBertModel.from_pretrained(constants.WEIGHTS_NAME)
        model = models.BERTLinear(bert,
                                  constants.OUTPUT_DIM,
                                  params['dropout'],
                                  added_dim = added_dim)
        optimizer = optim.Adam(model.parameters(),lr=params['lr'],betas=(0.9, 0.999),eps=1e-08)
        model = model.to(device)

        train_iterator, valid_iterator = get_iterators(train_data, valid_data, params['batch_size'], device)

        for epoch in range(params['n_epochs']):
            start_time = time.time()
            train_loss, train_corr = train(model, train_iterator, optimizer, criterion, added_feature=added_feature)
            valid_loss, valid_corr = evaluate(model, valid_iterator, criterion, added_feature=added_feature)
            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
            
            if save_weights:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    # Save the weights of the model with the best valid loss
                    print('updating saved weights of best model')
                    torch.save(model.state_dict(), filename)

            print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Train Loss: {train_loss:.3f} | Train Corr: {train_corr:.2f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Corr: {valid_corr:.2f}')
    
        valid_corrs[fold] = valid_corr
        fold += 1
    return valid_corrs

# TODO: return best params
# Takes a parameter grid and search for the best model using all combinations of parameters
# The param_grid is a dictionary like the input for sklearn.model_selection.GridSearchCV
# Example: {'batch_size': [1,8], 'lr': [5e-05, 1e-05]}
# train_array should be a numpy array containing the training examples
# Each model will be evaluated using k-fold (default = 5) cross validations
# The model with the highest average correlations across all folds will be selected as the best model
# The function returns the performance of all models (k-element lists stored in dictionaries)
# These results can be used for model comparison (e.g., Wilcoxin test)
# and the best model (a tuple with the parameters and the average correlation)
def perform_hyperparameter_search(param_grid, train_array, added_feature=None, cv=constants.N_SPLITS):
    
    # Set default arguments. If the argument is not given in the parameter grid, the default will be used
    default = {'dropout': [.2], 
              'batch_size': [8],
               'lr': [5e-05],
              'n_epochs': [3]}
    for argum in default:
        if argum not in param_grid:
            param_grid[argum] = default[argum]
    
    # Use this function to expand the parameter grid
    grid = ParameterGrid(param_grid)
    
    # Place holder for model performance
    results = {}
    results_mean = {}
    
    eid = 0 # experiment id
    for params in grid:
        print(params)
        # Index of the model, represents the parameters
        index = '; '.join(x + '_' + str(y) for x, y in params.items())
        
        # Launch an experiment using the current set of parameters
        result = launch_experiment(eid,
                                   train_array,
                                   cv,
                                   params,
                                   added_feature)
        eid += 1
        
        # Store the correlation results
        results[index] = result
        results_mean[index] = np.mean(result)
    
    # Select the best results
    best_index = max(results_mean, key = results_mean.get)
    best_model = (best_index,results_mean[best_index])

    return results, best_model