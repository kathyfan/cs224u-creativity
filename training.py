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