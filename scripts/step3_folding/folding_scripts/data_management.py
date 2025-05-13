from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    # A PyTorch Dataset class to manage protein sequences loaded from a DataFrame.
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a tuple: (row_dict, index)
        return self.data.iloc[idx].to_dict(), idx

#helper function, dont remember what its for. 
def custom_collate_fn(batch):
    # batch is a list of (row_dict, idx) tuples
    rows, indices = zip(*batch)
    return list(rows), list(indices)