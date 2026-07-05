import pandas as pd
from torch.utils.data import Dataset

class GenericNLPDataset(Dataset):
    """
    A generic Dataset class for NLP tasks.
    It takes a dataframe and a processor function to prepare the data.
    This allows for flexible data handling for various tasks like
    classification, NER, QA, etc., without needing a new Dataset class for each.
    """
    def __init__(self, dataframe, processor_fn):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe containing the data.
            processor_fn (callable): A function that takes a row of the dataframe
                                     and returns a dictionary of tensors.
        """
        self.data = dataframe
        self.processor_fn = processor_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset and processes it.
        """
        row = self.data.iloc[idx]
        return self.processor_fn(row)
