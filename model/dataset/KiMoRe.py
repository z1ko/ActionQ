import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L

# All available subjects in the dataset
_subject_types = {
    'expert': 'CG/Expert', 
    'non-expert': 'CG/NotExpert', 
    'stroke': 'GPP/Stroke', 
    'parkinson': 'GPP/Parkinson', 
    'backpain': 'GPP/BackPain'
}

# All available exercise types in the dataset
_exercise_range = range(0, 5)

def KiMoReDataset(Dataset):
    """
        KInematic Assessment of MOvement and Clinical Scores for 
        Remote Monitoring of Physical REhabilitation

        Each dataset item is a temporal skeleton evolution 
        with a quality scores assigned
    """
    
    def __init__(self, root_dir, exercise, subjects):
        
        if not subjects in _subject_types:
            raise ValueError('Unrecognized subject type')

        if not exercise in _exercise_range:
            raise ValueError('Unrecognized exercise index')
       
        self.root_dir = root_dir
        self.subjects = subjects
        self.exercise = exercise

        self.dirs = [ f'{self.root_dir}/{_subject_types[s]}/{self.exercise}' for s in subjects]
        for dir in self.dirs:
            self._load_from_directory(dir)

    def _load_from_directory(self, dir):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


    def __getitem__(self, idx):
        raise NotImplementedError


def KiMoReDataModule(L.LightningDataModule):
    def __init__(self, root_dir, batch_size, exercise, subjects):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.exercise = exercise
        self.subjects = subjects

    def setup(self, _stage: str):
        self.dataset_total = KiMoReDataset(self.root_dir, self.exercise, self.subjects)
        self.dataset_train, self.dataset_val = random_split(
            self.dataset_total, 
            [0.8, 0.2], 
            torch.Generator().manual_seed(69)
        ) # TODO: Remove manual seed

    def train_dataloader(self):
        return DataLoader(self.dataset_train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, self.batch_size)




