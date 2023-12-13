from torch.utils.data import Dataset

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
    """
    
    def __init__(self, root_dir, exercise, subject):
        
        if not subject in _subject_types:
            raise ValueError('Unrecognized subject type')

        if not exercise in _exercise_range:
            raise ValueError('Unrecognized exercise index')
       
        self.root_dir = root_dir
        self.subject = subject
        self.exercise = exercise

        self.dir = f'{self.root_dir}/{_subject_types[self.subject]}/{self.exercise}'
        self._load_from_directory(self.dir)

    def _load_from_directory(self, dir):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


    def __getitem__(self, idx):
        raise NotImplementedError
