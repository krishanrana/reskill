from torch.utils.data import Dataset
from reskill.utils.general_utils import AttrDict
import numpy as np
import os

class SkillsDataset(Dataset):

    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, dataset_name, phase, subseq_len, transform):        
        self.phase = phase
        self.subseq_len = subseq_len
        curr_dir = os.path.dirname(__file__)
        fname = os.path.join(curr_dir, "../../dataset/" + dataset_name + "/demos.npy")
        self.seqs = np.load(fname, allow_pickle=True)
        self.transform = transform

        self.n_seqs = len(self.seqs)
        print("Dataset size: ", self.n_seqs)

 
        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        elif self.phase == "test":
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs  



    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx = np.random.randint(0, (len(seq.actions)-self.subseq_len-1))
        actions = np.array(seq.actions[start_idx:start_idx+self.subseq_len], dtype=np.float32)
        obs = np.array(seq.obs[start_idx:start_idx+self.subseq_len], dtype=np.float32)

        output = AttrDict(obs=obs,
                          actions=actions
                         )
        return output 

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])
    
    def __len__(self):
        return int(self.end-self.start)

    
        
