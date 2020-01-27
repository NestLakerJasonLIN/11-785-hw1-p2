from torch.utils.data import Dataset
from bisect import bisect
import torch

class uniformDataset(Dataset):
    def __init__(self, x, y=None, is_test=False):
        super().__init__()
        
        self.is_test = is_test

        # train/eval dataset
        if not self.is_test:
            assert y is not None
            assert len(x) == len(y)
            self._x = x
            self._y = y
            self.utter_num = len(x) # number of utterances
            self.frame_len = 0 # total number of frames
            self.cum_utter_lens = [0] # a list of cumulative frames for each utterance
            for idx, utter in enumerate(self._x):
                utter_labels = self._y[idx]
                assert len(utter) == len(utter_labels)
                self.frame_len += len(utter)
                self.cum_utter_lens.append(self.frame_len)
        # test dataset
        else:
            self._x = x
            self.utter_num = len(x) # number of utterances
            self.frame_len = 0 # total number of frames
            self.cum_utter_lens = [0] # a list of cumulative frames for each utterance
            for utter in self._x:
                self.frame_len += len(utter)
                self.cum_utter_lens.append(self.frame_len)

    def __len__(self):
        return self.frame_len
      
    def __getitem__(self, index):
        if not self.is_test:
            utter_idx = bisect(self.cum_utter_lens, index) - 1 
            frame_idx = index - self.cum_utter_lens[utter_idx]
            x_item = torch.from_numpy(self._x[utter_idx][frame_idx])
            y_item = self._y[utter_idx][frame_idx]
            
            return x_item.float(), y_item
        else:
            utter_idx = bisect(self.cum_utter_lens, index) - 1 
            frame_idx = index - self.cum_utter_lens[utter_idx]
            x_item = torch.from_numpy(self._x[utter_idx][frame_idx])
            
            return x_item.float()
        