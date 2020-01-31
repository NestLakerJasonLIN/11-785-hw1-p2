from torch.utils.data import Dataset
from bisect import bisect
import torch
import numpy as np

class uniformDataset(Dataset):
    def __init__(self, x, y=None, is_test=False):
        super().__init__()
        
        # self.is_test = is_test
        # self._x = np.concatenate(x)
        # if not is_test:
        #     self._y = np.concatenate(y)
        self.is_test = is_test
        self._x = x
        self._y = y
        self._build_fields()

    # internal usage for building fields. Only valid for uniformly sampled dataset
    def _build_fields(self):
        # train/eval dataset
        if not self.is_test:
            assert self._y is not None
            assert len(self._x) == len(self._y)

            self.utter_num = len(self._x) # number of utterances
            self.frame_len = 0 # total number of frames
            self.cum_utter_lens = [0] # a list of cumulative frames for each utterance
            for idx, utter in enumerate(self._x):
                utter_labels = self._y[idx]
                assert len(utter) == len(utter_labels)
                self.frame_len += len(utter)
                self.cum_utter_lens.append(self.frame_len)
        # test dataset
        else:
            self.utter_num = len(self._x) # number of utterances
            self.frame_len = 0 # total number of frames
            self.cum_utter_lens = [0] # a list of cumulative frames for each utterance
            for utter in self._x:
                self.frame_len += len(utter)
                self.cum_utter_lens.append(self.frame_len)

        self.utter_index_table = np.ones(self.frame_len).astype(int)

        # generate table mapping dataset_idx -> utter idx
        for idx in range(len(self.cum_utter_lens) - 1):
            self.utter_index_table[self.cum_utter_lens[idx]: self.cum_utter_lens[idx + 1]] = idx

    def _decode_index(self, index):
        utter_idx = bisect(self.cum_utter_lens, index) - 1
        frame_idx = index - self.cum_utter_lens[utter_idx]
        return utter_idx, frame_idx

    def _decode_index_table(self, index):
        utter_idx = self.utter_index_table[index]
        frame_idx = index - self.cum_utter_lens[utter_idx]
        return utter_idx, frame_idx

    def __len__(self):
        # return len(self._x)
        return self.frame_len
      
    def __getitem__(self, index):
        # v2: one flat ndarray
        # x_item = torch.from_numpy(self._x[index]).float()
        # y_item = self._y[index]
        # if not self.is_test:
        #     return x_item, y_item
        # else:
        #     return x_item

        # v3: mapping table
        utter_idx, frame_idx = self._decode_index_table(index)

        # v1: bisect
        # utter_idx, frame_idx = self._decode_index(index)
        x_item = torch.from_numpy(self._x[utter_idx][frame_idx]).float()
        if not self.is_test:
            y_item = self._y[utter_idx][frame_idx]
            return x_item, y_item
        else:
            return x_item

# uniformly sampling utterances and use context padding
class contextUniformDataset(uniformDataset):
    def __init__(self, x, y=None, is_test=False, context_size = 10):
        super(contextUniformDataset, self).__init__(x, y, is_test)
        self._context_size = context_size

    def __getitem__(self, index):
        # v2: flat ndarray
        # x_item = build_context_features(self._x, index, self._context_size)
        # if not self.is_test:
        #     y_item = self._y[index]
        #     return x_item, y_item
        # else:
        #     return x_item

        # v3: mapping table
        utter_idx, frame_idx = self._decode_index_table(index)

        # v1: bisect
        # utter_idx, frame_idx = self._decode_index(index)
        x_item = build_context_features(self._x[utter_idx], frame_idx, self._context_size)
        if not self.is_test:
            y_item = self._y[utter_idx][frame_idx]
            return x_item, y_item
        else:
            return x_item



# ===================== helper function ====================
# given an utterance and a frame index, build context features from original feature
# return constructed features with context of context_size
def build_context_features(utterance, frame_idx, context_size):
    # pad with zero
    prev_shape = utterance.shape
    utterance = np.pad(utterance, ((context_size, context_size), (0, 0)),
                       "constant", constant_values=0)

    assert utterance.shape == (prev_shape[0] + context_size * 2, prev_shape[1])

    # index shift right by context_size
    x = utterance[frame_idx]
    before = utterance[frame_idx: frame_idx + context_size]
    after = utterance[frame_idx + 1 + context_size: frame_idx + 2 * context_size + 1]
    before, after = before.reshape(-1), after.reshape(-1)

    concat_x = np.concatenate((before, x, after), axis=0)

    assert concat_x.shape == ((2 * context_size + 1) * x.shape[0],)

    return torch.from_numpy(concat_x).float()