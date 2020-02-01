from torch.utils.data import Dataset
import torch
import numpy as np

class uniformDataset(Dataset):
    def __init__(self, x, y=None, is_test=False):
        super().__init__()

        self.is_test = is_test
        self._x = x
        self._y = y
        self._build_fields()

    # internal usage for building fields. Only valid for uniformly sampled dataset
    def _build_fields(self, context_size=0):
        # train/eval dataset
        if not self.is_test:
            assert self._y is not None
            assert len(self._x) == len(self._y)

            self.utter_num = len(self._x) # number of utterances
            self.frame_len = 0 # total number of frames
            self.cum_utter_lens = [0] # a list of cumulative frames for each utterance
            for idx, utter in enumerate(self._x):
                utter_labels = self._y[idx]
                assert len(utter)-2*context_size == len(utter_labels)
                self.frame_len += (len(utter)-2*context_size)
                self.cum_utter_lens.append(self.frame_len)
        # test dataset
        else:
            self.utter_num = len(self._x) # number of utterances
            self.frame_len = 0 # total number of frames
            self.cum_utter_lens = [0] # a list of cumulative frames for each utterance
            for utter in self._x:
                self.frame_len += (len(utter)-2*context_size)
                self.cum_utter_lens.append(self.frame_len)

        # generate table mapping dataset_idx -> utter idx
        self.utter_index_table = np.ones(self.frame_len).astype(int)
        for idx in range(len(self.cum_utter_lens) - 1):
            self.utter_index_table[self.cum_utter_lens[idx]: self.cum_utter_lens[idx + 1]] = idx

    def _decode_index(self, index):
        utter_idx = self.utter_index_table[index]
        frame_idx = index - self.cum_utter_lens[utter_idx]
        return utter_idx, frame_idx

    def __len__(self):
        return self.frame_len
      
    def __getitem__(self, index):
        utter_idx, frame_idx = self._decode_index(index)

        x_item = torch.from_numpy(self._x[utter_idx][frame_idx]).float()
        if not self.is_test:
            y_item = self._y[utter_idx][frame_idx]
            return x_item, y_item
        else:
            return x_item

# uniformly sampling utterances and use context padding
class contextUniformDataset(uniformDataset):
    def __init__(self, x, y=None, is_test=False, context_size=10):
        super(uniformDataset, self).__init__()
        self._x = x
        self._y = y
        self.is_test = is_test

        # pad with zero
        for idx in range(len(self._x)):
            prev_shape = self._x[idx].shape
            self._x[idx] = np.pad(self._x[idx], ((context_size, context_size), (0, 0)),
                               "constant", constant_values=0)
            assert self._x[idx].shape == (prev_shape[0] + context_size * 2, prev_shape[1])

        self._context_size = context_size
        self._build_fields(context_size=context_size)

    def __getitem__(self, index):
        utter_idx, frame_idx = self._decode_index(index)

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
    # index shift right by context_size
    x = utterance[context_size + frame_idx]
    before = utterance[frame_idx: frame_idx + context_size]
    after = utterance[frame_idx + 1 + context_size: frame_idx + 2 * context_size + 1]
    before, after = before.reshape(-1), after.reshape(-1)

    concat_x = np.concatenate((before, x, after), axis=0)

    assert concat_x.shape == ((2 * context_size + 1) * x.shape[0],)

    return torch.from_numpy(concat_x).float()