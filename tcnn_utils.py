import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

class sEEG_EvalDataset(Dataset):
    def __init__(self, X, y, va, T, sr=100):
        """
        Dataset for evaluation: returns contiguous, non-overlapping T-second segments.
        
        Args:
            X: np.ndarray or tensor of shape (T_big*sr, n_channels)
            y: np.ndarray or tensor of shape (T_big*sr, n_mel_bins)
            T: segment duration (in seconds)
            sr: sampling rate (Hz)
        """
        # convert to tensors if numpy
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        self.X = X.to(dtype=torch.float32)
        self.y = y.to(dtype=torch.float32)
        self.sr = sr
        self.va = va
        self.seg_len = int(T * sr)
        self.total_len = X.shape[0]
        # Number of full contiguous segments
        self.n_segments = self.total_len // self.seg_len
    def __len__(self):
        return self.n_segments
    def __getitem__(self, idx):
        """
        Returns the idx-th contiguous T-second segment.
        """
        start = idx * self.seg_len
        end = start + self.seg_len
        sEEG = self.X[start:end]
        mel = self.y[start:end]
        va = self.va[start:end]
        return sEEG, mel, va
    
def get_fold_i(X, y, va, k, i): ## k is total folds
    """
    Return the train/val split for fold i (0 <= i < k) 
    by dividing a long continuous sequence into contiguous folds.

    Args:
        X: np.ndarray of shape (T_big*sr, n_channels)
        y: np.ndarray of shape (T_big*sr, n_mel_bins)
        T_big: total duration in seconds
        sr: sampling rate (Hz)
        k: total number of folds
        i: fold index (0 <= i < k)

    Returns:
        X_train, y_train, X_val, y_val
    """
    total_samples = X.shape[0] # int(T_big * sr)
    fold_size = total_samples // k

    if not (0 <= i < k):
        raise ValueError(f"Fold index i must be in [0, {k-1}]")

    # Validation region (contiguous time block)
    val_start = i * fold_size
    val_end = total_samples if i == k - 1 else (i + 1) * fold_size

    # Slice validation
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    va_val = va[val_start:val_end]

    # Slice training (all other regions)
    if i == 0:
        X_train = X[val_end:]
        y_train = y[val_end:]
        va_train = y[val_end:]
    elif i == k - 1:
        X_train = X[:val_start]
        y_train = y[:val_start]
        va_train = va[:val_start]
    else:
        X_train = np.concatenate((X[:val_start], X[val_end:]), axis=0)
        y_train = np.concatenate((y[:val_start], y[val_end:]), axis=0)
        va_train = np.concatenate((va[:val_start], va[val_end:]), axis=0)

    return X_train, y_train, va_train, X_val, y_val, va_val
