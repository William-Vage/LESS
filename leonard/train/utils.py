import numpy as np
import torch
import os


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, name):
        if self.verbose:
            print(f'Validation loss decreased ({self.loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, os.path.join(path, name))
        self.loss_min = val_loss
