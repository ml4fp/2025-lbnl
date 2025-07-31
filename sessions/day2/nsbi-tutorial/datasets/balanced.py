import os
import pickle
import re

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

from physics.simulation import mcfm, msq
from physics.analysis import zz4l, zz2l2v

class BalancedDataModule(L.LightningDataModule):

    def __init__(self, numerator_events: str = '', denominator_events: str = '', numerator_reweight : tuple = None, denominator_reweight : tuple = None, features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], batch_size: int = 32, random_state: int=None, data_dir : str = './'):
        super().__init__()

        self.features = features

        self.numerator_file = numerator_events
        self.denominator_file = denominator_events
        self.numerator_rwt = numerator_reweight
        self.denominator_rwt = denominator_reweight

        self.batch_size = batch_size
        self.random_state = random_state

        self.data_dir = data_dir
        self.scaler = StandardScaler()

    def prepare_data(self):

        if isinstance(self.numerator_file, mcfm.Process):
            events_numerator = self.numerator_file
        else:
            events_numerator = mcfm.from_csv(cross_section=None, file_path=self.numerator_file, kinematics=self.features)

        if isinstance(self.denominator_file, mcfm.Process):
            events_denominator = self.denominator_file
        else:
            events_denominator = mcfm.from_csv(cross_section=None, file_path=self.denominator_file, kinematics=self.features)

        if self.numerator_rwt is not None:
            events_numerator = events_numerator.reweight(denominator=self.numerator_rwt[0], numerator=self.numerator_rwt[1])
        if self.denominator_rwt is not None:
            events_denominator = events_denominator.reweight(denominator=self.denominator_rwt[0], numerator=self.denominator_rwt[1])

        train_size, val_size, test_size = 6, 2, 2
        events_numerator_train, events_numerator_val, events_numerator_test = events_numerator.split(train_size=train_size, val_size=val_size, test_size=test_size)
        events_denominator_train, events_denominator_val, events_denominator_test = events_denominator.split(train_size=train_size, val_size=val_size, test_size=test_size)

        self.training_data = BalancedDataset(events_numerator_train, events_denominator_train, self.features, scaler = None, random_state = self.random_state)
        self.scaler.fit(self.training_data.X)

        # save stuff for later
        with open(os.path.join(self.data_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(self.data_dir, 'events_numerator_train.pkl'), 'wb') as f:
            pickle.dump(events_numerator_train, f)
        with open(os.path.join(self.data_dir, 'events_denominator_train.pkl'), 'wb') as f:
            pickle.dump(events_denominator_train, f)
        with open(os.path.join(self.data_dir, 'events_numerator_val.pkl'), 'wb') as f:
            pickle.dump(events_numerator_val, f)
        with open(os.path.join(self.data_dir, 'events_denominator_val.pkl'), 'wb') as f:
            pickle.dump(events_denominator_val, f)
        with open(os.path.join(self.data_dir, 'events_numerator_test.pkl'), 'wb') as f:
            pickle.dump(events_numerator_test, f)
        with open(os.path.join(self.data_dir, 'events_denominator_test.pkl'), 'wb') as f:
            pickle.dump(events_denominator_test, f)

    def setup(self, stage: str):

        if stage == 'fit':

            with open(os.path.join(self.data_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_numerator_train.pkl'), 'rb') as f:
                events_numerator_train = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_train.pkl'), 'rb') as f:
                events_denominator_train = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_numerator_val.pkl'), 'rb') as f:
                events_numerator_val = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_val.pkl'), 'rb') as f:
                events_denominator_val = pickle.load(f)

            self.training_data = BalancedDataset(events_numerator_train, events_denominator_train, self.features, scaler = self.scaler, random_state=self.random_state)
            self.validation_data = BalancedDataset(events_numerator_val, events_denominator_val, self.features, scaler = self.scaler, random_state=self.random_state)

        elif stage == 'test':
            with open(os.path.join(self.data_dir, 'events_numerator_test.pkl'), 'rb') as f:
                events_numerator_test = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_test.pkl'), 'rb') as f:
                events_denominator_test = pickle.load(f)

            self.testing_data = BalancedDataset(events_numerator_test, events_denominator_test, self.features, scaler = self.scaler, random_state=self.random_state)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=8)

class BalancedDataset(Dataset):
    def __init__(self, events_numerator = None, events_denominator = None, features = None, scaler = None, random_state = None):
        super().__init__()

        # get features
        X_numerator = events_numerator.kinematics[features].to_numpy()
        X_denominator = events_denominator.kinematics[features].to_numpy()
        self.X = np.concatenate([X_numerator, X_denominator])

        # balanced weights
        w_numerator = events_numerator.weights.to_numpy()
        w_denominator = events_denominator.weights.to_numpy()
        w_numerator /= w_numerator.sum()
        w_denominator /= w_denominator.sum()
        self.w = np.concatenate([w_numerator, w_denominator])

        # numerator = signal = 1, denominator = background = 0
        self.s = np.concatenate([np.ones_like(w_numerator), np.zeros_like(w_denominator)])

        if scaler is not None:
            self.X = scaler.transform(self.X)
        
        self.X, self.s, self.w = shuffle(self.X, self.s, self.w, random_state=random_state)
    
    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32), torch.tensor(self.w[index], dtype=torch.float32)
