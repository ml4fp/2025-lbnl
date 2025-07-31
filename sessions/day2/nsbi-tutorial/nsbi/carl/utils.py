import os
import re
import pickle

from models.carl import CARL

import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as L

def load_results(run_dir, run_name = 'carl'):

    carl_dir = os.path.join(run_dir, run_name)
    logs_dir = os.path.join(carl_dir, 'lightning_logs')

    with open(os.path.join(carl_dir, 'events_numerator_val.pkl'), 'rb') as f:
        events_num_val = pickle.load(f)
    with open(os.path.join(carl_dir, 'events_denominator_val.pkl'), 'rb') as f:
        events_denom_val = pickle.load(f)
    with open(os.path.join(carl_dir, 'events_numerator_test.pkl'), 'rb') as f:
        events_num_test = pickle.load(f)
    with open(os.path.join(carl_dir, 'events_denominator_test.pkl'), 'rb') as f:
        events_denom_test = pickle.load(f)
    with open(os.path.join(carl_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Find the latest version folder
    versions = [d for d in os.listdir(logs_dir) if re.match(r'version_\d+', d)]
    if not versions:
        raise FileNotFoundError("No version folders found in lightning_logs.")

    # Extract version numbers and sort
    latest_version = max(versions, key=lambda v: int(re.search(r'\d+', v).group()))
    checkpoint_dir = os.path.join(logs_dir, latest_version, 'checkpoints')

    # Find all checkpoint files matching the pattern
    checkpoints = [f for f in os.listdir(checkpoint_dir) if re.match(r'epoch=\d+-val_loss=.+\.ckpt', f)]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Get the checkpoint with the largest epoch number
    ckpt_path = max(checkpoints, key=lambda f: int(re.search(r'epoch=(\d+)', f).group(1)))

    ckpt = CARL.load_from_checkpoint(checkpoint_path=os.path.join(checkpoint_dir, ckpt_path))

    return (events_num_val, events_num_test), (events_denom_val, events_denom_test), scaler, ckpt

def get_likelihood_ratio(events, features, scaler_X, model):
    trainer = L.Trainer(accelerator='gpu', devices=1, enable_progress_bar=False)

    kinematics = events.kinematics[features]
    X_carl = scaler_X.transform(kinematics.to_numpy())
    dl_carl = DataLoader(TensorDataset(torch.tensor(X_carl, dtype=torch.float32)), batch_size=128, num_workers=1) 
    s_carl = torch.cat(trainer.predict(model, dl_carl))

    return s_carl/(1-s_carl)