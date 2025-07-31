import pandas as pd
import numpy as np
import torch
from numpy.polynomial import polynomial as P

from enum import Enum
from typing import Optional, Union

from ..simulation import mcfm, msq

def scale(events : mcfm.Process = None, component : msq.Component = msq.Component.SBI, signal_strength : Union[float, torch.Tensor] = 1.0):
    w_sm = events.weights.to_numpy()

    mu_is_scalar = np.isscalar(signal_strength)
    if mu_is_scalar:
        signal_strength = torch.tensor([signal_strength])
        mu_is_scalar = True

    msq_sig_scaled = torch.tensor(events.components[mcfm.csv_component_sm[msq.Component.SIG]].to_numpy())[:,None] * signal_strength[None,:]
    msq_int_scaled = torch.tensor(events.components[mcfm.csv_component_sm[msq.Component.INT]].to_numpy())[:,None] * torch.sqrt(signal_strength)[None,:]
    msq_bkg_sm     = torch.tensor(events.components[mcfm.csv_component_sm[msq.Component.BKG]].to_numpy())[:,None]
    msq_sbi_sm     = torch.tensor(events.components[mcfm.csv_component_sm[msq.Component.SBI]].to_numpy())[:,None]

    msq_scaling = (msq_sig_scaled + msq_int_scaled + msq_bkg_sm) / msq_sbi_sm

    w_scaled = msq_scaling * w_sm[:, None]
    p_scaled = w_scaled / torch.sum(w_scaled, dim=0)

    if mu_is_scalar:
        w_scaled = w_scaled.flatten()
        p_scaled = p_scaled.flatten()

    return w_scaled, p_scaled

    # return mcfm.Process(
    #     kinematics=events.kinematics.reset_index(drop=True),
    #     components=events.components.reset_index(drop=True),
    #     weights=pd.Series(w_scaled).reset_index(drop=True)
    # )