import numpy as np
import vector

import pandas as pd

from ..constants import mZ
from .zz4l import LeptonPtEtaCut

class ZZ2L2V():
    def __init__(self, lepton_indices = [3,4], neutrino_indices = [5,6]):
        """
        """

        self.lepton_indices = lepton_indices
        self.neutrino_indices = neutrino_indices

    def __call__(self, kinematics):
        from ..constants import mZ

        l1 = vector.array({'px': kinematics[f'p{self.lepton_indices[0]}_px'], 'py': kinematics[f'p{self.lepton_indices[0]}_py'], 'pz': kinematics[f'p{self.lepton_indices[0]}_pz'], 'E': kinematics[f'p{self.lepton_indices[0]}_E']})
        l2 = vector.array({'px': kinematics[f'p{self.lepton_indices[1]}_px'], 'py': kinematics[f'p{self.lepton_indices[1]}_py'], 'pz': kinematics[f'p{self.lepton_indices[1]}_pz'], 'E': kinematics[f'p{self.lepton_indices[1]}_E']})
        v1 = vector.array({'px': kinematics[f'p{self.neutrino_indices[0]}_px'], 'py': kinematics[f'p{self.neutrino_indices[0]}_py'], 'pz': kinematics[f'p{self.neutrino_indices[0]}_pz'], 'E': kinematics[f'p{self.neutrino_indices[0]}_E']})
        v2 = vector.array({'px': kinematics[f'p{self.neutrino_indices[1]}_px'], 'py': kinematics[f'p{self.neutrino_indices[1]}_py'], 'pz': kinematics[f'p{self.neutrino_indices[1]}_pz'], 'E': kinematics[f'p{self.neutrino_indices[1]}_E']})

        pt = np.array([l1.pt, l2.pt]).T
        indices = np.argsort(pt, axis=1)[:,::-1]
        leptons = np.array([l1,l2]).T
        leptons_sorted = vector.array(np.take_along_axis(leptons, indices, axis=1), dtype=[("px", np.float32), ("py", np.float32), ("pz", np.float32), ("E", np.float32)])
        
        results = {'l1_pt': leptons_sorted[:,0].pt, 'l1_eta': leptons_sorted[:,0].eta, 'l1_phi': leptons_sorted[:,0].phi, 'l1_energy': leptons_sorted[:,0].energy,
                   'l2_pt': leptons_sorted[:,1].pt, 'l2_eta': leptons_sorted[:,1].eta, 'l2_phi': leptons_sorted[:,1].phi, 'l2_energy': leptons_sorted[:,1].energy}

        ll = l1+l2
        met = (v1+v2).to_2D()

        results['ll_pt']   = ll.pt
        results['ll_eta']  = ll.eta
        results['ll_phi']  = ll.phi
        results['ll_mass'] = ll.mass

        results['met']     = met.pt
        results['met_phi'] = met.phi

        results['zz_mt'] = np.sqrt( (np.sqrt(mZ**2 + ll.pt2) + np.sqrt(mZ**2 + met.pt2))**2 - (ll.to_2D() + met).pt2 )

        results['ll_dr'] = l1.deltaR(l2)
        results['llmet_dphi'] = ll.to_2D().deltaphi(met)

        return results


class MTZZMinCut():
    def __init__(self, mtzz_min = 270):
        self.mtzz_min = mtzz_min

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        mtzz = kinematics['zz_mt']
        indices, = np.where((mtzz>=self.mtzz_min))
        return indices

class ZMassWindow():
    def __init__(self, mz_min = 75, mz_max = 105):
        self.mz_min = mz_min
        self.mz_max = mz_max

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        Z_mass = kinematics['ll_mass']
        indices, = np.where((Z_mass>=self.mz_min)&(Z_mass<=self.mz_max))
        return indices

class MinMETCut():
    def __init__(self, met_min = 100):
        self.met_min = met_min

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        met = kinematics['met']
        indices, = np.where( met > self.met_min )
        return indices

class MinDPhillMETCut():
    def __init__(self, dphi_min = 2.5):
        self.dphi_min = dphi_min

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        dphi = kinematics['llmet_dphi']
        indices, = np.where( np.abs(dphi) > self.dphi_min )
        return indices

class MaxDRllCut():
    def __init__(self, dr_max = 1.8):
        self.dr_max = dr_max

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        dr = kinematics['ll_dr']
        indices, = np.where(dr < self.dr_max)
        return indices

def analyze(events):

    events_analyzed = events.calculate(ZZ2L2V(lepton_indices=[3,4],neutrino_indices=[5,6]))
    print(f'Inclusive            | {events_analyzed.weights.sum()} +/- {np.sqrt(np.sum(np.square(events_analyzed.weights)))}')

    events_analyzed = events_analyzed.filter(LeptonPtEtaCut(1,pt_min=30,eta_max=2.5)).filter(LeptonPtEtaCut(2,pt_min=20,eta_max=2.5))
    print(f'lepton (pT,eta) cuts | {events_analyzed.weights.sum()} +/- {np.sqrt(np.sum(np.square(events_analyzed.weights)))}')

    met_max = 60
    events_analyzed = events_analyzed.filter(MinMETCut(met_max))
    print(f'MET cut              | {events_analyzed.weights.sum()} +/- {np.sqrt(np.sum(np.square(events_analyzed.weights)))}')

    events_analyzed = events_analyzed.filter(MTZZMinCut(250))
    print(f'mTZZ cut             | {events_analyzed.weights.sum()} +/- {np.sqrt(np.sum(np.square(events_analyzed.weights)))}')

    events_analyzed = events_analyzed.filter(ZMassWindow(80,100))
    print(f'mZ window            | {events_analyzed.weights.sum()} +/- {np.sqrt(np.sum(np.square(events_analyzed.weights)))}')

    dphillmet_min = 2.5
    events_analyzed = events_analyzed.filter(MinDPhillMETCut(dphillmet_min))
    print(f'DPhillMET cut        | {events_analyzed.weights.sum()} +/- {np.sqrt(np.sum(np.square(events_analyzed.weights)))}')

    drll_max = 2.0
    events_analyzed = events_analyzed.filter(MaxDRllCut(drll_max))
    print(f'DRll cut             | {events_analyzed.weights.sum()} +/- {np.sqrt(np.sum(np.square(events_analyzed.weights)))}')

    return events_analyzed

features = ["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "met", "met_phi"]