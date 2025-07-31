import numpy as np
import vector
from vector import MomentumObject4D

import pandas as pd

class ZPairCandidate:
    def __init__(self, algorithm: str = 'leastsquare'):
        if algorithm not in ['leastsquare', 'closest']:
            raise ValueError('algorithm has to be one of ["leastsquare", "closest"]')

        self.algorithm = algorithm
        self.Z_mass = 91.18
    
    def __call__(self, kinematics):
        l1 = vector.array({'px': kinematics['p3_px'], 'py': kinematics['p3_py'], 'pz': kinematics['p3_pz'], 'E': kinematics['p3_E']}) #negative l1
        l2 = vector.array({'px': kinematics['p4_px'], 'py': kinematics['p4_py'], 'pz': kinematics['p4_pz'], 'E': kinematics['p4_E']}) #positive l1
        l3 = vector.array({'px': kinematics['p5_px'], 'py': kinematics['p5_py'], 'pz': kinematics['p5_pz'], 'E': kinematics['p5_E']}) #negative l2
        l4 = vector.array({'px': kinematics['p6_px'], 'py': kinematics['p6_py'], 'pz': kinematics['p6_pz'], 'E': kinematics['p6_E']}) #positive l2

        if self.algorithm == 'leastsquare':
            return self.find_Z_lsq(l1, l2, l3, l4)
        elif self.algorithm == 'closest':
            return self.find_Z_closest(l1, l2, l3, l4)

    def find_Z_lsq(self, l1, l2, l3, l4):
        # Possible Z bosons from leptons 
        p12 = (l1 + l2)
        p14 = (l1 + l4)
        p23 = (l2 + l3)
        p34 = (l3 + l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',np.float64),('py',np.float64),('pz',np.float64),('E',np.float64)])
        lepton_pairs = vector.array([[[l1,l2],[l3,l4]],
                                      [[l1,l4],[l3,l2]]], dtype=[('px',np.float64),('py',np.float64),('pz',np.float64),('E',np.float64)])

        # Squared minimization to determine the closest pair
        sq = np.array([(pair[0].mass - self.Z_mass)**2 + (pair[1].mass - self.Z_mass)**2 for pair in pairs]).T
        closest_pair_indices = np.argmin(sq, axis=1)
        closest_pair = pairs.transpose(2,0,1)[np.arange(len(closest_pair_indices)), closest_pair_indices].T

        # Determine the Z boson with the higher pT
        # That one will be Z1, the other one Z2
        pT_max_ind = np.argmax(closest_pair.mass,axis=0) # Z1
        pT_min_ind = np.argmin(closest_pair.mass,axis=0) # Z2

        # Determine the order manually if both Z bosons have the same pT
        cond=(pT_max_ind==pT_min_ind)

        pT_max_ind[np.where(cond)] = 0
        pT_min_ind[np.where(cond)] = 1

        # (l1_1, l2_1) = Z1; (l1_2, l2_2) = Z2
        l1_1, l2_1 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_max_ind)), pT_max_ind].T
        l1_2, l2_2 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_min_ind)), pT_min_ind].T

        return {
            'Z1_l1_px': l1_1.px, 'Z1_l1_py': l1_1.py, 'Z1_l1_pz': l1_1.pz, 'Z1_l1_E': l1_1.E,
            'Z1_l2_px': l2_1.px, 'Z1_l2_py': l2_1.py, 'Z1_l2_pz': l2_1.pz, 'Z1_l2_E': l2_1.E,
            'Z2_l1_px': l1_2.px, 'Z2_l1_py': l1_2.py, 'Z2_l1_pz': l1_2.pz, 'Z2_l1_E': l1_2.E,
            'Z2_l2_px': l2_2.px, 'Z2_l2_py': l2_2.py, 'Z2_l2_pz': l2_2.pz, 'Z2_l2_E': l2_2.E,
            'Z1_mass': (l1_1 + l2_1).mass, 'Z2_mass': (l1_2 + l2_2).mass,
            'Z1_pt': (l1_1 + l2_1).pt, 'Z2_pt': (l1_2 + l2_2).pt
            }
    
    def find_Z_closest(self, l1, l2, l3, l4):
        # Possible Z bosons from leptons 
        p12 = (l1 + l2)
        p14 = (l1 + l4)
        p23 = (l2 + l3)
        p34 = (l3 + l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',float),('py',float),('pz',float),('E',float)])
        lepton_pairs = vector.array([[[l1,l2],[l3,l4]],
                                      [[l1,l4],[l3,l2]]], dtype=[('px',float),('py',float),('pz',float),('E',float)])

        # Just choose the Z boson pair which contains the Z boson closest to the true rest mass
        pairs_diffs = ((pairs.mass - np.ones(pairs.shape)*self.Z_mass)**2).transpose(2,0,1).reshape(pairs.shape[2],4)
        min_ind = np.floor(np.argmin(pairs_diffs, axis=1)/2.0).astype(int)
        closest_Z_pair = pairs.transpose(2,0,1)[np.arange(len(min_ind)),min_ind].T

        closest_Z_min_ind = np.argmin((closest_Z_pair.mass-self.Z_mass)**2, axis=0)
        closest_Z_max_ind = np.argmax((closest_Z_pair.mass-self.Z_mass)**2, axis=0)

        # (l1_1, l2_1) = Z1; (l1_2, l2_2) = Z2
        l1_1, l2_1 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_min_ind)), closest_Z_min_ind].T
        l1_2, l2_2 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_max_ind)), closest_Z_max_ind].T

        return {
            'Z1_l1_px': l1_1.px, 'Z1_l1_py': l1_1.py, 'Z1_l1_pz': l1_1.pz, 'Z1_l1_E': l1_1.E,
            'Z1_l2_px': l2_1.px, 'Z1_l2_py': l2_1.py, 'Z1_l2_pz': l2_1.pz, 'Z1_l2_E': l2_1.E,
            'Z2_l1_px': l1_2.px, 'Z2_l1_py': l1_2.py, 'Z2_l1_pz': l1_2.pz, 'Z2_l1_E': l1_2.E,
            'Z2_l2_px': l2_2.px, 'Z2_l2_py': l2_2.py, 'Z2_l2_pz': l2_2.pz, 'Z2_l2_E': l2_2.E,
            'Z1_mass': (l1_1 + l2_1).mass, 'Z2_mass': (l1_2 + l2_2).mass,
            'Z1_pt': (l1_1 + l2_1).pt, 'Z2_pt': (l1_2 + l2_2).pt
            }

class ZPairMassWindow():
    def __init__(self, z1: tuple[int, int] = None, z2: tuple[int, int] = None):
        self.z1 = z1
        self.z2 = z2

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        #Outgoing leptons
        Z1_mass = kinematics['Z1_mass']
        Z2_mass = kinematics['Z2_mass']

        if self.z1 is not None:
            cond1 = np.where((Z1_mass>=self.z1[0])&(Z1_mass<=self.z1[1]))
        else:
            cond1 = np.arange(Z1_mass.shape[0])

        if self.z2 is not None:
            cond2 = np.where((Z2_mass>=self.z2[0])&(Z2_mass<=self.z2[1]))
        else:
            cond2 = np.arange(Z2_mass.shape[0])

        # Get only indices where cond1 and cond2 apply
        indices = np.intersect1d(cond1,cond2)

        return indices

class AngularVariables():
    def __init__(self):
        """
        Calculator class that calculates the kinematics needed for constructing datasets.
        Angles cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙 used in this class are described in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.095031.
        """
        self.variable_functions = {'cth_star': self.calc_cth_star, 'cth_1': self.calc_cth_1, 'cth_2': self.calc_cth_2, 'phi_1': self.calc_phi_1, 
                                   'phi': self.calc_phi, 'mZ1': self.calc_mZ1, 'mZ2': self.calc_mZ2}
    
    def __call__(self, kinematics):
        l1 = vector.array({'px': kinematics['l1_px'], 'py': kinematics['l1_py'], 'pz': kinematics['l1_pz'], 'E': kinematics['l1_energy']})
        l2 = vector.array({'px': kinematics['l2_px'], 'py': kinematics['l2_py'], 'pz': kinematics['l2_pz'], 'E': kinematics['l2_energy']})
        l3 = vector.array({'px': kinematics['l3_px'], 'py': kinematics['l3_py'], 'pz': kinematics['l3_pz'], 'E': kinematics['l3_energy']})
        l4 = vector.array({'px': kinematics['l4_px'], 'py': kinematics['l4_py'], 'pz': kinematics['l4_pz'], 'E': kinematics['l4_energy']})
        
        results = {}
        for variable in self.variable_functions.keys():
            results[variable] = self.variable_functions[variable](l1, l2, l3, l4)

        return results
    
    def calc_cth_star(self, *leptons: MomentumObject4D):
        Z1 = leptons[0] + leptons[1]
        H = Z1 + leptons[2] + leptons[3]
        return Z1.boost(-H).to_3D().unit().z

    def calc_cth_1(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        Z1_h = Z1.boost(-H)
        Z2_h = Z2.boost(-H)

        z2_in_Z1 = Z2_h.boost(-Z1_h).to_3D()
        l1 = leptons[0].boost(-Z1_h)
        return -z2_in_Z1.dot(l1.to_3D())/np.abs(z2_in_Z1.mag*l1.to_3D().mag)

    def calc_cth_2(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        Z1_h = Z1.boost(-H)
        Z2_h = Z2.boost(-H)

        z1_in_Z2 = Z1_h.boost(-Z2_h).to_3D()
        l3 = leptons[2].boost(-Z2_h)
        return -z1_in_Z2.dot(l3.to_3D())/np.abs(z1_in_Z2.mag*l3.to_3D().mag)

    def calc_phi_1(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        Z1_h = Z1.boost(-H)
        z1 = Z1_h.to_3D().unit()

        l1_h = leptons[0].boost(-H).to_3D()
        l2_h = leptons[1].boost(-H).to_3D()

        nz = vector.array({'x': np.zeros(Z1_h.shape[0]), 'y': np.zeros(Z1_h.shape[0]), 'z': np.ones(Z1_h.shape[0])})

        n12 = l1_h.cross(l2_h).unit() # Normal vector of the plane in which the Z1 decay takes place
        nscp = nz.cross(z1).unit() # Normal vector of the plane in which the H -> Z1, Z2 takes place
            
        return z1.dot(n12.cross(nscp))/np.abs(z1.dot(n12.cross(nscp)))*np.arccos(n12.dot(nscp))
    
    def calc_phi(self, *leptons: MomentumObject4D):
        Z1 = leptons[0]+leptons[1]
        Z2 = leptons[2]+leptons[3]
        H = Z1+Z2

        z1 = Z1.boost(-H).to_3D().unit()

        l1_h = leptons[0].boost(-H).to_3D()
        l2_h = leptons[1].boost(-H).to_3D()
        l3_h = leptons[2].boost(-H).to_3D()
        l4_h = leptons[3].boost(-H).to_3D()

        n12 = l1_h.cross(l2_h).unit() # Normal vector of the plane in which the Z1 decay takes place
        n34 = l3_h.cross(l4_h).unit() # Normal vector of the plane in which the Z2 decay takes place

        return z1.dot(n12.cross(n34))/np.abs(z1.dot(n12.cross(n34)))*np.arccos(-n12.dot(n34))

    def calc_mZ1(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]).mass

    def calc_mZ2(self, *leptons: MomentumObject4D):
        return (leptons[2]+leptons[3]).mass

class LeptonMomenta():
    def __call__(self, kinematics):
        l1 = vector.array({'px': kinematics['p3_px'], 'py': kinematics['p3_py'], 'pz': kinematics['p3_pz'], 'E': kinematics['p3_E']})
        l2 = vector.array({'px': kinematics['p4_px'], 'py': kinematics['p4_py'], 'pz': kinematics['p4_pz'], 'E': kinematics['p4_E']})
        l3 = vector.array({'px': kinematics['p5_px'], 'py': kinematics['p5_py'], 'pz': kinematics['p5_pz'], 'E': kinematics['p5_E']})
        l4 = vector.array({'px': kinematics['p6_px'], 'py': kinematics['p6_py'], 'pz': kinematics['p6_pz'], 'E': kinematics['p6_E']})

        leptons = np.array([l1,l2,l3,l4]).T
        pt = np.array([l1.pt, l2.pt, l3.pt, l4.pt]).T
        charges = np.tile(np.array([-1, +1, -1, +1]), (len(l1), 1))

        # sort by pt
        indices = np.argsort(pt, axis=1)[:,::-1]
        leptons_sorted = vector.array(np.take_along_axis(leptons, indices, axis=1), dtype=[("px", np.float32), ("py", np.float32), ("pz", np.float32), ("E", np.float32)])
        charges_sorted = np.take_along_axis(charges, indices, axis=1)

        return {'l1_pt': leptons_sorted[:,0].pt, 'l1_eta': leptons_sorted[:,0].eta, 'l1_phi': leptons_sorted[:,0].phi, 'l1_energy': leptons_sorted[:,0].energy, 'l1_charge' : charges_sorted[:,0],
                'l2_pt': leptons_sorted[:,1].pt, 'l2_eta': leptons_sorted[:,1].eta, 'l2_phi': leptons_sorted[:,1].phi, 'l2_energy': leptons_sorted[:,1].energy, 'l2_charge' : charges_sorted[:,1],
                'l3_pt': leptons_sorted[:,2].pt, 'l3_eta': leptons_sorted[:,2].eta, 'l3_phi': leptons_sorted[:,2].phi, 'l3_energy': leptons_sorted[:,2].energy, 'l3_charge' : charges_sorted[:,2],
                'l4_pt': leptons_sorted[:,3].pt, 'l4_eta': leptons_sorted[:,3].eta, 'l4_phi': leptons_sorted[:,3].phi, 'l4_energy': leptons_sorted[:,3].energy, 'l4_charge' : charges_sorted[:,3]}

class FourLeptonSystem():
    def __init__(self):
        """
        Calculator class that calculates the kinematics needed for constructing datasets.
        Angles cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙 used in this class are described in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.095031.
        """
        self.variable_functions = {'4l_mass': self.calc_m4l, '4l_rapidity': self.calc_y4l, '4l_pT': self.calc_pT, '4l_energy': self.calc_E}

    def __call__(self, kinematics):
        l1 = vector.array({'pt': kinematics['l1_pt'], 'eta': kinematics['l1_eta'], 'phi': kinematics['l1_phi'], 'energy': kinematics['l1_energy']})
        l2 = vector.array({'pt': kinematics['l2_pt'], 'eta': kinematics['l2_eta'], 'phi': kinematics['l2_phi'], 'energy': kinematics['l2_energy']})
        l3 = vector.array({'pt': kinematics['l3_pt'], 'eta': kinematics['l3_eta'], 'phi': kinematics['l3_phi'], 'energy': kinematics['l3_energy']})
        l4 = vector.array({'pt': kinematics['l4_pt'], 'eta': kinematics['l4_eta'], 'phi': kinematics['l4_phi'], 'energy': kinematics['l4_energy']})
        
        results = {}
        for variable in self.variable_functions.keys():
            results[variable] = self.variable_functions[variable](l1, l2, l3, l4)

        return results

    def calc_m4l(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]+leptons[2]+leptons[3]).mass

    def calc_y4l(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]+leptons[2]+leptons[3]).rapidity
    
    def calc_pT(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]+leptons[2]+leptons[3]).pt
    
    def calc_E(self, *leptons: MomentumObject4D):
        return (leptons[0]+leptons[1]+leptons[2]+leptons[3]).E

class M4lFilter():
    def __init__(self, m4l_min=None, m4l_max=None):
        self.m4l_min = m4l_min
        self.m4l_max = m4l_max

    def __call__(self, kinematics, components, weights, probabilities):
        l1 = vector.array({'pt': kinematics['l1_pt'], 'eta': kinematics['l1_eta'], 'phi': kinematics['l1_phi'], 'energy': kinematics['l1_energy']})
        l2 = vector.array({'pt': kinematics['l2_pt'], 'eta': kinematics['l2_eta'], 'phi': kinematics['l2_phi'], 'energy': kinematics['l2_energy']})
        l3 = vector.array({'pt': kinematics['l3_pt'], 'eta': kinematics['l3_eta'], 'phi': kinematics['l3_phi'], 'energy': kinematics['l3_energy']})
        l4 = vector.array({'pt': kinematics['l4_pt'], 'eta': kinematics['l4_eta'], 'phi': kinematics['l4_phi'], 'energy': kinematics['l4_energy']})

        m4l = (l1+l2+l3+l4).mass

        if self.m4l_min is not None:
            cond1 = np.where(m4l>=self.m4l_min)
        else:
            cond1 = np.arange(m4l.shape[0])

        if self.m4l_max is not None:
            cond2 = np.where(m4l<=self.m4l_max)
        else:
            cond2 = np.arange(m4l.shape[0])

        indices = np.intersect1d(cond1, cond2)

        return indices, None

class LeptonPtEtaCut():
    def __init__(self, lepton_index, *, pt_min = 20, eta_max = 2.5):
        self.lepton_index = lepton_index
        self.pt_min = pt_min
        self.eta_max = eta_max

    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
        l_pt = kinematics[f'l{self.lepton_index}_pt']
        l_eta = np.abs(kinematics[f'l{self.lepton_index}_eta'].to_numpy())

        indices, = np.where((l_pt > self.pt_min) & (l_eta < self.eta_max))
        return indices

def analyze(events):

    # angular_vars = AngularVariables()
    z_cand = ZPairCandidate(algorithm='leastsquare')
    lepton_momenta = LeptonMomenta()
    fourlep = FourLeptonSystem()
    events_analyzed = events.calculate(z_cand).calculate(lepton_momenta).calculate(fourlep)

    print('Inclusive |', events_analyzed.weights.sum())

    z_masses = ZPairMassWindow(z1=(70,110), z2=(70,110))
    events_analyzed = events_analyzed.filter(z_masses).filter(LeptonPtEtaCut(1,pt_min=20,eta_max=2.5)).filter(LeptonPtEtaCut(2,pt_min=15,eta_max=2.5)).filter(LeptonPtEtaCut(3,pt_min=10,eta_max=2.5)).filter(LeptonPtEtaCut(4,pt_min=7,eta_max=2.5))

    print('Analyzed  |', events_analyzed.weights.sum())

    return events_analyzed

features = ["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]