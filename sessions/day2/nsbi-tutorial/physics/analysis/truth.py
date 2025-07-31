import vector
from vector import MomentumObject4D

import pandas as pd

class TruthZZ():
    def __init__(self):
        pass

    def __call__(self, kinematics):
        l1 = vector.array({'px': kinematics['p3_px'], 'py': kinematics['p3_py'], 'pz': kinematics['p3_pz'], 'E': kinematics['p3_E']})#negative l1
        l2 = vector.array({'px': kinematics['p4_px'], 'py': kinematics['p4_py'], 'pz': kinematics['p4_pz'], 'E': kinematics['p4_E']})#positive l1
        l3 = vector.array({'px': kinematics['p5_px'], 'py': kinematics['p5_py'], 'pz': kinematics['p5_pz'], 'E': kinematics['p5_E']})#negative l2
        l4 = vector.array({'px': kinematics['p6_px'], 'py': kinematics['p6_py'], 'pz': kinematics['p6_pz'], 'E': kinematics['p6_E']})#positive l2

        return {'l1_px': kinematics['p3_px'].to_numpy(), 'l1_py': kinematics['p3_py'].to_numpy(), 'l1_pz': kinematics['p3_pz'].to_numpy(), 'l1_E': kinematics['p3_E'].to_numpy(),
                'l2_px': kinematics['p4_px'].to_numpy(), 'l2_py': kinematics['p4_py'].to_numpy(), 'l2_pz': kinematics['p4_pz'].to_numpy(), 'l2_E': kinematics['p4_E'].to_numpy(),
                'l3_px': kinematics['p5_px'].to_numpy(), 'l3_py': kinematics['p5_py'].to_numpy(), 'l3_pz': kinematics['p5_pz'].to_numpy(), 'l3_E': kinematics['p5_E'].to_numpy(),
                'l4_px': kinematics['p6_px'].to_numpy(), 'l4_py': kinematics['p6_py'].to_numpy(), 'l4_pz': kinematics['p6_pz'].to_numpy(), 'l4_E': kinematics['p6_E'].to_numpy(),
                'Z1_mass': (l1+l2).mass, 'Z2_mass': (l3+l4).mass,
                'Z1_pt': (l1 + l2).pt, 'Z2_pt': (l3 + l4).pt}

class MandelstamVariables():
    def __init__(self):
        """
        Calculator class that calculates the Mandelstam variables t,u and s=m4l
        """
        self.variable_functions = {'mandelstam_s': self.calc_s, 'mandelstam_t': self.calc_t, 'mandelstam_u': self.calc_u}

    def __call__(self, kinematics):
        g1 = vector.array({'px': kinematics['p1_px'], 'py': kinematics['p1_py'], 'pz': kinematics['p1_pz'], 'E': kinematics['p1_E']})
        g2 = vector.array({'px': kinematics['p2_px'], 'py': kinematics['p2_py'], 'pz': kinematics['p2_pz'], 'E': kinematics['p2_E']})
        l1 = vector.array({'px': kinematics['l1_px'], 'py': kinematics['l1_py'], 'pz': kinematics['l1_pz'], 'E': kinematics['l1_E']})
        l2 = vector.array({'px': kinematics['l2_px'], 'py': kinematics['l2_py'], 'pz': kinematics['l2_pz'], 'E': kinematics['l2_E']})
        l3 = vector.array({'px': kinematics['l3_px'], 'py': kinematics['l3_py'], 'pz': kinematics['l3_pz'], 'E': kinematics['l3_E']})
        l4 = vector.array({'px': kinematics['l4_px'], 'py': kinematics['l4_py'], 'pz': kinematics['l4_pz'], 'E': kinematics['l4_E']})
        
        Z1 = l1 + l2
        Z2 = l3 + l4

        results = {}
        for variable in self.variable_functions.keys():
            results[variable] = self.variable_functions[variable](g1, g2, Z1, Z2)

        return results

    def calc_s(self, *particles: MomentumObject4D):
        return (particles[2]+particles[3]).mass2

    def calc_t(self, *particles: MomentumObject4D):
        return (particles[0]-particles[2]).mass2

    def calc_u(self, *particles: MomentumObject4D):
        return (particles[0]-particles[3]).mass2