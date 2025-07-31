import os
import json
import argparse
import numpy as np
import pandas as pd
from nsbi import carl

from physics.simulation import mcfm
from physics.hstar import sigstr

def main(args):

    (events_sbi_test, _), _, _, _ = carl.utils.load_results('run/h4l/', 'sbi_over_bkg')

    kinematics = events_sbi_test.kinematics[args.kinematics]

    nu_obs, _ = sigstr.scale(events_sbi_test, signal_strength = float(args.signal_strength))
    nu_obs *= args.lumi

    df = pd.concat([kinematics, pd.Series(nu_obs).to_frame(name='n')], axis=1)

    df.to_csv(args.bsm_output, index=False)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge MCFM event CSVs from multiple processes.")

    parser.add_argument('--lumi', required=False, type=float, default=3000., help='Luminosity')
    parser.add_argument('--kinematics', required=False, default=['l1_pt', 'l1_eta', 'l1_phi', 'l1_energy', 'l2_pt', 'l2_eta', 'l2_phi', 'l2_energy', 'l3_pt', 'l3_eta', 'l3_phi', 'l3_energy', 'l4_pt', 'l4_eta', 'l4_phi', 'l4_energy'], help='Input events')
    parser.add_argument('--signal-strength', required=False, default=1.0, help='Signal strength')
    parser.add_argument('--bsm-output', required=False, default='observed.csv', help='Input events')

    args = parser.parse_args()

    main(args)
