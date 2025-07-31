import torch
import numpy as np
from numpy.polynomial import polynomial as P

from enum import Enum

from ..simulation import mcfm, msq

def msq_eft_over_sm(eft_coefficients, c6=None, ct=None, cg=None):
	eft_coefficients = torch.tensor(eft_coefficients)

	c6_degree = eft_coefficients.shape[1] - 1
	ct_degree = eft_coefficients.shape[2] - 1
	cg_degree = eft_coefficients.shape[3] - 1

	# Helper to wrap input as tensor of shape (n,) or (1,) if scalar
	def to_tensor(val, dtype):
		if val is None:
			return torch.tensor([0.0], dtype=dtype), True
		elif np.isscalar(val):
			return torch.tensor([val], dtype=dtype), np.isscalar(val)
		else:
			return torch.tensor(val, dtype=dtype), np.isscalar(val)

	c6_values, c6_isscalar = to_tensor(c6, eft_coefficients.dtype)
	ct_values, ct_isscalar = to_tensor(ct, eft_coefficients.dtype)
	cg_values, cg_isscalar = to_tensor(cg, eft_coefficients.dtype)

	# Compute power basis
	c6_powers = torch.stack([c6_values.pow(i) for i in range(c6_degree + 1)], dim=0)  # (i, Nc6)
	ct_powers = torch.stack([ct_values.pow(j) for j in range(ct_degree + 1)], dim=0)  # (j, Nct)
	cg_powers = torch.stack([cg_values.pow(k) for k in range(cg_degree + 1)], dim=0)  # (k, Ncg)

	msq_eft_over_sm = torch.einsum('nijk,ix,jy,kz->nxyz', eft_coefficients, c6_powers, ct_powers, cg_powers)

	# Slice down to scalar if input was scalar
	c6_slice = 0 if c6_isscalar else slice(None)
	ct_slice = 0 if ct_isscalar else slice(None)
	cg_slice = 0 if cg_isscalar else slice(None)

	return msq_eft_over_sm[:, c6_slice, ct_slice, cg_slice].numpy()

class Modifier():

	def __init__(self, *, events= None, baseline = msq.Component.SBI, c6_points = [-20,-10,0,10,20], ct_values = [-1,0,1], cg_values = [-1,0,1]):
		self.baseline = baseline
		self.events = events

		self.c6_points = np.array(c6_points)
		self.ct_values = np.array(ct_values)
		self.cg_values = np.array(cg_values)
		self.c6_degree = len(c6_points) - 1
		self.ct_degree = len(ct_values) - 1
		self.cg_degree = len(cg_values) - 1

		X, Y, Z = np.meshgrid(self.c6_points, self.ct_values, self.cg_values, indexing='ij')  # Shape: (5, 3, 3) 
		V = P.polyvander3d(X, Y, Z, [self.c6_degree, self.ct_degree, self.cg_degree])

		msq_sm  = self.events.components[mcfm.csv_component_sm[self.baseline]].to_numpy()
		xyz_bsm = []
		for i, c6_val in enumerate(self.c6_points):
			for j, ct_val in enumerate(self.ct_values):
				for k, cg_val in enumerate(self.cg_values):
					xyz_bsm.append((c6_val, ct_val, cg_val))
		xyz_npts = len(xyz_bsm)
		msq_bsm = self.events.components[[mcfm.csv_component_bsm[self.baseline][xyz] for xyz in xyz_bsm]].to_numpy()

		bsm_values = msq_bsm / msq_sm[:, np.newaxis]

		V = V.reshape(xyz_npts, xyz_npts)
		bsm_values = bsm_values.reshape(-1, xyz_npts).T
		coefficients = np.linalg.solve(V, bsm_values)

		self.coefficients = coefficients.T.reshape(-1, self.c6_degree+1, self.ct_degree+1, self.cg_degree+1)

		# filter out non-physical coefficients
		# self.coefficients[:, 3, 2:, :] = 0
		# self.coefficients[:, 3, :, 2:] = 0
		# self.coefficients[:, 4, 1:, :] = 0
		# self.coefficients[:, 4, :, 1:] = 0

	def modify(self, c6 = None, ct = None, cg = None):
		
		# Create a tuple with `count_not_none` times np.newaxis
		rwt = msq_eft_over_sm(self.coefficients, c6, ct, cg)
		newaxes = (None,) * (rwt.ndim-1)
		w_eft = rwt * self.events.weights.to_numpy()[(slice(None), )+newaxes]
		p_eft = w_eft / np.sum(w_eft, axis=0, keepdims=True)

		return w_eft, p_eft