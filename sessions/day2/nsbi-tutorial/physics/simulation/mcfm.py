import pandas as pd
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .msq import Component

csv_kinematics = [
	'p1_px','p1_py','p1_pz','p1_E',
	'p2_px','p2_py','p2_pz','p2_E',
	'p3_px','p3_py','p3_pz','p3_E',
	'p4_px','p4_py','p4_pz','p4_E',
	'p5_px','p5_py','p5_pz','p5_E',
	'p6_px','p6_py','p6_pz','p6_E',
]

csv_weight = 'wt'

c6_points = [-20, -10, 0, 10, 20]
ct_values = [-1, 0, 1]
cg_values = [-1, 0, 1]
bsm_points = list(product(c6_points, ct_values, cg_values))
n_bsm_points = len(c6_points) * len(ct_values) * len(cg_values)

csv_components = [
	f"msq_{comp}_sm" for comp in ["sig", "int", "sbi", "bkg"]
] + [
	f"msq_{comp}_bsm_{i}"
	for comp in ["sig", "int", "sbi"]
	for i in range(1, n_bsm_points+1)
]

csv_component_sm = {
	comp: f"msq_{comp.value}_sm" for comp in Component
}

csv_component_c6 = {comp: {} for comp in Component}
csv_component_bsm = {comp: {} for comp in Component}
for idx, (cH, ct, cg) in enumerate(bsm_points, start=1):
	for comp in [Component.SBI, Component.INT, Component.SIG]:
		csv_component_bsm[comp][(cH, ct, cg)] = f"msq_{comp}_bsm_{idx}"
		if ct == 0.0 and cg == 0.0:
			csv_component_c6[comp][cH] = f"msq_{comp}_bsm_{idx}"
	csv_component_bsm[Component.BKG][(cH, ct, cg)] = "msq_bkg_sm"
	if ct == 0.0 and cg == 0.0:
		csv_component_c6[Component.BKG][cH] = "msq_bkg_sm"

def stack( *events ):
	"""
	Form a single Process object from multiple Process objects by concatenating their kinematics, components, and weights.

	Parameters
	----------
	*events : Process
			One or more Process objects to be stacked together.
			
	Returns
	-------
	Process
			A new Process object containing the concatenated kinematics, components, and weights.
	"""
	kinematics = pd.concat([e.kinematics for e in events], ignore_index=True)
	components = pd.concat([e.components for e in events], ignore_index=True)
	weights = pd.concat([e.weights for e in events], ignore_index=True)

	return Process(kinematics, components, weights)

def from_csv(file_path : str, *, cross_section : float = None, n_rows : int = None, kinematics : list = None, ignore_negative_weights : bool = True):
	"""
	Open an MCFM CSV file containing a physics process.

	Parameters
	----------
	file_path : str
			Path to the CSV file.

	cross_section : float, optional
			The cross-section of the process in femtobarns (fb). Event weights will be normalized
			so that their sum equals this value. If None, no normalization will be performed.

	n_rows : int or None, optional
			Number of rows to read from the CSV file. If None, all rows are read.
	"""

	import glob
	file_paths = glob.glob(file_path)
	dfs = [pd.read_csv(fp, nrows=n_rows) for fp in file_paths]
	df = pd.concat(dfs, ignore_index=True)

	if kinematics is not None:
		kinematics = df[csv_kinematics + kinematics]
	else:
		kinematics = df[csv_kinematics]
	components = df[csv_components]
	weights = df[csv_weight]

	# HACK: to avoid negative weights
	# only e.g. O(1)/O(1M) events have infinitesimally-small negative weights due to numerical precision
	# you do NOT want to do this for interference-only samples (or NLO samples in the future...)
	if ignore_negative_weights:
		weights = weights.copy()
		weights[weights < 0] = 0.0

	if cross_section is not None:
		weights *= cross_section / weights.sum() 

	return Process(kinematics, components, weights)

def check_consistency(events):
	g1_px, g1_py = events.kinematics['p1_px'], events.kinematics['p1_py']
	g2_px, g2_py = events.kinematics['p2_px'], events.kinematics['p2_py']

	msq_sbi_sm = events.components[csv_component_sm[Component.SBI]].to_numpy()
	msq_sig_sm = events.components[csv_component_sm[Component.SIG]].to_numpy()
	msq_int_sm = events.components[csv_component_sm[Component.INT]].to_numpy()

	msq_sbi_bsm0 = events.components[csv_component_bsm[Component.SBI][(0.0,0.0,0.0)]].to_numpy()
	msq_sig_bsm0 = events.components[csv_component_bsm[Component.SIG][(0.0,0.0,0.0)]].to_numpy()
	msq_int_bsm0 = events.components[csv_component_bsm[Component.INT][(0.0,0.0,0.0)]].to_numpy()

	assert np.allclose(msq_sbi_sm, msq_sbi_bsm0), "msq_sbi_sm and msq_sbi_bsm at (0,0,0) mis-match"
	assert np.allclose(msq_sig_sm, msq_sig_bsm0), "msq_sig_sm and msq_sig_bsm at (0,0,0) mis-match"
	assert np.allclose(msq_int_sm, msq_int_bsm0), "msq_int_sm and msq_int_bsm at (0,0,0) mis-match"

	return True

def to_csv(self, file_path, **kwargs):
	df = pd.concat([self.kinematics, self.components, self.weights.to_frame(name=csv_weight)], axis=1)
	df.to_csv(file_path, index=False)

class Process():

	def __init__(self, kinematics=None, components=None, weights=None):
		self.kinematics = kinematics
		self.components = components
		self.weights = weights

		self.probabilities = weights / weights.sum()

	def calculate(self, calculator):
		new_kinematics = self.kinematics.copy()

		new_columns = calculator(new_kinematics)
		for column_name, column_series in new_columns.items():
			# IMPORTANT: to_numpy() ignores pandas indexing, since DataFrame and Series might mis-match
			new_kinematics.loc[:, column_name] = column_series

		return Process(
			kinematics=new_kinematics.reset_index(drop=True),
			components=self.components.reset_index(drop=True),
			weights=self.weights.reset_index(drop=True)
		)

	def filter(self, filter):
		accepted_indices = filter(self.kinematics, self.components, self.weights, self.probabilities)
		
		return Process(
			self.kinematics.iloc[accepted_indices].reset_index(drop=True),
			self.components.iloc[accepted_indices].reset_index(drop=True),
			self.weights.iloc[accepted_indices].reset_index(drop=True)
		)

	def shuffle(self, random_state=None):
		shuffled_kinematics, shuffled_components, shuffled_weights = shuffle(self.kinematics, self.components, self.weights, random_state=random_state)
		return Process(
			shuffled_kinematics.reset_index(drop=True), 
			shuffled_components.reset_index(drop=True), 
			shuffled_weights.reset_index(drop=True)
		)
	
	def split(self, train_size=1, val_size=1, test_size=None):

		if test_size is not None:
			total_size = train_size + val_size + test_size
				
			kinematics_train, kinematics_val_test, components_train, components_val_test, weights_train, weights_val_test = train_test_split(self.kinematics, self.components, self.weights, train_size=train_size/total_size, test_size=(val_size+test_size)/total_size, shuffle=False)
			kinematics_val, kinematics_test, components_val, components_test, weights_val, weights_test = train_test_split(kinematics_val_test, components_val_test, weights_val_test, train_size=val_size/(val_size+test_size), test_size=test_size/(val_size+test_size), shuffle=False)

			# the weights now must be scaled up so the sum of weights remains the cross-section
			total_wsum = self.weights.sum()
			weights_train *= total_wsum / weights_train.sum()
			weights_val *= total_wsum / weights_val.sum()
			weights_test *= total_wsum / weights_test.sum()

			return Process(
				kinematics_train.reset_index(drop=True), components_train.reset_index(drop=True), weights_train.reset_index(drop=True)
			), Process(
				kinematics_val.reset_index(drop=True), components_val.reset_index(drop=True), weights_val.reset_index(drop=True)
			), Process(
				kinematics_test.reset_index(drop=True), components_test.reset_index(drop=True), weights_test.reset_index(drop=True)
			)

		else:
			total_size = train_size + val_size
			train_size /= total_size
			val_size /= total_size

			kinematics_train, kinematics_val, components_train, components_val, weights_train, weights_val = train_test_split(self.kinematics, self.components, self.weights, test_size=val_size, train_size=train_size, shuffle=False)
			
			# the weights now must be scaled up so the sum of weights remains the cross-section
			total_wsum = self.weights.sum()
			weights_train *= total_wsum / weights_train.sum()
			weights_val *= total_wsum / weights_val.sum()
			
			return Process(
				kinematics_train.reset_index(drop=True), components_train.reset_index(drop=True), weights_train.reset_index(drop=True)
			), Process(
				kinematics_val.reset_index(drop=True), components_val.reset_index(drop=True), weights_val.reset_index(drop=True)
			)

	def sample(self, n, random_state=None):
		"""
  		Select a random portion of the events which (hopefully) corresponds to a lower-statistics sampling of the same hypothesis.
    	"""

		# if sampling more events than available, simply take all shuffled events
		if n >= len(self.weights):
			return self.shuffle(random_state=random_state)

		sampled_events_indices = self.weights.sample(n, replace=False, weights=None, random_state=random_state).index

		# the weights now must be scaled up so the sum of weights remains the cross-section
		sampled_weights = self.weights.loc[sampled_events_indices]
		sampled_weights *= self.weights.sum() / sampled_weights.sum()

		return Process(
			self.kinematics.loc[sampled_events_indices].reset_index(drop=True),
			self.components.loc[sampled_events_indices].reset_index(drop=True),
			sampled_weights.reset_index(drop=True)
		)

	def resample(self, random_state = None):
		"""
  		For wifi ensembles.
		"""
		# TODO: IMPLEMENT ME
		# for now, just return itself
		return self

	def unweight(self, n, random_state=None):
		"""
  		Draw an integer-valued number of occurences of the events by resampling with replacement.
    	"""
		unweighted_events_indices = self.weights.sample(n=n, replace=True, weights=self.weights, random_state=random_state).index

		return Process(
			self.kinematics.loc[unweighted_events_indices].reset_index(drop=True),
			self.components.loc[unweighted_events_indices].reset_index(drop=True),
			pd.Series(np.ones_like(unweighted_events_indices) * self.weights.sum() / n).reset_index(drop=True)
		)

	def reweight(self, denominator, numerator):
		reweights = self.weights * self.components[csv_component_sm[numerator]] / self.components[csv_component_sm[denominator]]
		return Process(
			self.kinematics.reset_index(drop=True), 
			self.components.reset_index(drop=True), 
			reweights.reset_index(drop=True)
		)
	
	def __getitem__(self, item):
		return Process(
			self.kinematics.iloc[item],
			self.components.iloc[item],
			self.weights.iloc[item]
		)