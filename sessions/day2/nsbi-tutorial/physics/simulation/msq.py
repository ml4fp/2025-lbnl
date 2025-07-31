from enum import Enum
import numpy as np

class Component(str, Enum):
  SBI = "sbi"
  SIG = "sig"
  INT = "int"
  BKG = "bkg"

class MSQFilter():
  def __init__(self, component, value):
    self.component = component
    self.value = value

  def __call__(self, kinematics, components, weights, probabilities):
    indices = np.where(np.array(components[self.component])!=self.value)[0]
    return indices