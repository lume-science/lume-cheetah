# lume-cheetah
Cheetah-specific implementation of LUMEModel classes for virtual accelerators

## What this package provides

- `CheetahSimulator`: wraps a Cheetah accelerator `Segment` and initial beam distribution.
- `LUMECheetahModel`: a LUME action model driven by simulator-backed variables.

## Example: implementing a `LUMECheetahModel` object

```python
import torch
from cheetah.accelerator import Drift, Quadrupole, Segment
from cheetah.particles import ParticleBeam

from lume_cheetah import CheetahSimulator, LUMECheetahModel
from lume_cheetah.actions import CheetahWritableScalarVariable


# 1) Build a Cheetah lattice segment
segment = Segment(
	[
		Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
		Drift(name="D1", length=torch.tensor(1.0)),
	]
)

# 2) Create an initial beam distribution
beam = ParticleBeam.from_twiss(
	beta_x=torch.tensor(1.0),
	beta_y=torch.tensor(1.0),
	num_particles=1000,
	energy=torch.tensor(1e6),
)

# 3) Wrap the segment and beam in a CheetahSimulator
simulator = CheetahSimulator(segment=segment, initial_beam_distribution=beam)

# 4) Define LUME action variables that map to Cheetah element attributes
action_variables = [
	CheetahWritableScalarVariable(
		name="Q1_k1",
		element_name="Q1",
		element_attribute="k1",
	)
]

# 5) Create the LUME model object
model = LUMECheetahModel(simulator=simulator, action_variables=action_variables)

# 6) Use the model API
model.set({"Q1_k1": torch.tensor(2.0)})
current_value = model.get("Q1_k1")
print("Q1.k1 =", current_value)

# Optional: sync model state if the simulator was changed externally
model.update_state()

# Optional: reset simulator + model state to initial conditions
model.reset()
```
