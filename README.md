# lume-cheetah
Cheetah-specific implementation of LUMEModel classes for virtual accelerators

## What this package provides

- `CheetahSimulator`: wraps a Cheetah accelerator `Segment` and initial beam distribution.
- `LUMECheetahModel`: a LUME action model driven by simulator-backed variables.
- `CheetahStagedModel`: passes Cheetah beams between stages while preserving autograd.

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

## Implementation notes
Staged execution batches each incoming beam with its stage's control updates
before tracking. Full runs track each `LUMECheetahModel` once, including nested
stages; the first `set()` initializes the pipeline in that same pass. Subsequent
partial updates retain cached upstream results and evaluate stages receiving
beams even when their controls have not changed.

Direct assignment to `LUMECheetahModel.cheetah_initial_particles` or
`initial_particles` still tracks immediately and refreshes state. Custom stages
with eager particle setters can override
`CheetahInitialParticlesMixIn._set_cheetah_initial_particles` to store the beam
without tracking; staged execution follows this hook with `set()`.

Assigning to a staged model's `cheetah_initial_particles` or `initial_particles`
immediately evaluates the whole pipeline, including nested stages. Each
`LUMECheetahModel` stage tracks once, and final particles are available when the
assignment returns. Direct changes to child stages do not automatically run the
containing pipeline; use its `set()` or `run_start_to_end()` to propagate those changes.
