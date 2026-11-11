# lume-cheetah
Cheetah-specific implementation of LUMEModel classes for virtual accelerators

## What this package provides

- `CheetahSimulator`: wraps a Cheetah accelerator `Segment` and initial beam distribution.
- `LUMECheetahModel`: a LUME action model driven by simulator-backed variables.
- `CheetahOverlay`: named controls that update multiple element attributes through
  Python callable rules.

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

## Callable overlays

An overlay maps `(element_name, attribute_name)` targets to functions of its
variable mapping. Pass overlays into the simulator, then assign their variables
as attributes or update several variables together with `set`.

```python
from lume_cheetah import CheetahOverlay, CheetahSimulator

weights = {
    "K21_1B": 0.4142135623730951,
    "K21_1C": 0.2928932188134525,
    "K21_1D": 0.2928932188134525,
}
targets = {}
for name, weight in weights.items():
    targets[name, "voltage"] = (
        lambda v, weight=weight: v["f"] * v["in_use"] * v["ENLD_MeV"] * 1e6 * weight
    )
    targets[name, "phase"] = lambda v: v["phase_deg"]

overlay = CheetahOverlay(
    "K21_1",
    variables={"ENLD_MeV": 100.0, "phase_deg": 0.0, "f": 1.0, "in_use": 1.0},
    targets=targets,
)
# segment contains the three named cavities; beam is the initial distribution.
simulator = CheetahSimulator(
    segment=segment,
    initial_beam_distribution=beam,
    overlays=[overlay],
)
simulator.overlays["K21_1"].phase_deg = 20.0
overlay.set(ENLD_MeV=120.0, f=0.95)
final_beam = simulator.track()
simulator.reset()  # restores initial overlay values and rebinds the same object
```

