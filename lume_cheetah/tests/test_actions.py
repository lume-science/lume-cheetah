from types import SimpleNamespace

import torch
from cheetah.accelerator import Drift, Quadrupole, Segment
from cheetah.particles import ParticleBeam
from lume_torch.variables import TorchScalarVariable

from lume_cheetah.actions import (
    CheetahReadonlyActionMixin,
    CheetahWritableActionMixin,
    _get_variable_value,
)
from lume_cheetah.simulator import CheetahSimulator


def _make_simulator():
    segment = Segment(
        [
            Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
            Drift(name="D1", length=torch.tensor(1.0)),
        ]
    )
    beam = ParticleBeam.from_twiss(
        beta_x=torch.tensor(1.0),
        beta_y=torch.tensor(1.0),
        num_particles=256,
        energy=torch.tensor(1e6),
    )
    return CheetahSimulator(segment=segment, initial_beam_distribution=beam)


def test_get_variable_value_reads_segment_attribute():
    simulator = _make_simulator()
    variable = SimpleNamespace(element_name="Q1", attribute_name="k1")

    value = _get_variable_value(simulator, variable)

    assert torch.equal(value, torch.tensor(1.0))


def test_writable_action_mixin_set_and_get():
    simulator = _make_simulator()

    class Q1K1Variable(TorchScalarVariable, CheetahWritableActionMixin):
        pass

    variable = Q1K1Variable(name="Q1_k", element_name="Q1", attribute_name="k1")
    variable._set(simulator, torch.tensor(2.5))

    assert torch.equal(variable._get(simulator), torch.tensor(2.5))
    assert torch.equal(simulator.segment.Q1.k1, torch.tensor(2.5))


def test_readonly_action_mixin_get():
    simulator = _make_simulator()

    class Q1ReadonlyVariable(TorchScalarVariable, CheetahReadonlyActionMixin):
        pass

    variable = Q1ReadonlyVariable(name="Q1_k", element_name="Q1", attribute_name="k1")

    assert torch.equal(variable._get(simulator), torch.tensor(1.0))