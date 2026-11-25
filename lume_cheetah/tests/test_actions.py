from cheetah.accelerator import Drift, Screen, Segment
from cheetah.particles import ParticleBeam
import pytest
import torch

from lume.actions import ReadOnlyActionMixin
from lume_cheetah.actions import (
    CheetahBeamPropertyVariable,
    create_builtin_actions,
)
from lume_cheetah.simulator import CheetahSimulator


@pytest.mark.parametrize("screen_names", [(), ("S1",), ("S1", "S2")])
def test_create_builtin_actions(screen_names):
    screens = [
        Screen(
            name=name,
            is_active=True,
            misalignment=torch.tensor([index * 0.001, 0.0]),
        )
        for index, name in enumerate(screen_names)
    ]
    simulator = CheetahSimulator(
        segment=Segment([Drift(name="D1", length=torch.tensor(1.0)), *screens]),
        initial_beam_distribution=ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            num_particles=128,
            energy=torch.tensor(1e8),
        ),
    )

    actions = create_builtin_actions(simulator)

    properties = ("mu_x", "mu_y", "sigma_x", "sigma_y", "sigma_tau")
    expected_names = {
        name
        for screen_name in screen_names
        for name in (*(f"beam:{screen_name}:{prop}" for prop in properties),)
    }
    assert len(actions) == len(expected_names)
    assert {action.name for action in actions} == expected_names
    by_name = {action.name: action for action in actions}
    for screen in screens:
        for prop in properties:
            action = by_name[f"beam:{screen.name}:{prop}"]
            assert isinstance(action, CheetahBeamPropertyVariable)
            assert isinstance(action, ReadOnlyActionMixin)
            assert action.read_only
            assert action.element_name == screen.name
            assert action.beam_property == prop
            read_beam = screen.get_read_beam()
            torch.testing.assert_close(action._get(simulator), getattr(read_beam, prop))
