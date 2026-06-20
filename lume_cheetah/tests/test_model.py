from cheetah.accelerator import Segment, Quadrupole, Drift
from cheetah.particles import ParticleBeam
import pytest
import torch
from lume_cheetah.actions import CheetahWritableActionMixin
from lume_cheetah.model import LUMECheetahModel
from lume_cheetah.simulator import CheetahSimulator
from lume_torch.variables import TorchScalarVariable

class TestLUMECheetahModel:
    @pytest.fixture
    def model(self):
        # Create a simple accelerator with a quadrupole and a drift
        segment = Segment([
            Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
            Drift(name="D1", length=torch.tensor(1.0)),
        ])

        # Create a simple particle beam
        particle_beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            num_particles=1000,
            energy=torch.tensor(1e6),
        )

        simulator = CheetahSimulator(segment=segment, initial_beam_distribution=particle_beam)

        # Define action variables (e.g., quadrupole strength)
        class Q1K1Variable(TorchScalarVariable, CheetahWritableActionMixin):
            pass

        variables = [
            Q1K1Variable(name="Q1_k", element_name="Q1", attribute_name="k1"),
        ]

        return LUMECheetahModel(simulator, action_variables=variables)

    def test_set_and_get(self, model):
        # Set the quadrupole strength and check if it updates correctly
        model.set({"Q1_k": torch.tensor(2.0)})
        assert model.simulator.segment.Q1.k1 == torch.tensor(2.0)

        # Get the current value of the quadrupole strength
        value = model.get("Q1_k")
        assert value == torch.tensor(2.0)

    def test_update_state_reads_external_simulator_changes(self, model):
        model.simulator.segment.Q1.k1 = torch.tensor(3.0)

        model.update_state()

        assert model.get("Q1_k") == torch.tensor(3.0)

    def test_reset_restores_initial_state(self, model):
        model.set({"Q1_k": torch.tensor(2.0)})

        model.reset()

        assert model.get("Q1_k") == torch.tensor(1.0)