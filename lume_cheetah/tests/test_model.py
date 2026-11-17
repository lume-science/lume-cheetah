from beamphysics import ParticleGroup
from unittest.mock import patch

from cheetah.accelerator import Segment, Quadrupole, Drift, Screen
from cheetah.particles import ParticleBeam
import pytest
import torch
from lume.exceptions import ReadOnlyError
from lume_cheetah.actions import (
    CheetahWritableScalarVariable,
    CheetahReadOnlyScalarVariable,
)
from lume_cheetah.model import LUMECheetahModel
from lume_cheetah.simulator import CheetahSimulator


class TestLUMECheetahModel:
    @pytest.fixture
    def model(self):
        # Create a simple accelerator with a quadrupole and a drift
        segment = Segment(
            [
                Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
                Drift(name="D1", length=torch.tensor(1.0)),
            ]
        )

        # Create a simple particle beam
        particle_beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            num_particles=1000,
            energy=torch.tensor(1e6),
        )

        simulator = CheetahSimulator(
            segment=segment, initial_beam_distribution=particle_beam
        )

        variables = [
            CheetahWritableScalarVariable(
                name="Q1_k", element_name="Q1", element_attribute="k1"
            ),
            CheetahReadOnlyScalarVariable(
                name="Q1_k_readback", element_name="Q1", element_attribute="k1"
            ),
        ]

        return LUMECheetahModel(simulator, action_variables=variables)

    @pytest.fixture
    def screen_model(self, model):
        segment = Segment(
            [
                *model.simulator.segment.elements,
                Screen(name="S1", is_active=True),
            ]
        )
        simulator = CheetahSimulator(
            segment=segment,
            initial_beam_distribution=model.cheetah_initial_particles,
        )
        return LUMECheetahModel(
            simulator, action_variables=list(model.supported_variables.values())
        )

    @pytest.mark.parametrize(
        "operation", ["initialize", "set", "reset", "initial_particles"]
    )
    def test_builtin_screen_outputs(self, screen_model, operation):
        model = screen_model
        initial_sigma_x = model.get_value("beam:S1_sigma_x").clone()
        if operation in ("set", "reset"):
            model.set({"Q1_k": torch.tensor(2.0)})
            assert not torch.equal(model.get_value("beam:S1_sigma_x"), initial_sigma_x)
        if operation == "reset":
            model.reset()
            torch.testing.assert_close(
                model.get_value("beam:S1_sigma_x"), initial_sigma_x
            )
        elif operation == "initial_particles":
            initial_mu_x = model.get_value("beam:S1_mu_x").clone()
            beam = model.cheetah_initial_particles
            beam.x = beam.x + 0.001
            model.cheetah_initial_particles = beam
            assert not torch.equal(model.get_value("beam:S1_mu_x"), initial_mu_x)

        properties = ("mu_x", "mu_y", "sigma_x", "sigma_y", "sigma_tau")
        names = ["beam:S1", *(f"beam:S1_{prop}" for prop in properties)]
        assert set(model.supported_variables) == {"Q1_k", "Q1_k_readback", *names}
        values = model.get(names)
        read_beam = model.simulator.segment.S1.get_read_beam()
        assert isinstance(values["beam:S1"], ParticleGroup)
        assert values["beam:S1"].n_particle == 1000
        assert values["beam:S1"] == read_beam.to_openpmd_particlegroup()
        for prop in properties:
            torch.testing.assert_close(
                values[f"beam:S1_{prop}"], getattr(read_beam, prop)
            )

    @pytest.mark.parametrize(
        "name",
        [
            "beam:S1",
            "beam:S1_mu_x",
            "beam:S1_mu_y",
            "beam:S1_sigma_x",
            "beam:S1_sigma_y",
            "beam:S1_sigma_tau",
        ],
    )
    def test_builtin_screen_outputs_are_read_only(self, screen_model, name):
        value = screen_model.get_value(name)
        with patch.object(
            screen_model.simulator, "track", wraps=screen_model.simulator.track
        ) as track:
            with pytest.raises(ReadOnlyError):
                screen_model.set({name: value})
            track.assert_not_called()

    def test_set_and_get(self, model):
        # Set the quadrupole strength and check if it updates correctly
        model.set({"Q1_k": torch.tensor(2.0)})
        assert model.simulator.segment.Q1.k1 == torch.tensor(2.0)

        # Get the current value of the quadrupole strength
        value = model.get_value("Q1_k")
        assert value == torch.tensor(2.0)

        # test getting the read-only variable
        readback_value = model.get_value("Q1_k_readback")
        assert readback_value == torch.tensor(2.0)

        # try to set the read-only variable and expect an error
        with pytest.raises(ReadOnlyError):
            model.set({"Q1_k_readback": torch.tensor(3.0)})

    def test_update_state_reads_external_simulator_changes(self, model):
        model.simulator.segment.Q1.k1 = torch.tensor(3.0)

        model.update_state()

        assert model.get_value("Q1_k") == torch.tensor(3.0)

    def test_reset_restores_initial_state(self, model):
        model.set({"Q1_k": torch.tensor(2.0)})

        model.reset()

        assert model.get_value("Q1_k") == torch.tensor(1.0)

    def test_staged_model_initial_and_final_particles(self, model):
        # Check that the initial particles are set correctly
        initial_particles = model.initial_particles
        assert isinstance(initial_particles, ParticleGroup)
        assert initial_particles.n_particle == 1000

        # Run the model to get final particles
        model.set({"Q1_k": torch.tensor(2.0)})
        final_particles = model.final_particles
        assert isinstance(final_particles, ParticleGroup)
        assert final_particles.n_particle == 1000

    @pytest.mark.parametrize("particle_format", ["cheetah", "openpmd"])
    def test_initial_particle_assignment_tracks_once(self, model, particle_format):
        model.double()
        beam = model.cheetah_initial_particles
        beam.x = beam.x + 0.001
        with patch.object(
            model.simulator, "track", wraps=model.simulator.track
        ) as track:
            if particle_format == "cheetah":
                model.cheetah_initial_particles = beam
            else:
                model.initial_particles = beam.to_openpmd_particlegroup()
            track.assert_called_once_with()

        torch.testing.assert_close(model.cheetah_initial_particles.x, beam.x)
        expected = model.simulator.segment.track(model.simulator.beam_distribution)
        torch.testing.assert_close(
            model.cheetah_final_particles.particles, expected.particles
        )
