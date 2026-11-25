from beamphysics import ParticleGroup
from unittest.mock import patch
from io import BytesIO

from cheetah.accelerator import Segment, Quadrupole, Drift, Screen
from cheetah.particles import ParticleBeam
import pytest
import torch
import h5py
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
        initial_sigma_x = model.get_value("beam:S1:sigma_x").clone()
        if operation in ("set", "reset"):
            model.set({"Q1_k": torch.tensor(2.0)})
            assert not torch.equal(model.get_value("beam:S1:sigma_x"), initial_sigma_x)
        if operation == "reset":
            model.reset()
            torch.testing.assert_close(
                model.get_value("beam:S1:sigma_x"), initial_sigma_x
            )
        elif operation == "initial_particles":
            initial_mu_x = model.get_value("beam:S1:mu_x").clone()
            beam = model.cheetah_initial_particles
            beam.x = beam.x + 0.001
            model.cheetah_initial_particles = beam
            assert not torch.equal(model.get_value("beam:S1:mu_x"), initial_mu_x)

        properties = ("mu_x", "mu_y", "sigma_x", "sigma_y", "sigma_tau")
        names = [f"beam:S1:{prop}" for prop in properties]
        assert set(model.supported_variables) == {"Q1_k", "Q1_k_readback", *names}
        values = model.get(names)
        read_beam = model.simulator.segment.S1.get_read_beam()
        for prop in properties:
            torch.testing.assert_close(
                values[f"beam:S1:{prop}"], getattr(read_beam, prop)
            )

    @pytest.mark.parametrize(
        "name",
        [
            "beam:S1:mu_x",
            "beam:S1:mu_y",
            "beam:S1:sigma_x",
            "beam:S1:sigma_y",
            "beam:S1:sigma_tau",
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

    def test_to_archive_with_file_path(self, model, tmp_path):
        """Test archiving to an HDF5 file using a file path."""
        archive_path = tmp_path / "model_archive.h5"
        model.to_archive(archive_path)
        assert archive_path.exists()
        assert archive_path.stat().st_size > 0

    def test_to_archive_with_hdf5_group(self, model, tmp_path):
        """Test archiving to an HDF5 group."""
        archive_path = tmp_path / "model_archive.h5"
        with h5py.File(archive_path, "w") as f:
            group = f.create_group("model_state")
            model.to_archive(group)
            assert "simulator" in group
            assert "variables" in group
            assert "state" in group

    def test_from_archive_with_file_path(self, model, tmp_path):
        """Test restoring from an HDF5 file using a file path."""
        # Archive the model
        archive_path = tmp_path / "model_archive.h5"
        model.to_archive(archive_path)

        # Restore the model
        restored_model = LUMECheetahModel.from_archive(archive_path)

        # Verify restoration
        assert isinstance(restored_model, LUMECheetahModel)
        torch.testing.assert_close(
            model.get_value("Q1_k"), restored_model.get_value("Q1_k")
        )
        torch.testing.assert_close(
            model.get_value("Q1_k_readback"),
            restored_model.get_value("Q1_k_readback"),
        )

    def test_from_archive_with_hdf5_group(self, model, tmp_path):
        """Test restoring from an HDF5 group."""
        # Archive the model to a group
        archive_path = tmp_path / "model_archive.h5"
        with h5py.File(archive_path, "w") as f:
            group = f.create_group("model_state")
            model.to_archive(group)

        # Restore from the group
        with h5py.File(archive_path, "r") as f:
            restored_model = LUMECheetahModel.from_archive(f["model_state"])

        # Verify restoration
        assert isinstance(restored_model, LUMECheetahModel)
        torch.testing.assert_close(
            model.get_value("Q1_k"), restored_model.get_value("Q1_k")
        )

    def test_archive_preserves_state(self, model, tmp_path):
        """Test that archiving and restoring preserves model state."""
        # Modify the model state
        model.set({"Q1_k": torch.tensor(2.5)})
        expected_k = model.get_value("Q1_k")

        # Archive and restore
        archive_path = tmp_path / "model_archive.h5"
        model.to_archive(archive_path)
        restored_model = LUMECheetahModel.from_archive(archive_path)

        # Verify the state is preserved
        torch.testing.assert_close(restored_model.get_value("Q1_k"), expected_k)

    def test_archive_preserves_final_distribution(self, model, tmp_path):
        """Test that archiving preserves the final particle distribution."""
        # Set up the model and get the final distribution
        model.set({"Q1_k": torch.tensor(2.0)})
        original_final = model.cheetah_final_particles

        # Archive and restore
        archive_path = tmp_path / "model_archive.h5"
        model.to_archive(archive_path)
        restored_model = LUMECheetahModel.from_archive(archive_path)

        # Verify the final distribution is preserved (allow for dtype differences)
        torch.testing.assert_close(
            original_final.particles.to(torch.float64),
            restored_model.cheetah_final_particles.particles.to(torch.float64),
            atol=1e-6,
            rtol=1e-5,
        )

    def test_from_archive_wrong_type_raises_error(self, tmp_path):
        """Test that from_archive raises error for incompatible archive."""
        # Create an archive with wrong structure (no simulator)
        archive_path = tmp_path / "wrong_model.h5"
        with h5py.File(archive_path, "w") as f:
            payload = BytesIO()
            torch.save(torch.tensor([1, 2, 3]), payload)
            f.create_dataset("dummy", data=bytearray(payload.getvalue()), dtype="u1")

        # Try to restore as LUMECheetahModel
        with pytest.raises(TypeError, match="must contain a LUMECheetahModel"):
            LUMECheetahModel.from_archive(archive_path)

    def test_archive_roundtrip_with_multiple_operations(self, model, tmp_path):
        """Test archive roundtrip after multiple model operations."""
        # Perform multiple operations
        model.set({"Q1_k": torch.tensor(1.5)})
        model.reset()
        model.set({"Q1_k": torch.tensor(3.0)})

        expected_state = {
            "Q1_k": model.get_value("Q1_k"),
            "Q1_k_readback": model.get_value("Q1_k_readback"),
        }

        # Archive and restore
        archive_path = tmp_path / "model_archive.h5"
        model.to_archive(archive_path)
        restored_model = LUMECheetahModel.from_archive(archive_path)

        # Verify all state is preserved
        for name, expected_value in expected_state.items():
            torch.testing.assert_close(restored_model.get_value(name), expected_value)

    def test_archive_preserves_reference_energy_with_momentum_spread(
        self, model, tmp_path
    ):
        """Test that reference energy is preserved for beams with momentum spread.

        This is critical for beams with nonzero mean relative momentum deviation,
        where using the mean energy instead of the reference energy would
        renormalize momentum coordinates and change the beam.
        """
        # Set up the model and run it
        model.set({"Q1_k": torch.tensor(2.0)})
        original_final = model.cheetah_final_particles

        # Add momentum spread to the final distribution
        final_beam_spread = original_final.clone()
        # Add 2% energy spread by modifying the p coordinate (column 5)
        final_beam_spread.particles[:, 5] = final_beam_spread.particles[
            :, 5
        ] + 0.02 * torch.randn(original_final.num_particles)
        model._final_distribution = final_beam_spread

        original_energy = model._final_distribution.energy
        original_particles = model._final_distribution.particles.clone()

        # Archive and restore
        archive_path = tmp_path / "model_archive.h5"
        model.to_archive(archive_path)
        restored_model = LUMECheetahModel.from_archive(archive_path)

        # Verify the reference energy is preserved
        restored_energy = restored_model._final_distribution.energy
        torch.testing.assert_close(restored_energy, original_energy)

        # Verify the particles are preserved (not renormalized by mean energy)
        torch.testing.assert_close(
            original_particles.to(torch.float64),
            restored_model._final_distribution.particles.to(torch.float64),
            atol=1e-5,
            rtol=1e-5,
        )
