"""
Test archiving with custom Variable subclasses.

Simulates the use case from lume-applications where custom Variables like
ThinQuadrupolePV and SplitSolenoidPV are defined in scripts and used with
LUMECheetahModel. These can't be pickled directly, so we test that the new
archiving strategy (storing class path and config) works correctly.
"""

import tempfile
from io import BytesIO
from pathlib import Path

import h5py
import pytest
import torch
from cheetah.accelerator import Drift, Quadrupole, Segment, Solenoid
from cheetah.particles import ParticleBeam
from lume.actions import WritableActionMixin
from lume_torch.variables import TorchScalarVariable

from lume_cheetah.model import LUMECheetahModel
from lume_cheetah.simulator import CheetahSimulator


class CustomQuadrupolePV(TorchScalarVariable, WritableActionMixin):
    """
    Custom quadrupole PV variable (mimics ThinQuadrupolePV from calibration).

    This variable can't be pickled when used in archived models, but can be
    reconstructed via its class path and config.
    """

    element_name: str
    pv_offset: float = 0.0
    unit: str = "kG-m"

    def _get(self, simulator: CheetahSimulator) -> torch.Tensor:
        element = getattr(simulator.segment, self.element_name)
        if isinstance(element, Quadrupole):
            return element.k1 + self.pv_offset
        return torch.tensor(0.0, dtype=torch.float64)

    def _set(self, simulator: CheetahSimulator, value: float | torch.Tensor) -> None:
        element = getattr(simulator.segment, self.element_name)
        if isinstance(element, Quadrupole):
            strength = torch.as_tensor(value, dtype=element.k1.dtype)
            element.k1 = strength - self.pv_offset


class CustomSolenoidPV(TorchScalarVariable, WritableActionMixin):
    """
    Custom solenoid PV variable (mimics SplitSolenoidPV from calibration).

    Demonstrates multiple attributes that need to be preserved during archiving.
    """

    element_name: str
    scale_factor: float = 2.0
    unit: str = "kG"

    def _get(self, simulator: CheetahSimulator) -> torch.Tensor:
        element = getattr(simulator.segment, self.element_name)
        if isinstance(element, Solenoid):
            return element.k * self.scale_factor
        return torch.tensor(0.0, dtype=torch.float64)

    def _set(self, simulator: CheetahSimulator, value: float | torch.Tensor) -> None:
        element = getattr(simulator.segment, self.element_name)
        if isinstance(element, Solenoid):
            strength = torch.as_tensor(value, dtype=element.k.dtype)
            element.k = strength / self.scale_factor


class TestCustomVariablesArchiving:
    """Test archiving with custom Variable subclasses."""

    @pytest.fixture
    def model_with_custom_vars(self):
        """Create a model with custom variables."""
        segment = Segment(
            [
                Quadrupole(name="q01", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
                Solenoid(name="sol1", length=torch.tensor(0.5), k=torch.tensor(1.0)),
                Drift(name="d1", length=torch.tensor(1.0)),
            ]
        )

        particle_beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            num_particles=100,
            energy=torch.tensor(1e6),
        )

        simulator = CheetahSimulator(
            segment=segment, initial_beam_distribution=particle_beam
        )

        variables = [
            CustomQuadrupolePV(
                name="Q01:BCTRL",
                element_name="q01",
                pv_offset=0.5,
                read_only=False,
            ),
            CustomSolenoidPV(
                name="SOL1:BCTRL",
                element_name="sol1",
                scale_factor=2.5,
                read_only=False,
            ),
        ]

        return LUMECheetahModel(simulator, action_variables=variables)

    def test_custom_variables_archive_to_file(self, model_with_custom_vars):
        """Test archiving model with custom variables to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "custom_vars.h5"
            model_with_custom_vars.to_archive(archive_path)

            # Verify archive structure
            with h5py.File(archive_path, "r") as f:
                assert "simulator" in f
                assert "variables" in f
                assert "Q01:BCTRL" in f["variables"]
                assert "SOL1:BCTRL" in f["variables"]

    def test_custom_variables_archive_stores_config(self, model_with_custom_vars):
        """Test that variable config is correctly stored in archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "custom_vars.h5"
            model_with_custom_vars.to_archive(archive_path)

            with h5py.File(archive_path, "r") as f:
                # Check CustomQuadrupolePV config
                q_group = f["variables"]["Q01:BCTRL"]
                assert (
                    q_group.attrs["class_module"]
                    == "lume_cheetah.tests.test_custom_variables_archiving"
                )
                assert q_group.attrs["class_name"] == "CustomQuadrupolePV"

                # Verify config dataset exists and can be loaded
                assert "config" in q_group
                config_buffer = BytesIO(bytes(q_group["config"][()]))
                q_config = torch.load(config_buffer, weights_only=False)
                assert q_config["name"] == "Q01:BCTRL"
                assert q_config["element_name"] == "q01"
                assert q_config["pv_offset"] == pytest.approx(0.5)
                assert not q_config["read_only"]

                # Check CustomSolenoidPV config
                sol_group = f["variables"]["SOL1:BCTRL"]
                assert (
                    sol_group.attrs["class_module"]
                    == "lume_cheetah.tests.test_custom_variables_archiving"
                )
                assert sol_group.attrs["class_name"] == "CustomSolenoidPV"

                # Verify config dataset exists and can be loaded
                assert "config" in sol_group
                config_buffer = BytesIO(bytes(sol_group["config"][()]))
                sol_config = torch.load(config_buffer, weights_only=False)
                assert sol_config["name"] == "SOL1:BCTRL"
                assert sol_config["element_name"] == "sol1"
                assert sol_config["scale_factor"] == pytest.approx(2.5)

    def test_custom_variables_restore_from_file(self, model_with_custom_vars):
        """Test restoring model with custom variables from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "custom_vars.h5"
            model_with_custom_vars.to_archive(archive_path)

            # Restore
            restored = LUMECheetahModel.from_archive(archive_path)

            # Verify model structure
            assert isinstance(restored, LUMECheetahModel)
            assert "Q01:BCTRL" in restored.supported_variables
            assert "SOL1:BCTRL" in restored.supported_variables

    def test_custom_variables_restored_with_correct_type(self, model_with_custom_vars):
        """Test that restored variables have correct type and attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "custom_vars.h5"
            model_with_custom_vars.to_archive(archive_path)

            restored = LUMECheetahModel.from_archive(archive_path)

            # Check Q variable
            q_var = restored.supported_variables["Q01:BCTRL"]
            assert isinstance(q_var, CustomQuadrupolePV)
            assert q_var.element_name == "q01"
            assert q_var.pv_offset == pytest.approx(0.5)
            assert not q_var.read_only

            # Check SOL variable
            sol_var = restored.supported_variables["SOL1:BCTRL"]
            assert isinstance(sol_var, CustomSolenoidPV)
            assert sol_var.element_name == "sol1"
            assert sol_var.scale_factor == pytest.approx(2.5)

    def test_custom_variables_restored_model_set_get(self, model_with_custom_vars):
        """Test that restored model can set/get custom variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "custom_vars.h5"
            model_with_custom_vars.to_archive(archive_path)

            restored = LUMECheetahModel.from_archive(archive_path)

            # Set value on restored model
            test_value = torch.tensor(2.0)
            restored.set({"Q01:BCTRL": test_value})

            # Verify value was set (should be offset by pv_offset)
            retrieved = restored.get_value("Q01:BCTRL")
            assert retrieved == pytest.approx(test_value)

    def test_custom_variables_roundtrip_preserves_state(self, model_with_custom_vars):
        """Test that archiving/restoring roundtrip preserves model state."""
        # Modify original model
        model_with_custom_vars.set({"Q01:BCTRL": torch.tensor(3.5)})
        model_with_custom_vars.set({"SOL1:BCTRL": torch.tensor(1.5)})

        original_q = model_with_custom_vars.get_value("Q01:BCTRL")
        original_sol = model_with_custom_vars.get_value("SOL1:BCTRL")

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "custom_vars.h5"
            model_with_custom_vars.to_archive(archive_path)

            restored = LUMECheetahModel.from_archive(archive_path)

            # Verify state is preserved
            torch.testing.assert_close(restored.get_value("Q01:BCTRL"), original_q)
            torch.testing.assert_close(restored.get_value("SOL1:BCTRL"), original_sol)

    def test_custom_variables_archive_to_group(self, model_with_custom_vars):
        """Test archiving model with custom variables to HDF5 group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "custom_vars.h5"
            with h5py.File(archive_path, "w") as f:
                group = f.create_group("model")
                model_with_custom_vars.to_archive(group)

                # Verify group structure
                assert "simulator" in group
                assert "variables" in group
                assert "Q01:BCTRL" in group["variables"]
                assert "SOL1:BCTRL" in group["variables"]

    def test_custom_variables_restore_from_group(self, model_with_custom_vars):
        """Test restoring model with custom variables from HDF5 group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "custom_vars.h5"
            with h5py.File(archive_path, "w") as f:
                group = f.create_group("model")
                model_with_custom_vars.to_archive(group)

            # Restore from group
            with h5py.File(archive_path, "r") as f:
                restored = LUMECheetahModel.from_archive(f["model"])

            # Verify restoration
            assert isinstance(restored, LUMECheetahModel)
            assert isinstance(
                restored.supported_variables["Q01:BCTRL"], CustomQuadrupolePV
            )
            assert isinstance(
                restored.supported_variables["SOL1:BCTRL"], CustomSolenoidPV
            )

    def test_missing_custom_variable_class_raises_import_error(self):
        """Test that restoring from archive with missing Variable class raises ImportError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "bad_archive.h5"
            with h5py.File(archive_path, "w") as f:
                # Create simulator dataset (minimal)
                simulator_buffer = BytesIO()
                segment = Segment([Drift(name="d1", length=torch.tensor(1.0))])
                particle_beam = ParticleBeam.from_twiss(
                    beta_x=torch.tensor(1.0),
                    beta_y=torch.tensor(1.0),
                    num_particles=100,
                    energy=torch.tensor(1e6),
                )
                simulator = CheetahSimulator(
                    segment=segment, initial_beam_distribution=particle_beam
                )
                torch.save(simulator, simulator_buffer)
                f.create_dataset(
                    "simulator",
                    data=bytearray(simulator_buffer.getvalue()),
                    dtype="u1",
                )

                # Create a variable group with a non-existent class
                variables_group = f.create_group("variables")
                var_group = variables_group.create_group("bad_var")
                var_group.attrs["class_module"] = "nonexistent.module"
                var_group.attrs["class_name"] = "NonexistentVariable"

                # Create minimal config dataset
                config_buffer = BytesIO()
                torch.save({"name": "bad_var", "read_only": False}, config_buffer)
                var_group.create_dataset(
                    "config",
                    data=bytearray(config_buffer.getvalue()),
                    dtype="u1",
                )

            # Attempt to restore should raise ImportError
            with pytest.raises(ImportError, match="Cannot restore Variable"):
                LUMECheetahModel.from_archive(archive_path)
