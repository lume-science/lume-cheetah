"""Tests for CheetahStagedModel class.

Consolidated test suite for CheetahStagedModel with:
- Reduced overlapping test coverage
- Comprehensive gradient propagation tests between stages
- Full workflow validation
"""

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

from cheetah.particles import ParticleBeam
from cheetah.accelerator import Drift, Segment

from lume.model import LUMEModel
from lume.staged_model import InitialParticlesMixIn, FinalParticlesMixIn
from lume.variables import ScalarVariable, Variable
from lume_torch.variables import TorchScalarVariable
from lume_cheetah.actions import CheetahWritableScalarVariable
from lume_cheetah.model import LUMECheetahModel
from lume_cheetah.simulator import CheetahSimulator

from lume_cheetah.staged_model import (
    CheetahInitialParticlesMixIn,
    CheetahFinalParticlesMixIn,
    CheetahStagedModel,
)


def make_test_particle_group(
    n_particles: int = 8,
    x_offset: float = 0.0,
    pz: float = 1.0e7,
) -> ParticleGroup:
    """Create a test ParticleGroup with a given number of particles."""
    x = np.linspace(-1.0e-3, 1.0e-3, n_particles) + x_offset
    zeros = np.zeros(n_particles)
    return ParticleGroup(
        data={
            "x": x,
            "px": zeros,
            "y": zeros,
            "py": zeros,
            "z": zeros,
            "pz": np.full(n_particles, pz),
            "t": zeros,
            "status": np.ones(n_particles),
            "weight": np.full(n_particles, 1.0e-12),
            "species": "electron",
        }
    )


def make_test_particle_beam(
    n_particles: int = 8,
    x_offset: float = 0.0,
) -> ParticleBeam:
    """Create a test ParticleBeam with a given number of particles."""
    pg = make_test_particle_group(n_particles, x_offset)
    energy = torch.tensor(pg.energy.mean())
    return ParticleBeam.from_openpmd_particlegroup(pg, energy=energy)


class SourceModel(
    LUMEModel,
    InitialParticlesMixIn,
    FinalParticlesMixIn,
    CheetahInitialParticlesMixIn,
    CheetahFinalParticlesMixIn,
):
    """Source model: generates particles with controllable phase offset."""

    def __init__(self):
        super().__init__()
        self.call_count_set = 0
        self._variables = {
            "phase": TorchScalarVariable(
                name="phase", default_value=0.0, read_only=False
            ),
        }
        self._state = {"phase": 0.0}
        self._initial_state = self._state.copy()
        beam = make_test_particle_beam(n_particles=8, x_offset=0.0)
        self._cheetah_initial_particles = beam

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def cheetah_initial_particles(self) -> ParticleBeam:
        return self._cheetah_initial_particles

    @cheetah_initial_particles.setter
    def cheetah_initial_particles(self, val: ParticleBeam) -> None:
        self._cheetah_initial_particles = val

    @property
    def cheetah_final_particles(self) -> ParticleBeam:
        beam = self._cheetah_initial_particles.clone()
        beam.x = beam.x + self._state["phase"]
        return beam

    @property
    def initial_particles(self) -> ParticleGroup:
        return self.cheetah_initial_particles.to_openpmd_particlegroup()

    @initial_particles.setter
    def initial_particles(self, val: ParticleGroup) -> None:
        energy = torch.tensor(val.energy.mean())
        self.cheetah_initial_particles = ParticleBeam.from_openpmd_particlegroup(
            val, energy=energy
        )

    @property
    def final_particles(self) -> ParticleGroup:
        return self.cheetah_final_particles.to_openpmd_particlegroup()

    def set(self, values: dict[str, Any]) -> None:
        self.call_count_set += 1
        super().set(values)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        if "phase" in values:
            self._state["phase"] = values["phase"]

    def reset(self) -> None:
        self._state = self._initial_state.copy()


class ProcessModel(
    LUMEModel,
    InitialParticlesMixIn,
    FinalParticlesMixIn,
    CheetahInitialParticlesMixIn,
    CheetahFinalParticlesMixIn,
):
    """Process model: scales particle positions by a factor."""

    def __init__(self):
        super().__init__()
        self.call_count_set = 0
        self._variables = {
            "scale": ScalarVariable(name="scale", default_value=1.0, read_only=False),
        }
        self._state = {"scale": 1.0}
        self._initial_state = self._state.copy()
        beam = make_test_particle_beam(n_particles=8, x_offset=0.0)
        self._cheetah_initial_particles = beam
        self._cheetah_final_particles = beam

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def cheetah_initial_particles(self) -> ParticleBeam:
        return self._cheetah_initial_particles

    @cheetah_initial_particles.setter
    def cheetah_initial_particles(self, val: ParticleBeam) -> None:
        self._cheetah_initial_particles = val

    @property
    def cheetah_final_particles(self) -> ParticleBeam:
        scale = self._state["scale"]
        beam = self._cheetah_initial_particles.clone()
        beam.x = beam.x * scale
        return beam

    @property
    def initial_particles(self) -> ParticleGroup:
        return self.cheetah_initial_particles.to_openpmd_particlegroup()

    @initial_particles.setter
    def initial_particles(self, val: ParticleGroup) -> None:
        energy = torch.tensor(val.energy.mean())
        self.cheetah_initial_particles = ParticleBeam.from_openpmd_particlegroup(
            val, energy=energy
        )

    @property
    def final_particles(self) -> ParticleGroup:
        return self.cheetah_final_particles.to_openpmd_particlegroup()

    def set(self, values: dict[str, Any]) -> None:
        self.call_count_set += 1
        super().set(values)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        if "scale" in values:
            self._state["scale"] = values["scale"]

    def reset(self) -> None:
        self._state = self._initial_state.copy()


class DampingModel(
    LUMEModel,
    InitialParticlesMixIn,
    FinalParticlesMixIn,
    CheetahInitialParticlesMixIn,
    CheetahFinalParticlesMixIn,
):
    """Damping model: applies damping factor to particle positions."""

    def __init__(self):
        super().__init__()
        self.call_count_set = 0
        self._variables = {
            "damping": ScalarVariable(
                name="damping", default_value=0.1, read_only=False
            ),
        }
        self._state = {"damping": 0.1}
        self._initial_state = self._state.copy()
        beam = make_test_particle_beam(n_particles=8, x_offset=0.0)
        self._cheetah_initial_particles = beam
        self._cheetah_final_particles = beam

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def cheetah_initial_particles(self) -> ParticleBeam:
        return self._cheetah_initial_particles

    @cheetah_initial_particles.setter
    def cheetah_initial_particles(self, val: ParticleBeam) -> None:
        self._cheetah_initial_particles = val

    @property
    def cheetah_final_particles(self) -> ParticleBeam:
        damping = self._state["damping"]
        beam = self._cheetah_initial_particles.clone()
        beam.x = beam.x * (1.0 - damping)
        return beam

    @property
    def initial_particles(self) -> ParticleGroup:
        return self.cheetah_initial_particles.to_openpmd_particlegroup()

    @initial_particles.setter
    def initial_particles(self, val: ParticleGroup) -> None:
        energy = torch.tensor(val.energy.mean())
        self.cheetah_initial_particles = ParticleBeam.from_openpmd_particlegroup(
            val, energy=energy
        )

    @property
    def final_particles(self) -> ParticleGroup:
        return self.cheetah_final_particles.to_openpmd_particlegroup()

    def set(self, values: dict[str, Any]) -> None:
        self.call_count_set += 1
        super().set(values)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        if "damping" in values:
            self._state["damping"] = values["damping"]

    def reset(self) -> None:
        self._state = self._initial_state.copy()


class SinkModel(LUMEModel, CheetahInitialParticlesMixIn):
    """Sink model: only accepts particles, no final distribution."""

    def __init__(self):
        super().__init__()
        self.call_count_set = 0
        self._variables = {
            "damping": ScalarVariable(
                name="damping", default_value=0.1, read_only=False
            ),
        }
        self._state = {"damping": 0.1}
        self._initial_state = self._state.copy()
        beam = make_test_particle_beam(n_particles=8, x_offset=0.0)
        self._cheetah_initial_particles = beam

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def cheetah_initial_particles(self) -> ParticleBeam:
        return self._cheetah_initial_particles

    @cheetah_initial_particles.setter
    def cheetah_initial_particles(self, val: ParticleBeam) -> None:
        self._cheetah_initial_particles = val

    def set(self, values: dict[str, Any]) -> None:
        self.call_count_set += 1
        super().set(values)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        if "damping" in values:
            self._state["damping"] = values["damping"]

    def reset(self) -> None:
        self._state = self._initial_state.copy()


class TestCheetahStagedModelValidation:
    """Test model validation for CheetahStagedModel."""

    def test_validation_accepts_valid_configs(self):
        """Test that valid staged model configurations are accepted."""
        # Single model
        source = SourceModel()
        model = CheetahStagedModel([source])
        assert len(model.lume_model_instances) == 1

        # Two stages
        process = ProcessModel()
        model = CheetahStagedModel([source, process])
        assert len(model.lume_model_instances) == 2

        # Three stages
        damping = DampingModel()
        model = CheetahStagedModel([source, process, damping])
        assert len(model.lume_model_instances) == 3

    def test_validation_rejects_invalid_model_types(self):
        """Test that non-LUMEModel instances are rejected."""
        source = SourceModel()
        with pytest.raises((TypeError, ValueError)):
            CheetahStagedModel([source, "not a model"])

    def test_validation_rejects_missing_initial_particles_mixin(self):
        """Test that models missing InitialParticlesMixIn are rejected in non-first positions."""

        class NoInitialModel(
            LUMEModel, FinalParticlesMixIn, CheetahFinalParticlesMixIn
        ):
            @property
            def supported_variables(self) -> dict[str, Variable]:
                return {}

            @property
            def final_particles(self) -> ParticleGroup:
                return self.cheetah_final_particles.to_openpmd_particlegroup()

            @property
            def cheetah_final_particles(self) -> ParticleBeam:
                return make_test_particle_beam()

            def _get(self, names: list[str]) -> dict[str, Any]:
                return {}

            def _set(self, values: dict[str, Any]) -> None:
                pass

            def reset(self) -> None:
                pass

        source = SourceModel()
        process = NoInitialModel()
        with pytest.raises(ValueError, match="must implement InitialParticlesMixIn"):
            CheetahStagedModel([source, process])

    def test_validation_rejects_duplicate_variables(self):
        """Test that models with duplicate variable names are rejected."""

        class Model1(
            LUMEModel,
            InitialParticlesMixIn,
            FinalParticlesMixIn,
            CheetahInitialParticlesMixIn,
            CheetahFinalParticlesMixIn,
        ):
            def __init__(self):
                super().__init__()
                self._variables = {
                    "param": ScalarVariable(name="param", default_value=1.0)
                }
                self._state = {"param": 1.0}
                self._initial_state = self._state.copy()
                self._beam = make_test_particle_beam()

            @property
            def supported_variables(self) -> dict[str, Variable]:
                return self._variables

            @property
            def cheetah_initial_particles(self) -> ParticleBeam:
                return self._beam

            @cheetah_initial_particles.setter
            def cheetah_initial_particles(self, val: ParticleBeam):
                self._beam = val

            @property
            def cheetah_final_particles(self) -> ParticleBeam:
                return self._beam

            @property
            def initial_particles(self) -> ParticleGroup:
                return self.cheetah_initial_particles.to_openpmd_particlegroup()

            @initial_particles.setter
            def initial_particles(self, val: ParticleGroup):
                energy = torch.tensor(val.energy.mean())
                self.cheetah_initial_particles = (
                    ParticleBeam.from_openpmd_particlegroup(val, energy=energy)
                )

            @property
            def final_particles(self) -> ParticleGroup:
                return self.cheetah_final_particles.to_openpmd_particlegroup()

            def _get(self, names: list[str]) -> dict[str, Any]:
                return {name: self._state[name] for name in names}

            def _set(self, values: dict[str, Any]) -> None:
                if "param" in values:
                    self._state["param"] = values["param"]

            def reset(self) -> None:
                self._state = self._initial_state.copy()

        model1 = Model1()
        model2 = Model1()
        with pytest.raises(ValueError, match="Variable 'param' is defined"):
            CheetahStagedModel([model1, model2])


class TestCheetahStagedModelExecution:
    """Test core execution and particle propagation."""

    def test_run_start_to_end_propagates_particles(self):
        """Test that particles flow correctly through stages."""
        source = SourceModel()
        process = ProcessModel()
        damping = DampingModel()
        model = CheetahStagedModel([source, process, damping])

        model.run_start_to_end()

        assert source.call_count_set == 1
        assert process.call_count_set == 1
        assert damping.call_count_set == 1
        assert model._final_distribution is not None

    def test_set_propagates_particles_and_updates_state(self):
        """Test that set() updates state and propagates particles."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])

        model.set({"phase": 0.5, "scale": 2.0})

        assert source._state["phase"] == 0.5
        assert process._state["scale"] == 2.0
        assert process.cheetah_initial_particles is not None

    def test_set_partial_variables_only_updates_relevant_models(self):
        """Test that set() with partial variables only affects relevant models."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])

        initial_scale = process._state["scale"]
        model.set({"phase": 0.5})

        assert source._state["phase"] == 0.5
        assert process._state["scale"] == initial_scale
        assert process.cheetah_initial_particles is not None

    def test_set_empty_dict_runs_start_to_end(self):
        """Test that set({}) still runs the full pipeline."""
        source = SourceModel()
        model = CheetahStagedModel([source])

        model.set({})

        # Should have been called once in run_start_to_end
        assert source.call_count_set == 1

    @pytest.mark.parametrize("nested", ["none", "first", "last"])
    @pytest.mark.parametrize(
        "operation", ["run", "set_all", "set_first", "set_last", "set_empty"]
    )
    def test_real_stages_track_once_per_evaluation(
        self, monkeypatch, nested, operation
    ):
        """Check evaluation counts, results, and gradients for flat/nested stages."""
        beam = make_test_particle_beam().to(dtype=torch.float32)
        beam.px = torch.full_like(beam.px, 0.01)  # Nonzero drift-length sensitivity.
        stages = [
            LUMECheetahModel(
                CheetahSimulator(
                    segment=Segment([Drift(name=f"D{i}", length=torch.tensor(0.5))]),
                    initial_beam_distribution=beam,
                ),
                [
                    CheetahWritableScalarVariable(
                        name=f"length_{i}",  # Unique names route controls to each stage.
                        element_name=f"D{i}",
                        element_attribute="length",
                    )
                ],
            )
            for i in range(3)
        ]
        # Keep the physical lattice identical while varying its staged grouping.
        if nested == "first":
            model = CheetahStagedModel([CheetahStagedModel(stages[:2]), stages[2]])
        elif nested == "last":
            model = CheetahStagedModel([stages[0], CheetahStagedModel(stages[1:])])
        else:
            model = CheetahStagedModel(stages)
        # Spy after construction: count requested evaluations, not simulator setup.
        # wraps preserves real tracking, including its autograd graph.
        tracks = []
        for stage in stages:
            track = Mock(wraps=stage.simulator.track)
            monkeypatch.setattr(stage.simulator, "track", track)
            tracks.append(track)

        for iteration in range(2):  # Exercise initialization, then cached execution.
            length = torch.tensor(
                0.75 + iteration, dtype=beam.x.dtype, requires_grad=True
            )
            values = {
                "run": {},
                "set_all": {f"length_{i}": length for i in range(3)},
                "set_first": {"length_0": length},
                "set_last": {"length_2": length},
                "set_empty": {},
            }[operation]
            if operation == "run":
                model.run_start_to_end()
            else:
                model.set(values)

            # The first stage runs on initialization, full runs, or its own updates.
            first_runs = int(
                iteration == 0 or operation in ("run", "set_all", "set_first")
            )
            # Beam recipients run once. A cached first nested group skips both leaves;
            # otherwise the second leaf receives a beam and must run.
            expected_calls = [first_runs, first_runs if nested == "first" else 1, 1]
            assert [track.call_count for track in tracks] == expected_calls
            # Reference tracking bypasses model handoffs and the simulator spies.
            expected = beam
            for stage in stages:
                expected = stage.simulator.segment.track(expected)
            torch.testing.assert_close(
                model.cheetah_final_particles.particles, expected.particles
            )
            # Reading state must neither return stale controls nor retrigger initialization.
            readbacks = model.get(["length_0", "length_1", "length_2"])
            for name, value in values.items():
                torch.testing.assert_close(readbacks[name], value)
            assert [track.call_count for track in tracks] == expected_calls

            if operation in ("set_all", "set_first"):
                # For these drifts, dx/dlength = px. The shared input tensor contributes
                # once per controlled drift, including the first stage in both cases.
                model.cheetah_final_particles.x.sum().backward()
                assert length.grad is not None
                multiplier = 3 if operation == "set_all" else 1
                torch.testing.assert_close(
                    length.grad, (beam.px.sum() * multiplier).to(length)
                )
            for track in tracks:
                track.reset_mock()  # Assert per-pass counts, not cumulative totals.


class TestCheetahStagedModelProperties:
    """Test property getters and setters."""

    def test_cheetah_initial_particles_getter_and_setter(self):
        """Test initial particles property."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])

        # Getter
        particles = model.cheetah_initial_particles
        assert isinstance(particles, ParticleBeam)

        # Setter
        new_particles = make_test_particle_beam(n_particles=16)
        model.cheetah_initial_particles = new_particles
        assert source.cheetah_initial_particles is new_particles

    def test_cheetah_final_particles_getter(self):
        """Test final particles property after initialization."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])
        model.run_start_to_end()

        particles = model.cheetah_final_particles
        assert isinstance(particles, ParticleBeam)

    def test_cheetah_final_particles_error_before_init(self):
        """Test error when accessing final particles before initialization."""
        source = SourceModel()
        model = CheetahStagedModel([source])

        with pytest.raises(RuntimeError, match="final distribution is none"):
            _ = model.cheetah_final_particles

    def test_initial_and_final_particles_conversion(self):
        """Test ParticleGroup conversions for initial and final particles."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])

        # Initial particles conversion
        pg_initial = model.initial_particles
        assert isinstance(pg_initial, ParticleGroup)

        # Set from ParticleGroup
        pg_new = make_test_particle_group(n_particles=16)
        model.initial_particles = pg_new
        assert source.cheetah_initial_particles is not None

        # Final particles conversion
        model.run_start_to_end()
        pg_final = model.final_particles
        assert isinstance(pg_final, ParticleGroup)

    def test_final_particles_is_clone(self):
        """Test that final_particles returns independent clones."""
        source = SourceModel()
        model = CheetahStagedModel([source])
        model.run_start_to_end()

        particles1 = model.cheetah_final_particles
        particles2 = model.cheetah_final_particles
        assert torch.allclose(particles1.x, particles2.x)


class TestCheetahStagedModelVariables:
    """Test variable aggregation and access."""

    def test_supported_variables_aggregates_from_all_models(self):
        """Test that supported_variables combines variables from all models."""
        source = SourceModel()
        process = ProcessModel()
        damping = DampingModel()
        model = CheetahStagedModel([source, process, damping])

        variables = model.supported_variables
        assert "phase" in variables
        assert "scale" in variables
        assert "damping" in variables

    def test_get_returns_values_from_all_models(self):
        """Test that get() retrieves values across all models."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])

        model.set({"phase": 1.0, "scale": 2.0})
        values = model.get(["phase", "scale"])

        assert values["phase"] == 1.0
        assert values["scale"] == 2.0

    def test_get_nonexistent_variable_raises_error(self):
        """Test error handling for nonexistent variables."""
        source = SourceModel()
        model = CheetahStagedModel([source])

        with pytest.raises(ValueError):
            model.get(["nonexistent_variable"])


class TestGradientPropagation:
    """Verify backward reaches first-stage inputs across stage boundaries."""

    @pytest.mark.parametrize("n_stages", [2, 3])
    @pytest.mark.parametrize("execution", ["set", "run_start_to_end"])
    def test_backward_to_first_stage_inputs(self, n_stages, execution):
        source = SourceModel()
        process = ProcessModel()
        stages = [source, process]
        values = {"scale": 2.0}
        factor = 2.0
        if n_stages == 3:
            stages.append(DampingModel())
            values["damping"] = 0.25
            factor *= 0.75
        model = CheetahStagedModel(stages)

        initial_beam = make_test_particle_beam(n_particles=8)
        # Keep the leaf: ParticleBeam.x returns a new, non-leaf view on each access.
        initial_x = initial_beam.x.clone().detach().requires_grad_(True)
        initial_beam.x = initial_x
        phase = torch.tensor(0.002, dtype=initial_x.dtype, requires_grad=True)
        model.cheetah_initial_particles = initial_beam

        if execution == "set":
            model.set({**values, "phase": phase})
        else:
            source.set({"phase": phase})
            process.set({"scale": values["scale"]})
            if n_stages == 3:
                stages[-1].set({"damping": values["damping"]})
            model.run_start_to_end()

        final_x = model.cheetah_final_particles.x
        weights = torch.arange(1, initial_x.numel() + 1, dtype=initial_x.dtype)
        final_x.backward(weights)

        # final_x = (initial_x + phase) * scale * (1 - damping).
        for leaf, expected in (
            (initial_x, weights * factor),
            (phase, weights.sum() * factor),
        ):
            assert leaf.grad is not None
            assert torch.isfinite(leaf.grad).all()
            torch.testing.assert_close(leaf.grad, expected)


class TestCheetahStagedModelIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_with_parameter_sweeps(self):
        """Test running multiple parameter configurations."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])

        results = []
        for scale in [0.5, 1.0, 1.5, 2.0]:
            model.set({"scale": scale})
            particles = model.cheetah_final_particles
            results.append(particles.x.clone().detach())

        # Verify we got results for each scale
        assert len(results) == 4

    def test_workflow_with_multiple_output_accesses(self):
        """Test that final particles can be accessed multiple times consistently."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])

        model.set({"scale": 2.0})

        # Access final particles multiple times
        particles1 = model.cheetah_final_particles
        particles2 = model.cheetah_final_particles

        # Values should be identical
        assert torch.allclose(particles1.x, particles2.x)

    def test_workflow_particle_group_conversions(self):
        """Test conversions between ParticleBeam and ParticleGroup."""
        source = SourceModel()
        process = ProcessModel()
        model = CheetahStagedModel([source, process])

        model.set({"scale": 1.5})

        # Get as ParticleBeam
        beam = model.cheetah_final_particles
        assert isinstance(beam, ParticleBeam)

        # Get as ParticleGroup
        group = model.final_particles
        assert isinstance(group, ParticleGroup)

        # Verify consistency
        assert len(beam.x) == len(group.x)
