from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import importlib

from lume.actions import ActionModel
from lume.variables import Variable
from lume_cheetah.staged_model import (
    CheetahInitialParticlesMixIn,
    CheetahFinalParticlesMixIn,
)
from lume_cheetah.simulator import CheetahSimulator

from cheetah.particles import ParticleBeam
from beamphysics import ParticleGroup

import h5py
import torch
from lume_cheetah.actions import create_builtin_actions


class LUMECheetahModel(
    ActionModel,
    torch.nn.Module,
    CheetahInitialParticlesMixIn,
    CheetahFinalParticlesMixIn,
):
    """
    LumeModel subclass for wrapping Cheetah Simulations

    Attributes
    ----------
    simulator : CheetahSimulator
        The CheetahSimulator instance used for simulating the accelerator behavior.

    """

    def __init__(
        self,
        simulator: CheetahSimulator,
        action_variables: list[Variable],
    ):
        """
        Initialize the LUMECheetahModel.

        Parameters
        ----------
        simulator : CheetahSimulator
            The CheetahSimulator instance used for simulating the accelerator behavior.
        action_variables : list[Variable]
            A list of Variable instances representing the action variables.
        """
        torch.nn.Module.__init__(self)

        # Store externally provided action_variables for archiving
        self.action_variables = action_variables
        action_variables = [*action_variables, *create_builtin_actions(simulator)]

        super().__init__(simulator, action_variables)

        self._state = {}
        self._final_distribution: ParticleBeam | None = None

        self.update_state()

    def __setattr__(self, name, value):
        if name == "cheetah_initial_particles":
            # Invoke the property setter directly to bypass torch.nn.Module's __setattr__
            type(self).cheetah_initial_particles.fset(self, value)
        else:
            super().__setattr__(name, value)

    def _set(self, values: dict):
        """
        Internal method to set input variables and compute outputs.

        Parameters
        ----------
        values : dict[str, Any]
            Dictionary of variable names and values to set
        """
        # set the values in the simulator
        super()._set(values)

        # track the simulator to update the state
        self._final_distribution = self.simulator.track()

        # get the new state from the simulator
        self.update_state()

    def _get(self, variable_names: list):
        """
        Internal method to retrieve current values for specified variables.

        Parameters
        ----------
        variable_names : list[str]
            List of variable names to retrieve

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their current values
        """
        # return the requested variables from the state
        return {var: self._state[var] for var in variable_names}

    def update_state(self):
        """
        Update the model state by reading all supported variables.
        """
        # get the current state from the simulator
        for name, var in self.supported_variables.items():
            self._state[name] = var._get(self.simulator)

    def reset(self):
        """
        Reset the model to its initial state by resetting the simulator and updating the state.
        """
        self.simulator.reset()
        self._final_distribution = None
        self.update_state()

    @property
    def cheetah_initial_particles(self) -> ParticleBeam:
        return self.simulator.beam_distribution.clone()

    @cheetah_initial_particles.setter
    def cheetah_initial_particles(self, val: ParticleBeam):
        self._set_cheetah_initial_particles(val)
        self._final_distribution = self.simulator.track()
        self.update_state()

    def _set_cheetah_initial_particles(self, val: ParticleBeam) -> None:
        """Store a beam without tracking; staged execution follows with set()."""
        self.simulator.beam_distribution = val

    @property
    def cheetah_final_particles(self) -> ParticleBeam:
        if self._final_distribution is None:
            raise RuntimeError("final distribution is none")
        return self._final_distribution.clone()

    def to_archive(self, location: str | Path | h5py.Group) -> None:
        """
        Archive the current state of the model to an HDF5 file or group.

        Stores simulator (which includes initial distribution), action variable configs,
        internal state, and final distribution. Uses Pydantic's model_dump() for Variable
        serialization, which is fully reversible with model_validate().

        Parameters
        ----------
        location : str | Path | h5py.Group
            Path to an HDF5 file (created or overwritten) or an open HDF5 group
            to which the model state should be archived.
        """
        with (
            h5py.File(location, "w")
            if isinstance(location, (str, Path))
            else nullcontext(location)
        ) as archive:
            # Store archive metadata
            archive.attrs["model_class"] = (
                f"{self.__class__.__module__}.{self.__class__.__qualname__}"
            )

            # 1. Archive simulator (with full state)
            simulator_buffer = BytesIO()
            torch.save(self.simulator, simulator_buffer)
            archive.create_dataset(
                "simulator",
                data=bytearray(simulator_buffer.getvalue()),
                dtype="u1",
            )

            # 2. Store action variables using Pydantic's model_dump() and model_validate()
            # This is completely generic: works for any Pydantic Variable subclass
            if hasattr(self, "action_variables") and self.action_variables:
                variables_group = archive.create_group("variables")
                for var in self.action_variables:
                    var_group = variables_group.create_group(var.name)

                    # Store the Variable's class so it can be reimported
                    var_class = var.__class__
                    var_group.attrs["class_module"] = var_class.__module__
                    var_group.attrs["class_name"] = var_class.__name__

                    # Export Variable config using Pydantic's model_dump()
                    config = var.model_dump(exclude_none=False)

                    # Serialize config using torch.save (handles complex types)
                    config_buffer = BytesIO()
                    torch.save(config, config_buffer)
                    var_group.create_dataset(
                        "config",
                        data=bytearray(config_buffer.getvalue()),
                        dtype="u1",
                    )

            # 3. Archive internal state if available
            if hasattr(self, "_state") and self._state is not None:
                state_buffer = BytesIO()
                torch.save(self._state, state_buffer)
                archive.create_dataset(
                    "state",
                    data=bytearray(state_buffer.getvalue()),
                    dtype="u1",
                )

            # 4. Archive final distribution as OpenPMD data
            if (
                hasattr(self, "_final_distribution")
                and self._final_distribution is not None
            ):
                try:
                    particles_group = archive.create_group("particles")
                    particle_group = self._final_distribution.to_openpmd_particlegroup()
                    particle_group.write(particles_group, name="final_beam")
                except Exception:
                    # Non-fatal: beam data can be regenerated by running the model
                    pass

    @classmethod
    def from_archive(
        cls, location: str | Path | h5py.Group, model_registry: dict | None = None
    ) -> "LUMECheetahModel":
        """
        Restore a model instance from an archived HDF5 file or group.

        Reconstructs Variable instances from stored configs using Pydantic's
        model_validate(), which automatically calls model_post_init() to set up
        backend-dependent Variables.

        Parameters
        ----------
        location : str | Path | h5py.Group
            Path to an HDF5 file or an open HDF5 group containing the archived model.
        model_registry : dict, optional
            Registry of model classes for loading nested models (unused, kept for compatibility).

        Returns
        -------
        LUMECheetahModel
            The restored model instance.

        Raises
        ------
        ImportError
            If a Variable class cannot be imported.
        TypeError
            If the archive is missing required components.
        """
        with (
            h5py.File(location, "r")
            if isinstance(location, (str, Path))
            else nullcontext(location)
        ) as archive:
            # Check for old archive format (single "model" key with full pickle)
            if "model" in archive and "simulator" not in archive:
                payload = BytesIO(archive["model"][()].tobytes())
                model = torch.load(payload, map_location="cpu", weights_only=False)
                if not isinstance(model, cls):
                    raise TypeError(
                        f"The archive must contain a {cls.__name__} instance."
                    )
                return model

            # New format: separate simulator, variables, and state
            if "simulator" not in archive:
                raise TypeError(
                    f"The archive must contain a {cls.__name__} instance. "
                    "Expected 'simulator' key."
                )

            # 1. Restore simulator
            simulator_buffer = BytesIO(archive["simulator"][()].tobytes())
            simulator = torch.load(
                simulator_buffer, map_location="cpu", weights_only=False
            )

            # 2. Reconstruct action variables from stored configs using Pydantic
            action_variables = []
            variables_group = archive.get("variables")

            if variables_group is not None:
                for var_name in sorted(variables_group.keys()):
                    var_group = variables_group[var_name]

                    # Restore the variable from stored class reference and config
                    class_module = var_group.attrs["class_module"]
                    class_name = var_group.attrs["class_name"]

                    # Load config from archive
                    config_buffer = BytesIO(var_group["config"][()].tobytes())
                    config = torch.load(
                        config_buffer, map_location="cpu", weights_only=False
                    )

                    try:
                        # Import the Variable class
                        module = importlib.import_module(class_module)
                        VariableClass = getattr(module, class_name)

                        # Use Pydantic's model_validate() for all Variables
                        var = VariableClass.model_validate(config)

                        action_variables.append(var)
                    except Exception as e:
                        raise ImportError(
                            f"Cannot restore Variable '{var_name}' ({class_module}.{class_name}): {e}"
                        ) from e

            # 3. Create model with reconstructed variables and simulator
            model = cls(simulator, action_variables)

            # 4. Restore internal state if available
            if "state" in archive:
                state_buffer = BytesIO(archive["state"][()].tobytes())
                model._state = torch.load(
                    state_buffer, map_location="cpu", weights_only=False
                )

            # 6. Restore final distribution if available
            # Loads OpenPMD-compliant particle distribution and converts to Cheetah format
            # The final_beam is written to particles/final_beam by the OpenPMD writer
            # NOTE: Initial distribution restoration is handled by reconstructing the simulator
            if "particles" in archive and "final_beam" in archive["particles"]:
                try:
                    # Load ParticleGroup from OpenPMD format
                    particle_group = ParticleGroup(
                        h5=archive["particles"]["final_beam"]
                    )

                    # Convert back to Cheetah ParticleBeam
                    # Ensure energy tensor matches target dtype (beamphysics returns float64)
                    target_dtype = model.simulator.beam_distribution.particles.dtype
                    energy = torch.tensor(
                        particle_group.energy.mean(), dtype=target_dtype
                    )
                    model._final_distribution = ParticleBeam.from_openpmd_particlegroup(
                        particle_group,
                        energy=energy,
                        dtype=target_dtype,
                        device=model.simulator.beam_distribution.particles.device,
                    )
                except Exception:
                    # Non-fatal: beam data can be regenerated by running the model
                    pass

            return model
