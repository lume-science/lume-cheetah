from abc import ABC, abstractmethod
from typing import Any

from beamphysics import ParticleGroup
import torch

from lume.staged_model import StagedModel
from lume.model import LUMEModel
from cheetah.particles import ParticleBeam


class CheetahInitialParticlesMixIn(ABC):
    """
    Mix in to LUMEModel to indicate support for Cheetah initial particles.
    """

    @property
    @abstractmethod
    def cheetah_initial_particles(self) -> ParticleBeam: ...

    @cheetah_initial_particles.setter
    @abstractmethod
    def cheetah_initial_particles(self, val: ParticleBeam): ...

    def _set_cheetah_initial_particles(self, val: ParticleBeam) -> None:
        """Stage a beam before set(); override eager setters to defer tracking."""
        self.cheetah_initial_particles = val


class CheetahFinalParticlesMixIn(ABC):
    """
    Mix in to LUMEModel to indicate support for Cheetah final particles.
    """

    @property
    @abstractmethod
    def cheetah_final_particles(self) -> ParticleBeam: ...


class CheetahStagedModel(
    StagedModel, CheetahInitialParticlesMixIn, CheetahFinalParticlesMixIn
):
    """
    Staged model that supports lume model instances that handle Cheetah initial and final
    particles such that it maintains differentiability
    """

    def __init__(self, lume_model_instances: list[LUMEModel]):
        """
        Initialize the `CheetahStagedModel` with a list of LUMEModel instances.

        Parameters
        ----------
        lume_model_instances: list[LUMEModel]
            Ordered list of LUMEModel instances to stage.
        """
        super().__init__(lume_model_instances)
        self._final_distribution = None

    @classmethod
    def validate_lume_model_instances(cls, models: list[LUMEModel]):
        """overwrite this method to validate LUMEModel instances for Cheetah support"""

        # start by validating that all models are instances of LUMEModel
        super().validate_lume_model_instances(models)

        # make sure that specific models support Cheetah initial and final particles
        for i, model in enumerate(models[:-1]):
            if not isinstance(model, CheetahFinalParticlesMixIn):
                raise TypeError(
                    f"Model at index {i} must support Cheetah final particles"
                )

        for i, model in enumerate(models[1:], start=1):
            if not isinstance(model, CheetahInitialParticlesMixIn):
                raise TypeError(
                    f"Model at index {i} must support Cheetah initial particles"
                )

    def run_start_to_end(self):
        """Track every stage once, combining beam handoff and evaluation."""
        self._propagate({}, run_all=True)

    def _set(self, values: dict[str, Any]) -> None:
        """
        Set variable values across the staged models.

        Parameters
        ----------
        values: dict[str, Any]
            Variable names and values to set across the staged models.
        """
        self._propagate(values, run_all=not self._is_init)

    def _propagate(self, values: dict[str, Any], *, run_all: bool) -> None:
        """Pass beams through the stages, batching each handoff with its controls.

        run_all forces a complete evaluation for initialization or explicit runs.
        Otherwise, stages without incoming beams or changed controls reuse results.
        """
        # A failed pass must not leave the pipeline marked as fully evaluated.
        self._is_init = False
        incoming_particles = None
        for i, model in enumerate(self.lume_model_instances):
            # Route only this stage's controls through its public set() validation.
            model_values = {
                k: v for k, v in values.items() if k in model.supported_variables
            }

            # The first stage keeps its own input beam. Later stages receive the
            # upstream result without triggering an eager evaluation at handoff.
            has_incoming = i > 0 and incoming_particles is not None
            if (
                has_incoming
                and incoming_particles is not None
                and isinstance(model, CheetahInitialParticlesMixIn)
            ):
                model._set_cheetah_initial_particles(incoming_particles)

            if run_all and isinstance(model, CheetahStagedModel):
                # A full outer run must also refresh the first nested stage.
                model._is_init = False
            # Even unchanged controls need evaluation when the input beam arrives.
            # Combining both updates here avoids tracking twice for the same stage.
            if run_all or has_incoming or model_values:
                model.set(model_values)

            # Keep the Cheetah beam (and its autograd graph) for the next stage.
            if isinstance(model, CheetahFinalParticlesMixIn):
                incoming_particles = model.cheetah_final_particles

        # Publish completion only after every required stage has succeeded.
        self._final_distribution = incoming_particles
        self._is_init = True

    @property
    def cheetah_initial_particles(self) -> ParticleBeam:
        first_model = self.lume_model_instances[0]
        if not isinstance(first_model, CheetahInitialParticlesMixIn):
            raise AttributeError(
                "Cannot access initial_particles because the first model does not implement InitialParticlesMixIn."
            )
        return first_model.cheetah_initial_particles

    @cheetah_initial_particles.setter
    def cheetah_initial_particles(self, val: ParticleBeam):
        first_model = self.lume_model_instances[0]
        if not isinstance(first_model, CheetahInitialParticlesMixIn):
            raise AttributeError(
                "Cannot set initial_particles because the first model does not implement CheetahInitialParticlesMixIn."
            )
        first_model.cheetah_initial_particles = val

    def _set_cheetah_initial_particles(self, val: ParticleBeam) -> None:
        first_model = self.lume_model_instances[0]
        if not isinstance(first_model, CheetahInitialParticlesMixIn):
            raise AttributeError(
                "Cannot set initial_particles because the first model does not implement CheetahInitialParticlesMixIn."
            )
        first_model._set_cheetah_initial_particles(val)
        self._is_init = False

    @property
    def cheetah_final_particles(self) -> ParticleBeam:
        if not isinstance(self.lume_model_instances[-1], CheetahFinalParticlesMixIn):
            raise AttributeError(
                "Cannot access cheetah_final_particles because the last model does not implement CheetahFinalParticlesMixIn."
            )
        if self._final_distribution is None:
            raise RuntimeError("final distribution is none")
        return self._final_distribution.clone()

    @property
    def initial_particles(self) -> ParticleGroup:
        return self.cheetah_initial_particles.to_openpmd_particlegroup()

    @initial_particles.setter
    def initial_particles(self, particles: ParticleGroup):
        self.cheetah_initial_particles = ParticleBeam.from_openpmd_particlegroup(
            particles, energy=torch.tensor(particles.energy.mean())
        )

    @property
    def final_particles(self) -> ParticleGroup:
        return self.cheetah_final_particles.to_openpmd_particlegroup()
