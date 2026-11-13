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


class CheetahFinalParticlesMixIn(ABC):
    """
    Mix in to LUMEModel to indicate support for Cheetah final particles.
    """

    @property
    @abstractmethod
    def cheetah_final_particles(self) -> ParticleBeam: ...


class CheetahStagedModel(StagedModel, CheetahInitialParticlesMixIn, CheetahFinalParticlesMixIn):
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
        """ overwrite this method to validate LUMEModel instances for Cheetah support """

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
        incoming_particles = None
        for i, model in enumerate(self.lume_model_instances):
            if i > 0 and incoming_particles is not None:
                model.cheetah_initial_particles = incoming_particles

            model.set({})

            if isinstance(model, CheetahFinalParticlesMixIn):
                incoming_particles = model.cheetah_final_particles
        
        self._final_distribution = incoming_particles

    def _set(self, values: dict[str, Any]) -> None:
        """
        Set variable values across the staged models.

        Parameters
        ----------
        values: dict[str, Any]
            Variable names and values to set across the staged models.
        """
        self.coerce_evaluated_once()
        incoming_particles = None
        for i, model in enumerate(self.lume_model_instances):
            model_values = {
                k: v for k, v in values.items() if k in model.supported_variables
            }

            if i > 0 and incoming_particles is not None:
                model.cheetah_initial_particles = incoming_particles

            if model_values:
                model.set(model_values)

            if isinstance(model, CheetahFinalParticlesMixIn):
                incoming_particles = model.cheetah_final_particles
        
        self._final_distribution = incoming_particles

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

    @property
    def cheetah_final_particles(self) -> ParticleBeam:
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

