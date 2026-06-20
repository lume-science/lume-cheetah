import torch
from copy import deepcopy
from cheetah.accelerator import Segment
from cheetah.particles import Beam, ParticleBeam
from beamphysics import ParticleGroup
from lume_cheetah.utils import particlegroup_to_cheetah_beam

class CheetahSimulator(torch.nn.Module):
    """
    Simulator class for Cheetah accelerator simulations.

    This class provides an interface to simulate the behavior
    of a particle beam as it travels through a Cheetah accelerator segment.
    It allows for tracking the beam, retrieving energy profiles, and
    controlling the beam shutter state.

    Attributes
    ----------
    segment : Segment
        The Cheetah Segment representing the accelerator configuration.
    initial_beam_distribution : Beam
        The initial beam distribution to be tracked through the segment.
    initial_beam_distribution_charge : torch.Tensor
        A copy of the initial per-particle charges used to restore the beam
        after toggling the shutter.

    Methods
    -------
    reset()
        Resets the simulator to its initial state.
    track()
        Tracks the beam through the segment, updating the internal state.
    get_energy()
        Retrieves the energy of the beam at every element in the segment.
    set_shutter(value: bool)
        Sets the beam shutter state, controlling whether the beam is present or not.
    """

    def __init__(
        self, *,
        segment: Segment,
        initial_beam_distribution: Beam | None = None,
        initial_particle_group: ParticleGroup | None = None

    ) -> None:
        """
        Simulator class for Cheetah accelerator simulations.

        Parameters
        ----------
        segment : Segment
            The Cheetah Segment representing the accelerator configuration.
        initial_beam_distribution : Beam, optional
            The initial beam distribution to be tracked through the segment.
        initial_particle_group : ParticleGroup, optional
            An openPMD beamphysics ParticleGroup that will be converted into a
            Cheetah ParticleBeam. Must be provided if `initial_beam_distribution`
            is not.
        """
        super().__init__()

        self.segment = segment
        self._initial_segment = deepcopy(segment)


        if initial_beam_distribution and not initial_particle_group:
            self.initial_beam_distribution = initial_beam_distribution.clone()
        elif initial_particle_group and not initial_beam_distribution:
            self.initial_beam_distribution = particlegroup_to_cheetah_beam(
            initial_particle_group)
        else:
            raise ValueError("""Must provide either initial_beam_distribution"""
            """or initial_particle_group.""")

        self.initial_beam_distribution_charge = (
            self.initial_beam_distribution.particle_charges.clone()
        )

        self.beam_distribution = self.initial_beam_distribution.clone()

        self.track()
        self.energies = self.get_energy()

    def reset(self):
        self.segment = deepcopy(self._initial_segment)
        self.beam_distribution = self.initial_beam_distribution.clone()
        self.track()
        self.energies = self.get_energy()

    def track(self):
        self.segment.track(self.beam_distribution)

    def get_energy(self):
        """
        Get the energy of the beam in the virtual accelerator simulator at
        every element for use in calculating the magnetic rigidity.

        Note: need to track on a copy of the segment to not influence readings!
        """
        test_beam = ParticleBeam(
            torch.zeros(1, 7), energy=self.beam_distribution.energy
        )
        test_segment = deepcopy(self.segment)
        element_names = [e.name for e in test_segment.elements]
        return dict(
            zip(
                element_names,
                test_segment.get_beam_attrs_along_segment(("energy",), test_beam)[0],
            )
        )

    def set_shutter(self, value: bool):
        """
        Set the beam shutter state in the virtual accelerator simulator.
        If `value` is True, the shutter is closed (no beam), otherwise it is open (beam present).
        """
        if value:
            self.beam_distribution.particle_charges = torch.zeros_like(
                self.beam_distribution.particle_charges
            )
        else:
            self.beam_distribution.particle_charges = (
                self.initial_beam_distribution_charge.clone()
            )
