import pytest
import torch
from cheetah.accelerator import Cavity, Drift, Quadrupole, Segment
from cheetah.particles import ParticleBeam

from lume_cheetah import simulator as simulator_module
from lume_cheetah.simulator import CheetahSimulator


def _make_segment():
    return Segment(
        [
            Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
            Drift(name="D1", length=torch.tensor(1.0)),
        ]
    )


def _make_beam():
    return ParticleBeam.from_twiss(
        beta_x=torch.tensor(1.0),
        beta_y=torch.tensor(1.0),
        num_particles=512,
        energy=torch.tensor(1e6),
    )


def test_init_requires_exactly_one_initial_source():
    segment = _make_segment()
    beam = _make_beam()

    with pytest.raises(ValueError):
        CheetahSimulator(segment=segment)

    with pytest.raises(ValueError):
        CheetahSimulator(
            segment=segment,
            initial_beam_distribution=beam,
            initial_particle_group=object(),
        )


def test_init_with_particle_group_uses_converter(monkeypatch):
    expected_beam = _make_beam()
    called = {}

    def fake_converter(particle_group):
        called["particle_group"] = particle_group
        return expected_beam

    monkeypatch.setattr(
        simulator_module, "particlegroup_to_cheetah_beam", fake_converter
    )
    particle_group = object()
    simulator = CheetahSimulator(
        segment=_make_segment(), initial_particle_group=particle_group
    )

    assert called["particle_group"] is particle_group
    assert torch.equal(simulator.initial_beam_distribution.energy, expected_beam.energy)


def test_get_energy_returns_values_by_element_name():
    simulator = CheetahSimulator(
        segment=_make_segment(), initial_beam_distribution=_make_beam()
    )

    energies = simulator.get_energy()

    assert set(energies.keys()) == {"Q1", "D1"}


@pytest.mark.parametrize(
    "energy",
    [1e6, [1e6], [1e6, 2e6], [[1e6, 2e6], [3e6, 4e6]]],
)
def test_get_energy_preserves_batches_and_uses_element_entrances(energy):
    energy = torch.tensor(energy)
    segment = Segment(
        [
            Drift(name="D1", length=torch.tensor(0.1)),
            Cavity(
                name="C1",
                length=torch.tensor(0.1),
                voltage=torch.tensor(1e6),
                phase=torch.tensor(0.0),
                frequency=torch.tensor(2.856e9),
            ),
            Drift(name="D2", length=torch.tensor(0.1)),
            Cavity(
                name="C2",
                length=torch.tensor(0.1),
                voltage=torch.tensor(2e6),
                phase=torch.tensor(0.0),
                frequency=torch.tensor(2.856e9),
            ),
        ]
    )
    beam = ParticleBeam(
        particles=torch.zeros(1, 7),
        energy=energy,
        particle_charges=torch.tensor([1.0]),
    )
    simulator = CheetahSimulator(segment=segment, initial_beam_distribution=beam)
    expected = {"D1": energy, "C1": energy, "D2": energy + 1e6, "C2": energy + 1e6}

    for energies in (simulator.energies, simulator.get_energy()):
        assert energies.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(energies[name], value)

    simulator.reset()
    assert simulator.energies.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(simulator.energies[name], value)


def test_reset_restores_segment_and_beam_distribution():
    simulator = CheetahSimulator(
        segment=_make_segment(), initial_beam_distribution=_make_beam()
    )
    initial_k1 = simulator.segment.Q1.k1.clone()
    initial_charge = simulator.initial_beam_distribution_charge.clone()

    simulator.segment.Q1.k1 = torch.tensor(3.0)
    simulator.beam_distribution.particle_charges = torch.zeros_like(initial_charge)
    simulator.reset()

    assert torch.equal(simulator.segment.Q1.k1, initial_k1)
    assert torch.equal(simulator.beam_distribution.particle_charges, initial_charge)


def test_set_shutter_closes_and_reopens_beam():
    simulator = CheetahSimulator(
        segment=_make_segment(), initial_beam_distribution=_make_beam()
    )
    initial_charge = simulator.initial_beam_distribution_charge.clone()

    simulator.set_shutter(True)
    assert torch.equal(
        simulator.beam_distribution.particle_charges,
        torch.zeros_like(initial_charge),
    )

    simulator.set_shutter(False)
    assert torch.equal(simulator.beam_distribution.particle_charges, initial_charge)
