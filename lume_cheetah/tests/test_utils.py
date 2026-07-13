from types import SimpleNamespace

import torch

from lume_cheetah import utils


def test_particlegroup_to_cheetah_beam_passes_expected_arguments(monkeypatch):
    calls = {}
    returned_beam = SimpleNamespace(particle_charges=None)

    def fake_from_openpmd_particlegroup(*, particle_group, energy, dtype, device):
        calls["particle_group"] = particle_group
        calls["energy"] = energy
        calls["dtype"] = dtype
        calls["device"] = device
        return returned_beam

    monkeypatch.setattr(
        utils.cheetah.ParticleBeam,
        "from_openpmd_particlegroup",
        fake_from_openpmd_particlegroup,
    )

    particle_group = {"energy": torch.tensor([2.0, 4.0, 6.0])}
    beam = utils.particlegroup_to_cheetah_beam(
        particle_group,
        dtype=torch.float64,
        device="cpu",
    )

    assert beam is returned_beam
    assert calls["particle_group"] is particle_group
    assert torch.equal(calls["energy"], torch.tensor(4.0, dtype=torch.float64))
    assert calls["dtype"] == torch.float64
    assert calls["device"] == "cpu"
    assert torch.equal(beam.particle_charges, torch.tensor(1.0, dtype=torch.float64, device="cpu"))