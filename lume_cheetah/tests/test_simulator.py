import pytest
import torch
from cheetah.accelerator import Cavity, Drift, Quadrupole, Segment
from cheetah.particles import ParticleBeam

from lume_cheetah import simulator as simulator_module
from lume_cheetah import CheetahOverlay
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


def _make_rf_overlay_simulator(enld=100.0):
    weights = {
        "K21_1B": 0.4142135623730951,
        "K21_1C": 0.2928932188134525,
        "K21_1D": 0.2928932188134525,
    }
    segment = Segment(
        [
            Cavity(
                name=name,
                length=torch.tensor(length),
                frequency=torch.tensor(2.856e9),
            )
            for name, length in zip(weights, [2.8692, 2.8692, 3.0441], strict=True)
        ]
    )
    targets = {}
    for name, weight in weights.items():
        targets[name, "voltage"] = lambda v, weight=weight: (
            v["f"] * v["in_use"] * v["ENLD_MeV"] * 1e6 * weight
        )
        targets[name, "phase"] = lambda v: v["phase_deg"]
    overlay = CheetahOverlay(
        "K21_1",
        variables={"ENLD_MeV": enld, "phase_deg": 0.0, "f": 1.0, "in_use": 1.0},
        targets=targets,
    )
    simulator = CheetahSimulator(
        segment=segment,
        initial_beam_distribution=_make_beam(),
        overlays=[overlay],
    )
    return simulator, overlay, weights


def test_overlay_fans_out_voltage_and_phase_and_tracks():
    simulator, overlay, weights = _make_rf_overlay_simulator()
    assert simulator.overlays["K21_1"] is overlay
    for name, weight in weights.items():
        assert getattr(simulator.segment, name).voltage.item() == pytest.approx(
            100e6 * weight
        )
    assert simulator.energies["K21_1C"].item() == pytest.approx(
        1e6 + 100e6 * weights["K21_1B"]
    )

    overlay.phase_deg = 60.0
    overlay.set(ENLD_MeV=120.0, f=0.95)
    for name, weight in weights.items():
        cavity = getattr(simulator.segment, name)
        assert cavity.voltage.item() == pytest.approx(120e6 * 0.95 * weight)
        assert cavity.phase.item() == 60.0
    assert simulator.track().energy.item() == pytest.approx(1e6 + 120e6 * 0.95 * 0.5)

    overlay.in_use = 0.0
    for cavity in simulator.segment.elements:
        assert cavity.voltage.item() == 0.0
    assert simulator.track().energy.item() == pytest.approx(1e6)


def test_overlay_self_variable_can_be_updated():
    overlay = CheetahOverlay(
        "self_variable",
        variables={"self": 1.0},
        targets={("Q1", "k1"): lambda variables: variables["self"]},
    )
    simulator = CheetahSimulator(
        segment=_make_segment(),
        initial_beam_distribution=_make_beam(),
        overlays=[overlay],
    )

    overlay.self = 2.0
    assert simulator.segment.Q1.k1.item() == 2.0
    overlay.set(self=3.0)
    assert simulator.segment.Q1.k1.item() == 3.0


def test_overlay_reset_restores_variables_and_rebinds_existing_reference():
    simulator, overlay, weights = _make_rf_overlay_simulator()
    old_cavity = simulator.segment.K21_1B
    overlay.set(ENLD_MeV=20.0, phase_deg=30.0)
    simulator.reset()

    assert simulator.overlays["K21_1"] is overlay
    assert overlay.ENLD_MeV == 100.0
    assert overlay.phase_deg == 0.0
    assert simulator.segment.K21_1B is not old_cavity
    assert simulator.segment.K21_1B.voltage.item() == pytest.approx(
        100e6 * weights["K21_1B"]
    )
    overlay.phase_deg = 45.0
    assert simulator.segment.K21_1B.phase.item() == 45.0
    assert old_cavity.phase.item() == 30.0


def test_overlay_preserves_batches_dtype_and_gradients():
    enld = torch.tensor([100.0, 120.0], dtype=torch.float64, requires_grad=True)
    simulator, overlay, _ = _make_rf_overlay_simulator(enld)
    for cavity in simulator.segment.elements:
        assert cavity.voltage.shape == (2,)
        assert cavity.voltage.dtype == torch.float32
    assert simulator.get_energy()["K21_1C"].shape == (2,)
    torch.testing.assert_close(
        torch.autograd.grad(simulator.track().energy.sum(), enld)[0],
        torch.full_like(enld, 1e6),
    )

    phase = torch.tensor([30.0, 60.0], dtype=torch.float64, requires_grad=True)
    overlay.set(ENLD_MeV=100.0, phase_deg=phase)
    energy = simulator.track().energy
    torch.testing.assert_close(
        energy, (1e6 + 100e6 * torch.cos(torch.deg2rad(phase))).float()
    )
    torch.testing.assert_close(
        torch.autograd.grad(energy.sum(), phase)[0],
        -100e6 * torch.sin(torch.deg2rad(phase)) * torch.pi / 180,
        rtol=1e-6,
        atol=1e-6,
    )
    simulator.reset()
    torch.testing.assert_close(overlay.ENLD_MeV, enld)


def test_overlay_updates_repeated_names_in_nested_segments():
    first = Quadrupole(name="Q1", length=torch.tensor(0.5))
    second = Quadrupole(name="Q1", length=torch.tensor(0.5))
    segment = Segment([first, Segment([second], name="nested")])
    overlay = CheetahOverlay(
        "family",
        variables={"strength": 1.0},
        targets={("Q1", "k1"): lambda v: 2 * v["strength"]},
    )
    CheetahSimulator(
        segment=segment, initial_beam_distribution=_make_beam(), overlays=[overlay]
    )
    overlay.strength = 3.0
    assert first.k1.item() == second.k1.item() == 6.0


def test_overlay_evaluation_failure_does_not_partially_update():
    overlay = CheetahOverlay(
        "family",
        variables={"strength": 1.0},
        targets={
            ("Q1", "k1"): lambda v: 2 * v["strength"],
            ("D1", "length"): lambda v: 1 / v["strength"],
        },
    )
    simulator = CheetahSimulator(
        segment=_make_segment(),
        initial_beam_distribution=_make_beam(),
        overlays=[overlay],
    )
    with pytest.raises(ZeroDivisionError):
        overlay.strength = 0.0
    assert overlay.strength == 1.0
    assert simulator.segment.Q1.k1.item() == 2.0
    assert simulator.segment.D1.length.item() == 1.0
    with pytest.raises(AttributeError, match="no variable"):
        overlay.typo = 2.0
    with pytest.raises(TypeError, match="number or tensor"):
        overlay.strength = "bad"
    with pytest.raises(TypeError):
        overlay.variables["strength"] = 0.0


@pytest.mark.parametrize(
    "target, message",
    [
        (("missing", "k1"), "not found"),
        (("Q1", "gradient_err"), "existing tensor"),
        (("Q1", "name"), "existing tensor"),
    ],
)
def test_overlay_rejects_missing_or_non_tensor_targets(target, message):
    overlay = CheetahOverlay(
        "family", variables={"strength": 1.0}, targets={target: lambda v: v["strength"]}
    )
    with pytest.raises(ValueError, match=message):
        CheetahSimulator(
            segment=_make_segment(),
            initial_beam_distribution=_make_beam(),
            overlays=[overlay],
        )


def test_failed_overlay_binding_rolls_back_prior_bindings():
    segment = _make_segment()
    initial_k1 = segment.Q1.k1
    first_overlay = CheetahOverlay(
        "first",
        variables={"strength": 2.0},
        targets={("Q1", "k1"): lambda variables: variables["strength"]},
    )
    later_overlay = CheetahOverlay(
        "later",
        variables={"strength": 2.0},
        targets={("missing", "k1"): lambda variables: variables["strength"]},
    )

    with pytest.raises(ValueError, match="not found"):
        CheetahSimulator(
            segment=segment,
            initial_beam_distribution=_make_beam(),
            overlays=[first_overlay, later_overlay],
        )

    assert segment.Q1.k1 is initial_k1
    assert first_overlay._initial_variables is None
    assert first_overlay._targets == {}

    simulator = CheetahSimulator(
        segment=segment,
        initial_beam_distribution=_make_beam(),
        overlays=[first_overlay],
    )
    assert simulator.segment.Q1.k1.item() == 2.0


@pytest.mark.parametrize("variable", ["name", "set", "variables", "_targets", "a-b"])
def test_overlay_rejects_reserved_variable_names(variable):
    with pytest.raises(ValueError, match="Invalid or reserved"):
        CheetahOverlay(
            "family",
            variables={variable: 1.0},
            targets={("Q1", "k1"): lambda v: 1.0},
        )


@pytest.mark.parametrize("same_name", [True, False])
def test_simulator_rejects_conflicting_overlays(same_name):
    overlays = [
        CheetahOverlay(
            name,
            variables={"strength": 1.0},
            targets={("Q1", "k1"): lambda v: v["strength"]},
        )
        for name in ["family", "family" if same_name else "other"]
    ]
    with pytest.raises(ValueError, match="Duplicate overlay|Multiple overlays"):
        CheetahSimulator(
            segment=_make_segment(),
            initial_beam_distribution=_make_beam(),
            overlays=overlays,
        )


def test_overlay_cannot_be_reused_by_another_simulator():
    _, overlay, _ = _make_rf_overlay_simulator()
    with pytest.raises(ValueError, match="already bound"):
        CheetahSimulator(
            segment=_make_segment(),
            initial_beam_distribution=_make_beam(),
            overlays=[overlay],
        )
