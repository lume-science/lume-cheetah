
from lume.actions import WritableActionMixin, ReadOnlyActionMixin
from lume_torch.variables import TorchScalarVariable, TorchNDVariable
from lume.variables import EnumVariable
from lume.exceptions import ReadOnlyError

from lume_cheetah.simulator import CheetahSimulator
from cheetah.accelerator import Screen


class _CheetahElementAccessMixin:
    """Shared element/energy resolution and direct attribute helpers."""

    element_name: str
    element_attribute: str

    @staticmethod
    def _resolve_element_and_energy(simulator, element_name: str):
        """Resolve target element object(s) and matching beam energy."""
        element = getattr(simulator.segment, element_name)
        energy = simulator.energies.get(element_name)
        return element, energy

    def _get_direct_attribute(self, simulator, attribute_name: str):
        """Read a direct element attribute from the first resolved element."""
        element, _ = self._resolve_element_and_energy(simulator, self.element_name)
        return getattr(element, attribute_name)

    def _set_direct_attribute(self, simulator, attribute_name: str, value):
        """Set a direct element attribute on all resolved split elements."""
        element, _ = self._resolve_element_and_energy(simulator, self.element_name)
        setattr(element, attribute_name, value)

class _CheetahBeamAccessMixin:
    element_name: str

    def _get_read_beam(self, simulator):
        """Get the read beam for the resolved element. Currently only works for screens"""
        element = getattr(simulator.segment, self.element_name)
        if not isinstance(element, Screen):
            raise TypeError("Read beam is only available for Screen elements.")
        return element.get_read_beam()


class CheetahWritableActionMixin(_CheetahElementAccessMixin, WritableActionMixin):
    """Writable action mixin for Cheetah-backed PVs."""

    def _get(self, simulator):
        if not isinstance(simulator, CheetahSimulator):
            raise TypeError("CheetahWritableActionMixin requires a CheetahSimulator.")
        return self._get_direct_attribute(simulator, self.element_attribute)

    def _set(self, simulator, value):
        if not isinstance(simulator, CheetahSimulator):
            raise TypeError("CheetahWritableActionMixin requires a CheetahSimulator.")
        self._set_direct_attribute(simulator, self.element_attribute, value)


class CheetahReadOnlyActionMixin(CheetahWritableActionMixin, ReadOnlyActionMixin):
    """Read-only action mixin for Cheetah-backed PVs."""

    read_only = True

    def _get(self, simulator):
        if not isinstance(simulator, CheetahSimulator):
            raise TypeError("CheetahReadOnlyActionMixin requires a CheetahSimulator.")
        return self._get_direct_attribute(simulator, self.element_attribute)

    def _set(self, simulator, value):
        raise ReadOnlyError(
            "CheetahReadOnlyActionMixin does not support setting values."
        )


class CheetahWritableScalarVariable(TorchScalarVariable, CheetahWritableActionMixin):
    """Writable scalar variable for Cheetah-backed PVs."""


class CheetahReadOnlyScalarVariable(TorchScalarVariable, CheetahReadOnlyActionMixin):
    """Read-only scalar variable for Cheetah-backed PVs."""


class CheetahWritableNDVariable(TorchNDVariable, CheetahWritableActionMixin):
    """Writable N-dimensional variable for Cheetah-backed PVs."""


class CheetahReadOnlyNDVariable(TorchNDVariable, CheetahReadOnlyActionMixin):
    """Read-only N-dimensional variable for Cheetah-backed PVs."""


class CheetahReadOnlyEnumVariable(EnumVariable, CheetahReadOnlyActionMixin):
    """Read-only enum variable for Cheetah-backed PVs."""


class CheetahBeamPropertyVariable(TorchScalarVariable, _CheetahBeamAccessMixin, ReadOnlyActionMixin):
    """Read-only beam property variable for Cheetah-backed PVs."""
    beam_property: str

    def _get(self, simulator):
        read_beam = self._get_read_beam(simulator)
        return getattr(read_beam, self.beam_property)
