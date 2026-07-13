from lume.actions import WritableActionMixin, ReadOnlyActionMixin
from lume_torch.variables import TorchScalarVariable, TorchNDVariable
from lume.variables import EnumVariable

from lume_cheetah.simulator import CheetahSimulator

class _CheetahElementAccessMixin:
    """Shared element/energy resolution and direct attribute helpers."""

    element_name: str
    element_attribute: str

    def _resolve_element_and_energy(self, simulator):
        """Resolve target element object(s) and matching beam energy."""
        element = getattr(simulator.segment, self.element_name)
        energy = simulator.energies.get(self.element_name)
        return element, energy

    def _get_direct_attribute(self, simulator, attribute_name: str):
        """Read a direct element attribute from the first resolved element."""
        element, _ = self._resolve_element_and_energy(simulator)
        return getattr(element, attribute_name)

    def _set_direct_attribute(self, simulator, attribute_name: str, value):
        """Set a direct element attribute on all resolved split elements."""
        element, _ = self._resolve_element_and_energy(simulator)
        setattr(element, attribute_name, value)


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
        raise RuntimeError("CheetahReadOnlyActionMixin does not support setting values.")
    

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

