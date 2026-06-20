from lume.actions import WritableActionMixin, ReadOnlyActionMixin


def _get_variable_value(simulator, variable):
    """Helper function to get the value of a variable from the simulator."""
    element = getattr(simulator.segment, variable.element_name)
    return getattr(element, variable.attribute_name)

class CheetahReadonlyActionMixin(ReadOnlyActionMixin):
    """Base class for scalar variables in the Cheetah accelerator model."""
    element_name: str
    attribute_name: str

    def _get(self, simulator):
        """Get the current value of the variable from the simulator."""
        return _get_variable_value(simulator, self)


class CheetahWritableActionMixin(WritableActionMixin):
    """Base class for scalar variables in the Cheetah accelerator model."""
    element_name: str
    attribute_name: str

    def _set(self, simulator, value):
        """Set the value of the variable in the simulator."""
        element = getattr(simulator.segment, self.element_name)
        setattr(element, self.attribute_name, value)

    def _get(self, simulator):
        """Get the current value of the variable from the simulator."""
        return _get_variable_value(simulator, self)


