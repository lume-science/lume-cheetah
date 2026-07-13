from lume.actions import ActionModel
from lume.variables import Variable
from lume_cheetah.simulator import CheetahSimulator
import torch


class LUMECheetahModel(ActionModel, torch.nn.Module):
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
        super().__init__(simulator, action_variables)

        self._state = {}

        self.update_state()

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
        self.simulator.track()

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
        self.update_state()
