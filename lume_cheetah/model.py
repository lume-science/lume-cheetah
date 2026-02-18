from calendar import c
from lume.model import LUMEModel
from lume.variables import Variable
from lume_cheetah.simulator import CheetahSimulator
from lume_cheetah.transformer import CheetahTransformer


class LUMECheetahModel(LUMEModel):
    """
    LumeModel subclass for wrapping Cheetah Simulations

    Attributes
    ----------
    simulator : CheetahSimulator
        The CheetahSimulator instance used for simulating the accelerator behavior.
    transformer : CheetahTransformer
        The CheetahTransformer instance used for mapping between control variables and cheetah properties.

    """

    def __init__(
        self,
        simulator: CheetahSimulator,
        transformer: CheetahTransformer,
        control_variables: dict[str, Variable],
        observable_variables: dict[str, Variable],
    ):
        """
        Initialize the LUMECheetahModel.

        Parameters
        ----------
        simulator : CheetahSimulator
            The CheetahSimulator instance used for simulating the accelerator behavior.
        transformer : CheetahTransformer
            The CheetahTransformer instance used for mapping between control variables and cheetah properties.
        control_variables : dict[str, Variable]
            A dictionary mapping control variable names to Variable instances.
        observable_variables : dict[str, Variable]
            A dictionary mapping observable (read-only) variable names to Variable instances.
        """

        super().__init__()
        self.simulator = simulator
        self.transformer = transformer
        self._control_variables = control_variables
        self._observable_variables = observable_variables
        self._variables = {**control_variables, **observable_variables}
        self._state = {}


        #TODO: in _set check if pvs affect beam energy and recalc if necessary. note for self: only cavities 
        #

        # init state
        self.update_state()
        


    def _set(self, values: dict):
        """
        Internal method to set input variables and compute outputs.

        This method:
        1. Updates input variables in the state
        2. Performs calculations to update output variables
        3. Stores results in the state

        Parameters
        ----------
        values : dict[str, Any]
            Dictionary of variable names and values to set
        """
        # set the values in the simulator
        for control_name, value in values.items():
            self.transformer.set_cheetah_property(self.simulator, control_name, value)

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
        return {var: self._state[var] for var in variable_names}

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """
        Get a dictionary of all supported variables in the model.

        Returns
        -------
        dict[str, Variable]
            Dictionary mapping variable names to Variable instances
        """
        return self._variables 
    @property
    def control_variables(self) -> dict[str, Variable]:
        """
        Get a dictionary of control (input) variables in the model.

        Returns
        -------
        dict[str, Variable]
            Dictionary mapping control variable names to Variable instances
        """
        return self._control_variables

    @property
    def observable_variables(self) -> dict[str, Variable]:
        """
        Get a dictionary of observable (read-only) variables in the model.

        Returns
        -------
        dict[str, Variable]
            Dictionary mapping observable variable names to Variable instances
        """
        return self._observable_variables

    def update_state(self):
        """
        Update the model state by reading all supported variables.
        """
        # get the current state from the simulator
        for name in self.supported_variables.keys():
            self._state[name] = self.transformer.get_cheetah_property(
                self.simulator, name
            )

    def reset(self):
        """
        Reset the model to its initial state by resetting the simulator and updating the state.
        """
        self.simulator.reset()
        self.update_state()

    