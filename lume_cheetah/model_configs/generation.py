"""
SLAC Model Variable Generation Utilities.

This module constructs LUME-compatible `ScalarVariable` objects from SLAC
control-system YAML definitions and a Cheetah lattice.

It performs the following steps:

1. Loads SLAC YAML device definitions using `load_relevant_controls`.
2. Filters devices to those present in a provided `Segment` lattice.
3. Maps device PV attributes to structured variable definitions using
   `SLAC_VARIABLE_CONFIG`.
4. Generates LUME `ScalarVariable` objects for:
   - Control variables (writeable)
   - Output variables (read-only)

The resulting variables can be exported into a model configuration YAML
file suitable for use with LUME / Cheetah accelerator workflows.

Typical Usage
-------------
>>> gen = ModelVariableGeneration(rel_areas=["IN20.yaml"], lattice=segment)
>>> control_vars, output_vars = gen.variables
>>> gen.generate_model_config("model_config.yaml")
"""

import yaml
import pathlib
import os
import lcls_tools.common.devices.yaml as yaml_directory
from cheetah.accelerator import Segment
from lume_cheetah.model_configs.load_slac_yaml import load_relevant_controls
from lume_cheetah.model_configs.slac_variable_configs import SLAC_IGNORE_FLAGS, SLAC_VARIABLE_CONFIG

FILEPATH= pathlib.Path(yaml_directory.__file__).parent.resolve()

class ModelVariableGeneration:
    """
    Generate LUME scalar variables from SLAC YAML control definitions.

    This class bridges SLAC control-system YAML files and a Cheetah
    accelerator lattice (`Segment`) to produce structured LUME
    `ScalarVariable` objects.

    Parameters
    ----------
    rel_areas : list[str]
        List of YAML filenames (relative to SLAC device YAML directory)
        defining control-system devices to include.
    lattice : Segment
        Cheetah accelerator segment. Only devices that appear in the
        lattice will be converted into variables.

    Attributes
    ----------
    rel_areas : list[str]
        YAML areas requested.
    lattice : Segment
        Accelerator lattice segment.
    variable_config : dict
        Mapping of device type → PV attribute configuration.
    ignore_flags : list[str]
        PV attributes that should be ignored.
    devices : dict
        Mapping of device name → (device_type, PV dictionary).
    _variables : dict[str, ScalarVariable]
        Generated scalar variables keyed by PV name.

    Notes
    -----
    - Device names are matched against lattice element names.
    - Only device types defined in `SLAC_VARIABLE_CONFIG` are processed.
    - Read-only flags are determined by configuration.
    """

    def __init__(self, rel_areas, lattice: Segment):
        self.rel_areas = rel_areas
        self.lattice = lattice
        self.variable_config = SLAC_VARIABLE_CONFIG
        self.ignore_flags = SLAC_IGNORE_FLAGS
        self.devices = self.load_slac_variable_names()
        self._variables = self.generate_variables()

    def load_slac_variable_names(self):
        """
        Load SLAC control-system device definitions from YAML files.

        Constructs absolute filepaths using the SLAC YAML directory and
        requested relative area names, then loads and merges them using
        `load_relevant_controls`.

        Returns
        -------
        dict
            Dictionary mapping device names (lowercase) to:
                (device_type, PV dictionary)
        """
        filepaths = [os.path.join( FILEPATH, area) for area in self.rel_areas]
        devices = load_relevant_controls(filepaths)
        return devices
    
    def common_devices(self):
        """
        Identify devices present in both YAML definitions and lattice.

        Returns
        -------
        list[str]
            Device names that exist in:
            - Loaded SLAC YAML device definitions
            - The provided Cheetah lattice segment
        """
        element_names = [elem.name for elem in self.lattice.elements]
        device_names = [ device for device in self.devices if device in element_names]
        return device_names
    
    def create_scalar_variable(self, variable_info):
        """
        Instantiate a ScalarVariable from configuration metadata.

        Parameters
        ----------
        variable_info : dict
            Dictionary containing:
                - name : str
                - unit : str
                - read_only : bool
                - variable_class : type

        Returns
        -------
        ScalarVariable
            Instantiated LUME scalar variable.

        Notes
        -----
        The key `variable_class` is removed from the dictionary and used
        to instantiate the variable dynamically.
        """
        variable_class = variable_info.pop('variable_class')
        return variable_class(**variable_info)
    
    def generate_variables(self):
        """
        Generate scalar variables from device definitions and lattice.

        For each device present in both YAML and lattice:
        - Look up device type configuration.
        - Ignore PV attributes listed in `ignore_flags`.
        - Create `ScalarVariable` objects using configured metadata.

        Returns
        -------
        dict[str, ScalarVariable]
            Dictionary mapping PV names to instantiated ScalarVariable objects.

        Notes
        -----
        - Devices with unknown types are skipped.
        - Exceptions during variable creation are printed but do not stop execution.
        - PV attribute names are expected to match keys in `SLAC_VARIABLE_CONFIG`.
        """
        scalar_variables = {}
        device_names = self.common_devices()

        for device in device_names:
            dev_type, pvs = self.devices[device]

            try:
                dev_attr_mapping = self.variable_config[dev_type]
            except KeyError:
                continue

            for pv_attr, pv in pvs.items():
                if pv_attr in self.ignore_flags:
                    continue
                try:
                    variable_info = {
                        'name': pv,
                        **dev_attr_mapping[pv_attr],
                    }
                    variable = self.create_scalar_variable(variable_info)
                    scalar_variables[pv] = variable
                except Exception as e:
                    print(f"{device=} {dev_type=} {pv_attr=} {pv=} -> {e}")

        return scalar_variables

    @property
    def variables(self):
        """
        Separate generated variables into control and output variables.

        Returns
        -------
        tuple[dict, dict]
            (control_variables, output_variables)

            control_variables :
                Variables where `read_only == False`

            output_variables :
                Variables where `read_only == True`
        """
        control_variables = {name: var for name, var in self._variables.items() if not var.read_only}
        output_variables = {name: var for name, var in self._variables.items() if var.read_only}

        return control_variables, output_variables
    
    def generate_model_config(self, output_fp):
        d = {'device': 'cpu','input_transformer':' ',
        'scalar_variables': self._variables
        }
        with open(output_fp, 'w') as outfile:
            yaml.dump(d, outfile, default_flow_style=False)



#TODO: get original mapping from control system name to madname
#TODO: get list of control system names from variable names
#TODO: get a list of cheetah element names from the segment
#TODO: find the intersection of control system names and cheetah element names
#TODO filter variables to only those that are in the intersection
