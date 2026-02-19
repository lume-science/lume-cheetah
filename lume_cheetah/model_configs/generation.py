

from ast import dump
from uritemplate import variables
import yaml
import lcls_tools.common.devices.yaml as yaml_directory
import pprint
import pathlib
import os
from lume_cheetah.model_configs.load_slac_yaml import load_relevant_controls
from cheetah.accelerator import Segment
from lume.variables import ScalarVariable
import copy

#TODO: get original mapping from control system name to madname
#TODO: get list of control system names from variable names
#TODO: get a list of cheetah element names from the segment
#TODO: find the intersection of control system names and cheetah element names
#TODO filter variables to only those that are in the intersection


FILEPATH= pathlib.Path(yaml_directory.__file__).parent.resolve()

SLAC_VARIABLE_CONFIG = {
    'BPM': {
        'X': {
            'unit': 'mm',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
        'Y': {
            'unit': 'mm',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
    },

    'QUAD': {
        'BCTRL': {
            'unit': 'kG/m',
            'read_only': False,
            'variable_class': ScalarVariable,
        },
        'BACT': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
        'BMIN': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
        'BMAX': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
    },

    'XCOR': {
        'BCTRL': {
            'unit': 'kG/m',
            'read_only': False,
            'variable_class': ScalarVariable,
        },
        'BACT': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
        'BMIN': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
        'BMAX': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
    },

    'YCOR': {
        'BCTRL': {
            'unit': 'kG/m',
            'read_only': False,
            'variable_class': ScalarVariable,
        },
        'BACT': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
        'BMIN': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
        'BMAX': {
            'unit': 'kG/m',
            'read_only': True,
            'variable_class': ScalarVariable,
        },
    },
}

SLAC_IGNORE_FLAGS = [
    "BDES",
    "BCON",
    "ENB",
    "BST",
    "MODE",
    "ENABLE",
    "CTRL",
    "ArraySize0_RBV",
    "ArraySize1_RBV",
    "RESOLUTION",
    "ENB",
    "BST",
    "MODE",
    "ENABLE",
    ]


class ModelVariableGeneration:

    def __init__(self, rel_areas, lattice: Segment):
        self.rel_areas = rel_areas
        self.lattice = lattice
        self.variable_config = SLAC_VARIABLE_CONFIG
        self.ignore_flags = SLAC_IGNORE_FLAGS
        self.devices = self.load_slac_variable_names()
        self._variables = self.generate_variables()


    def setup_value_range(self, value, pv):
        if value[pv] >0 :
            return [value[pv]-value[pv]*0.2, value[pv]+value[pv]*0.2]
        else:
            return [value[pv]+value[pv]*0.2, value[pv]-value[pv]*0.2]

    def load_slac_variable_names(self):
        filepaths = [os.path.join( FILEPATH, area) for area in self.rel_areas]
        devices = load_relevant_controls(filepaths)
        return devices
    
    def common_devices(self):
        element_names = [elem.name for elem in self.lattice.elements]
        device_names = [ device for device in self.devices if device in element_names]
        return device_names
    

    
    def create_scalar_variable(self,variable_info):
        variable_class = variable_info.pop('variable_class')
        return variable_class(**variable_info)
    
    def generate_variables(self):
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
        control_variables = {name: var for name, var in self._variables.items() if not var.read_only}
        output_variables = {name: var for name, var in self._variables.items() if var.read_only}

        return control_variables, output_variables
    
    def generate_model_config(self, output_fp):
        d = {'device': 'cpu','input_transformer':' ',
        'scalar_variables': self._variables
        }
        with open(output_fp, 'w') as outfile:
            yaml.dump(d, outfile, default_flow_style=False)

'''
dump_dict = {
        'device': 'cpu','input_transformer':' ',
        'scalar_variables': scalar_variables
        }

with open('model_config_nc_injector_DL1.yaml', 'w') as outfile:
    yaml.dump(dump_dict, outfile, default_flow_style=False)
'''
'''
    def generate_variables(self):
        scalar_variables = {}
        device_names = self.common_devices()

        for device in device_names:
            dev_type, pvs = self.devices[device]
            #print(pvs)
            #print(f"Device: {device}, Type: {dev_type}, PVs: {pvs}")
            #Device: yc10, Type: YCOR, PVs: {'BACT': 'YCOR:IN20:762:BACT', 'BCON': 'YCOR:IN20:762:BCON', 
            #'BCTRL': 'YCOR:IN20:762:BCTRL', 'BDES': 'YCOR:IN20:762:BDES', 'BMAX': 'YCOR:IN20:762:BMAX', 'BMIN': 'YCOR:IN20:762:BMIN', 'CTRL': 'YCOR:IN20:762:CTRL'}
            #Device: bpm10, Type: BPM, PVs: 
            # {'TMIT': 'BPMS:IN20:581:TMIT', 'X': 'BPMS:IN20:581:X', 'Y': 'BPMS:IN20:581:Y'}
            try: 
                dev_attr_mapping = self.variable_config[dev_type]
                #print(dev_attr_mapping)
            except KeyError as e:
                #(f"Device type {dev_type} not"
                #f"found in variable config. Skipping device {device}.")
                continue

            for pv_attr, pv in pvs.items():
                #Processing PV attribute: BACT for device xc07
                if pv_attr in self.ignore_flags:
                    continue
                try:
                    print(pv_attr)
                    variable_info = {
                    'name': pv,
                    **dev_attr_mapping[pv_attr]
                    }
                    variable = self.create_scalar_variable(variable_info)
                    scalar_variables[pv] = variable
                except Exception as e:
                    print(e)
            return scalar_variables
'''