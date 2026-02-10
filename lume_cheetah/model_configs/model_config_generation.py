from ast import In
from pprint import pp, pprint
from turtle import setup
from numpy import zeros
import yaml
from simulation_server.virtual_accelerator import VirtualAccelerator
from simulation_server.factory import get_virtual_accelerator
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class InputVariable:

    default_value: float
    is_constant: bool
    value_range: list[float]
    value_range_tolerance: float
    variable_class: str

@dataclass
class OutputVariable:

    default_value: float | list[float]
    is_constant: bool
    value_range: list[float]
    value_range_tolerance: float
    variable_class: str

def setup_value_range(value, pv):
    if value[pv] >0 :
        return [value[pv]-value[pv]*0.2, value[pv]+value[pv]*0.2]
    else:
        return [value[pv]+value[pv]*0.2, value[pv]-value[pv]*0.2]



va = get_virtual_accelerator("nc_injector", monitor_overview=True, measurement_noise_level=0.01)
with open('simulation_server/yaml_configs/DL1.yaml', 'r') as file:
    config = yaml.safe_load(file)

    ignore_input_flags = [
        "BMAX",
        "BMIN",
        "BDES",
        "BCON",
        "ENB",
        "BST",
        "MODE",
        "ENABLE",
        "CTRL",
    ]

    ignore_output_flags = [

        "ArraySize0_RBV",
        "ArraySize1_RBV",
        "RESOLUTION",
        "ENB",
        "BST",
        "MODE",
        "ENABLE",
    ]

input_variables = {}
input_subsystems = {'magnets': config['magnets'], 'tcavs': config['tcavs']}
output_subsystems = {'screens': config['screens'], 'bpms': config['bpms']}
for subsystem in input_subsystems:
    
    print(f"Subsystem: {subsystem}")
    for device, fields in input_subsystems[subsystem].items():
        print(f" device {device}")
        PVs = fields.get('controls_information',{}).get('PVs', {})
        for attr, pv in PVs.items():
            if pv.rsplit(':')[-1] in ignore_input_flags:
                continue
            try:
                value = (va.get_pvs([pv])) 
                var = InputVariable(
                    default_value=value[pv],
                    is_constant= False,
                    value_range=setup_value_range(value, pv),
                    value_range_tolerance=.1,
                    variable_class= 'ScalarVariable',
                    )
                input_variables.update({pv: asdict(var)})
            except Exception as e:
                #print(f"  Could not get PVs for {attr} in device {device}: {e}")
                pass

            pprint(input_variables)



output_variables = {}
for subsystem in output_subsystems:
    print(f"Subsystem: {subsystem}")
    for device, fields in output_subsystems[subsystem].items():
        print(f" device {device}")
        PVs = fields.get('controls_information',{}).get('PVs', {})
        for attr, pv in PVs.items():
            if pv.rsplit(':')[-1] in ignore_output_flags:
                continue
            try:
                value = (va.get_pvs([pv]))
                if subsystem =='bpms':
                    var = OutputVariable(
                        default_value=value[pv],
                        is_constant= False,
                        value_range=setup_value_range(value, pv),
                        value_range_tolerance=.1,
                        variable_class= 'ScalarVariable',
                        )
                    output_variables.update({pv: asdict(var)})
                elif subsystem =='screens':
                    pass
                    #var = OutputVariable(
                    #    default_value=  0.0, #np.zeros((1024, 1024)).flatten().tolist(),
                    #    is_constant= False,
                    #    value_range= None,
                    #    value_range_tolerance= None,
                    #    variable_class= 'ArrayVariable',
                    #)
                    #output_variables.update({pv: asdict(var)})

            except Exception as e:
                #print(f"  Could not get PVs for {attr} in device {device}: {e}")
                pass 


dump_dict = {'device': 'cpu','input_transformer':' ',
            'input_variables': input_variables,
            'output_variables': output_variables}

with open('model_config_nc_injector_DL1.yaml', 'w') as outfile:
    yaml.dump(dump_dict, outfile, default_flow_style=False)