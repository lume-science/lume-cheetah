import pandas as pd 
from cheetah.accelerator import Segment
import torch


class NoSetMethodError(Exception):
    pass

class AttributeAccessor:
    """
    A class to access and set arbitrary attributes of Cheetah elements when logic
    is more complex than simple attribute access (ie. nested attributes).
    This class is used to map process variable (PV) attributes to Cheetah element attributes.
    It allows both getting and setting values of the attributes by providing a getter and setter function.
    """

    def __init__(self, getter, setter=None):
        self.get = getter
        self.set = setter

    def __call__(self, element, energy, value=None):
        if value is None:
            return self.get(element, energy)
        else:
            if self.set is None:
                raise NoSetMethodError(f"Cannot set value for this attribute")
            self.set(element, energy, value)


# -- include conversions for cheetah attributes to SLAC EPICS attributes
QUADRUPOLE_ATTR_NAME_MAPPING = {
    "BCTRL": 'k1',
    "BACT": 'k1',
}
SOLENOID_ATTR_NAME_MAPPING = {   
    "BCTRL": 'k1',
    "BACT": 'k1',
}

CORRECTOR_ATTR_NAME_MAPPING = {
    "BCTRL": 'k1',
    "BACT": 'k1',
}

TRANSVERSE_DEFLECTING_CAVITY_ATTR_NAME_MAPPING = {
    "AREQ": "voltage",
    "PREQ": "phase",
}

BPM_ATTR_NAME_MAPPING = {
    "X": "x",
    "Y": "y",
}

# multiply image intensity by 16 bit number range (is similar to real machine?)
SCREEN_MAPPING = {
    "Image:ArrayData": "reading", #reading.T * 65535)
    "PNEUMATIC": "is_active",
    "Image:ArraySize1_RBV": "resolution[0]",
    "Image:ArraySize0_RBV": "resolution[1]",
    "RESOLUTION": "pixel_size",
    "IMAGE": "reading.T * 65535",
    "N_OF_ROW": "resolution[0]",
    "N_OF_COL": "resolution[1]",
}

ATTRIBUTE_MAPPINGS = {
    "Quadrupole": QUADRUPOLE_ATTR_NAME_MAPPING,
    "Solenoid": SOLENOID_ATTR_NAME_MAPPING,
    "HorizontalCorrector": CORRECTOR_ATTR_NAME_MAPPING,
    "VerticalCorrector": CORRECTOR_ATTR_NAME_MAPPING,
    "BPM": BPM_ATTR_NAME_MAPPING,
    "Screen": SCREEN_MAPPING,}

def get_control_mad_mapping(fname):
    """
    Create a mapping from control system names to element names from a CSV file.

    Args:
        fname (str): Path to the CSV file containing the mapping.

    """
    print(fname)
    mapping = (
        pd.read_csv(fname, dtype=str)
        .set_index("Control System Name")["Element"]
        .T.to_dict()
    )
    return mapping


def get_control_attr_mapping(control_variable_names: list[str], lattice: Segment, mad_mapping:dict[str,str]):
    """
    Get the mapping for a specific control system name and attribute.

    Args:
        control_name (str): The name of the control system element.
        attr (str): The attribute to map.
    """
    control_to_cheetah = {}
    elements = {element.name: element for element in lattice.elements}
    for control_variable in control_variable_names:
        #doesn't handle screens yet
        control_name, attr = control_variable.rsplit(':',1)
        #print(control_name, attr)
        if control_name in mad_mapping:
            mad_name = mad_mapping[control_name].lower()
            #print(mad_name)
            if mad_name in elements:
                cheetah_attr = cheetah_attribute_mapping(elements[mad_name],attr)
                control_to_cheetah[control_variable] = f'{mad_name} {cheetah_attr}'
            else:
                #print('flag2')
                control_to_cheetah[control_variable] = None
        else:
            #print('flag1')
            control_to_cheetah[control_variable] = None       

    return control_to_cheetah
def cheetah_attribute_mapping(element, control_attr):
    """

    Return or set a Cheetah element attribute based on the PV attribute.
    If `set_value` is provided, it sets the value of the Cheetah attribute.

    Args:
        element (Element): The name of the Cheetah element.
        pv_attribute (str): The process variable attribute to map.
        energy (float): The beam energy in eV.
        set_value (optional): If provided, sets the value of the Cheetah attribute.

    Returns:
        value: The corresponding Cheetah attribute value if `set_value` is None, otherwise sets the value and returns None.
    """

    element_type = type(element).__name__
    if element_type not in ATTRIBUTE_MAPPINGS:
        print(f"Unsupported element type: {element_type}")
        return

    mapping = ATTRIBUTE_MAPPINGS[element_type]
    if control_attr not in mapping:
        print(
            f"Unsupported attribute: {control_attr} for element type: {element_type}"
        )
        return

    cheetah_attr = mapping[control_attr]
    return cheetah_attr


def get_mappings(fname, control_variables: dict, lattice: Segment):
    cm_mapping = get_control_mad_mapping(fname)
    control_variable_names = [cv for cv in control_variables]
    #print(f'control variable names : {control_variable_names}')
    control_to_cheetah = get_control_attr_mapping(control_variable_names, lattice, cm_mapping)
    return control_to_cheetah
    

