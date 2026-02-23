from lume.variables import ScalarVariable
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
