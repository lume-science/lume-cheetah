from torch import device
import yaml
import pprint
import copy


def load_yaml(yaml_file: str)-> dict:
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    return data

def deep_merge(a: dict, b: dict ) -> dict:
    """
    Recursively merge dictionary ``b`` into dictionary ``a``.

    For each key in ``b``:
      - If the key does not exist in ``a``, it is added.
      - If the key exists in both ``a`` and ``b`` and both values are dictionaries,
        the values are merged recursively.
      - Otherwise, the value from ``b`` replaces the value in ``a``.

    The merge is performed **in place** on ``a``.

    Parameters
    ----------
    a : dict
        Base dictionary to be updated. This dictionary is mutated in place.
    b : dict
        Dictionary whose values override or extend those in ``a``.

    Returns
    -------
    dict
        The updated dictionary ``a``.

    Notes
    -----
    - Non-dictionary values (e.g., lists, numbers, strings) are replaced, not merged.
    - Lists are not appended or merged by default.
    - No type coercion or validation is performed.
    - Later values always take precedence over earlier ones.

    Examples
    --------
    >>> base = {"a": {"x": 1, "y": 2}}
    >>> override = {"a": {"y": 3, "z": 4}}
    >>> deep_merge(base, override)
    {'a': {'x': 1, 'y': 3, 'z': 4}}
    """
        
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            deep_merge(a[k],v)
        else:
            a[k] = v
    return a

def load_relevant_controls(yaml_files: list[str]):

    data = {}
    for yaml_file in yaml_files:
        contents = load_yaml(yaml_file)
        data = deep_merge(data, contents)

    relevant_controls = {}
    for subsystem in data:
        for madname, sub_config  in data[subsystem].items():
                pvs = copy.copy(sub_config['controls_information']['PVs'])
                dev_type = sub_config['metadata']['type']
                pvs = {k.upper() : v for k,v in pvs.items()}
                relevant_controls[madname.lower()] = (dev_type, pvs)

    return relevant_controls
        




