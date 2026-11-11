from collections.abc import Callable, Mapping
from keyword import iskeyword
from types import MappingProxyType

import torch
from cheetah.accelerator import Element, Segment


OverlayValue = float | torch.Tensor
OverlayRule = Callable[[Mapping[str, OverlayValue]], OverlayValue]


class CheetahOverlay:
    """Fan out named variables to element attributes using callable rules.

    ``targets`` maps ``(element_name, attribute_name)`` pairs to pure functions
    of the variable mapping. All elements with a matching name are updated,
    including elements in nested segments. Targets must be tensor attributes,
    not registered parameters; the overlay variables control their values.

    Pass an overlay to ``CheetahSimulator(overlays=[...])`` to bind it. Assign
    variables as attributes or use ``set`` to update several together. Rules
    are evaluated before any target is written. Results use the target's dtype
    and device without detaching gradients. Reassign variables rather than
    mutating their tensors in place, which does not trigger rule evaluation.
    """

    def __init__(
        self,
        name: str,
        *,
        variables: Mapping[str, OverlayValue],
        targets: Mapping[tuple[str, str], OverlayRule],
    ) -> None:
        if not name:
            raise ValueError("An overlay must have a nonempty name.")
        if not variables or not targets:
            raise ValueError("An overlay requires variables and targets.")
        for variable, value in variables.items():
            if (
                not variable.isidentifier()
                or iskeyword(variable)
                or variable.startswith("_")
                or hasattr(type(self), variable)
            ):
                raise ValueError(f"Invalid or reserved overlay variable: {variable!r}")
            self._validate_value(variable, value)
        for target, rule in targets.items():
            if (
                not isinstance(target, tuple)
                or len(target) != 2
                or not all(isinstance(part, str) and part for part in target)
            ):
                raise ValueError(
                    f"Expected (element, attribute) target, got {target!r}"
                )
            if not callable(rule):
                raise TypeError(f"Overlay rule for {target!r} must be callable.")

        self._name = name
        self._variables = dict(variables)
        self._rules = dict(targets)
        self._targets: dict[tuple[str, str], list[Element]] = {}
        self._initial_variables: dict[str, OverlayValue] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def variables(self) -> Mapping[str, OverlayValue]:
        return MappingProxyType(self._variables)

    def __getattr__(self, name: str) -> OverlayValue:
        variables = vars(self).get("_variables", {})
        if name in variables:
            return variables[name]
        raise AttributeError(f"Overlay has no variable {name!r}.")

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self.set(**{name: self._validate_value(name, value)})

    @staticmethod
    def _validate_value(name: str, value: object) -> OverlayValue:
        if not isinstance(value, (int, float, torch.Tensor)):
            raise TypeError(f"Overlay variable {name!r} must be a number or tensor.")
        return value

    def set(self, **values: OverlayValue) -> None:
        """Update variables together and immediately recompute all targets."""
        for name, value in values.items():
            if name not in self._variables:
                raise AttributeError(f"Overlay {self.name!r} has no variable {name!r}.")
            self._validate_value(name, value)
        variables = self._variables | values
        self._apply(variables, self._targets)
        self._variables = variables

    def _apply(
        self,
        variables: Mapping[str, OverlayValue],
        targets: Mapping[tuple[str, str], list[Element]],
    ) -> None:
        updates = []
        for target, elements in targets.items():
            value = self._rules[target](MappingProxyType(dict(variables)))
            self._validate_value(str(target), value)
            attribute = target[1]
            for element in elements:
                current = getattr(element, attribute)
                result = torch.as_tensor(
                    value, dtype=current.dtype, device=current.device
                ).clone()
                updates.append((element, attribute, result))
        for element, attribute, result in updates:
            setattr(element, attribute, result)

    def _bind(self, segment: Segment, *, reset: bool = False) -> None:
        if self._initial_variables is not None and not reset:
            raise ValueError(f"Overlay {self.name!r} is already bound to a simulator.")
        elements = segment.flattened().elements
        targets = {}
        for element_name, attribute in self._rules:
            matches = [element for element in elements if element.name == element_name]
            if not matches:
                raise ValueError(f"Overlay target element {element_name!r} not found.")
            for element in matches:
                if not isinstance(getattr(element, attribute, None), torch.Tensor):
                    raise ValueError(
                        f"Overlay target {element_name}[{attribute}] "
                        "must be an existing tensor attribute."
                    )
                if attribute in dict(element.named_parameters(recurse=False)):
                    raise ValueError(
                        f"Overlay target {element_name}[{attribute}] is a registered "
                        "parameter; use a tensor buffer controlled by the overlay."
                    )
            targets[element_name, attribute] = matches

        if reset:
            if self._initial_variables is None:
                raise RuntimeError("Cannot reset an unbound overlay.")
            variables = self._copy_values(self._initial_variables)
        else:
            variables = dict(self._variables)
        self._apply(variables, targets)
        if not reset:
            self._initial_variables = self._copy_values(variables)
        self._variables = variables
        self._targets = targets

    @staticmethod
    def _copy_values(values: Mapping[str, OverlayValue]) -> dict[str, OverlayValue]:
        return {
            name: (
                value.detach().clone().requires_grad_(value.requires_grad)
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in values.items()
        }
