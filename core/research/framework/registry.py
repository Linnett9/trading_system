from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RegisteredComponent(Generic[T]):
    name: str
    component: T
    metadata: dict[str, Any] = field(default_factory=dict)


class ComponentRegistry(Generic[T]):
    def __init__(self) -> None:
        self._components: dict[str, RegisteredComponent[T]] = {}

    def register(
        self,
        name: str,
        component: T,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Registry component name cannot be empty")
        if normalized in self._components:
            raise ValueError(f"Component already registered: {normalized}")
        self._components[normalized] = RegisteredComponent(
            normalized,
            component,
            dict(metadata or {}),
        )

    def get(self, name: str) -> T:
        try:
            return self._components[name].component
        except KeyError as exc:
            raise KeyError(f"Unknown registered component: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._components)

    def items(self) -> Iterable[tuple[str, T]]:
        return (
            (name, registered.component)
            for name, registered in self._components.items()
        )

    def metadata(self, name: str) -> dict[str, Any]:
        if name not in self._components:
            raise KeyError(f"Unknown registered component: {name}")
        return dict(self._components[name].metadata)

    def __contains__(self, name: object) -> bool:
        return name in self._components

    def __len__(self) -> int:
        return len(self._components)


class FeatureRegistry(ComponentRegistry[T]):
    pass


class ModelRegistry(ComponentRegistry[T]):
    pass


class BenchmarkRegistry(ComponentRegistry[T]):
    pass


class ValidationRegistry(ComponentRegistry[T]):
    pass


class ReportRegistry(ComponentRegistry[T]):
    pass
