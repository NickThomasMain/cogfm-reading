"""A tiny name->class registry so components can be selected from config.

This is the bridge between a Hydra config value (e.g. ``encoder: scanpath``) and
the actual class that gets built. Components register themselves with a
decorator; the rest of the code builds them by name. Adding a component costs
one decorator line, with no central if/elif chain — which keeps swapping
components for ablations cheap.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    """A named lookup table mapping string names to classes."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator: register ``cls`` under ``name``."""

        def wrapper(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._entries[name] = cls
            return cls

        return wrapper

    def build(self, name: str, **kwargs):
        """Look up ``name`` and instantiate it with ``kwargs``."""
        if name not in self._entries:
            raise KeyError(f"unknown {self.kind} '{name}'. Registered: {self.available()}")
        return self._entries[name](**kwargs)

    def available(self) -> list[str]:
        """Names currently registered."""
        return sorted(self._entries)


# One registry per swappable component family.
ENCODERS = Registry("encoder")
CONNECTORS = Registry("connector")
LOSSES = Registry("loss")
ANCHORS = Registry("anchor")
