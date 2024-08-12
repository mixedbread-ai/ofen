from __future__ import annotations

import abc
import dataclasses
import re


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


@dataclasses.dataclass
class BaseConfig(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> BaseConfig:
        raise NotImplementedError

    @classmethod
    def _from_registry(cls, model_name: str) -> BaseConfig | None:
        return None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: dict) -> BaseConfig:
        return cls(**config_dict)

    def register(self, model_name: str):
        CONFIG_REGISTRY.register(model_name, camel_to_snake(self.__class__.__name__), self.to_dict())
