"""Checkpoint layout adapters, independent of Hub repository names.

Registered Julia/Laya-compatible layouts reuse the modernbert_marker_v1 graph.
Different backbones or decision heads require their own graph converter; declaring
an adapter id does not make an incompatible checkpoint executable.
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import cattrs

__all__ = [
    "DEFAULT_DECISION_ADAPTERS",
    "DecisionCheckpointAdapter",
    "DecisionCheckpointLayout",
    "DecisionCheckpointRegistry",
    "DecisionCheckpointSpec",
    "JuliaCheckpointAdapter",
    "JuliaInferencePolicy",
    "LayaCheckpointAdapter",
    "LayaCheckpointConfig",
    "UnsupportedDecisionArchitectureError",
]

_converter = cattrs.Converter()


class UnsupportedDecisionArchitectureError(ValueError):
    pass


@dataclass(frozen=True)
class JuliaCheckpointConfig:
    format_version: Literal[1]
    architecture: Literal["JuliaDecisionModel"]
    head_layers: int


@dataclass(frozen=True)
class DecisionCheckpointLayout:
    julia_config_file: str = "julia_config.json"
    encoder_config_file: str = "encoder/config.json"
    weights_file: str = "model.safetensors"
    tokenizer_directory: str = "tokenizer"


@dataclass(frozen=True)
class JuliaInferencePolicy:
    max_length: int = 8192
    head_length: int = 512
    strict_encoding: bool = True


@dataclass(frozen=True)
class LayaCheckpointConfig:
    head_layers: int
    max_len: int = 512
    head_max_len: int = 192
    temperature: tuple[float, float, float] = (1.0, 1.0, 1.0)
    temperature_by_options: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointIdentity:
    architecture: str | None = None
    architectures: tuple[str, ...] = ()
    model_type: str | None = None
    julia_config_file: str | None = None


@dataclass(frozen=True)
class DecisionCheckpointSpec:
    adapter_id: str
    architecture_id: str
    family: Literal["julia", "laya"]
    layout: DecisionCheckpointLayout
    head_layers: int
    inference: JuliaInferencePolicy | LayaCheckpointConfig

    def __post_init__(self) -> None:
        if not self.adapter_id or not self.architecture_id or self.head_layers < 1:
            raise ValueError("Decision checkpoint adapter/architecture ids and head layer count must be valid")
        if (self.family == "julia") != isinstance(self.inference, JuliaInferencePolicy):
            raise ValueError("Decision checkpoint family and inference policy disagree")


def _read_config[ConfigT](path: Path, config_type: type[ConfigT]) -> ConfigT:
    return _converter.structure(json.loads(path.read_text()), config_type)


def _identity(directory: Path) -> CheckpointIdentity:
    path = directory / "config.json"
    if path.is_file():
        return _read_config(path, CheckpointIdentity)
    return CheckpointIdentity()


@runtime_checkable
class DecisionCheckpointAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    @property
    def architecture_id(self) -> str: ...

    def matches(self, directory: Path) -> bool: ...

    def load(self, directory: Path) -> DecisionCheckpointSpec: ...


@dataclass(frozen=True)
class JuliaCheckpointAdapter:
    adapter_id: str = "julia_v1"
    architecture_id: str = "modernbert_marker_v1"

    def matches(self, directory: Path) -> bool:
        identity = _identity(directory)
        return (
            (directory / "julia_config.json").is_file()
            or identity.architecture == "JuliaDecisionModel"
            or identity.julia_config_file is not None
        )

    def load(self, directory: Path) -> DecisionCheckpointSpec:
        root_config = directory / "config.json"
        layout = (
            _read_config(root_config, DecisionCheckpointLayout)
            if root_config.is_file()
            else DecisionCheckpointLayout()
        )
        checkpoint = _read_config(directory / layout.julia_config_file, JuliaCheckpointConfig)
        policy_file = directory / "inference-policy.json"
        policy = _read_config(policy_file, JuliaInferencePolicy) if policy_file.is_file() else JuliaInferencePolicy()
        return DecisionCheckpointSpec(
            self.adapter_id, self.architecture_id, "julia", layout, checkpoint.head_layers, policy
        )


@dataclass(frozen=True)
class LayaCheckpointAdapter:
    adapter_id: str = "laya_v1"
    architecture_id: str = "modernbert_marker_v1"

    def matches(self, directory: Path) -> bool:
        identity = _identity(directory)
        known_architectures = {"LayaTypedDecisions", "LayaDecisionModel"}
        return (
            (directory / "rl_agent_config.json").is_file()
            and identity.model_type in (None, "laya")
            and (not identity.architectures or set(identity.architectures) <= known_architectures)
            and identity.architecture is None
        )

    def load(self, directory: Path) -> DecisionCheckpointSpec:
        checkpoint = _read_config(directory / "rl_agent_config.json", LayaCheckpointConfig)
        return DecisionCheckpointSpec(
            self.adapter_id,
            self.architecture_id,
            "laya",
            DecisionCheckpointLayout(),
            checkpoint.head_layers,
            checkpoint,
        )


@dataclass(frozen=True)
class DecisionCheckpointRegistry:
    adapters: tuple[DecisionCheckpointAdapter, ...] = ()

    def with_adapter(self, adapter: DecisionCheckpointAdapter) -> "DecisionCheckpointRegistry":
        if any(registered.adapter_id == adapter.adapter_id for registered in self.adapters):
            raise ValueError(f"Decision adapter id already registered: {adapter.adapter_id}")
        return replace(self, adapters=(*self.adapters, adapter))

    def resolve(self, directory: Path | str) -> DecisionCheckpointSpec:
        directory = Path(directory)
        matches = tuple(adapter for adapter in self.adapters if adapter.matches(directory))
        if not matches:
            supported = ", ".join(adapter.adapter_id for adapter in self.adapters) or "none"
            raise UnsupportedDecisionArchitectureError(
                f"Unsupported decision checkpoint: {directory}. Registered adapters: {supported}. "
                "A compatible checkpoint must match an adapter layout and pass graph/weight validation.",
            )
        if len(matches) != 1:
            raise UnsupportedDecisionArchitectureError(
                f"Ambiguous decision checkpoint: {directory}; matching adapters: "
                + ", ".join(adapter.adapter_id for adapter in matches),
            )
        adapter = matches[0]
        spec = adapter.load(directory)
        if spec.adapter_id != adapter.adapter_id or spec.architecture_id != adapter.architecture_id:
            raise ValueError("Decision adapter returned inconsistent identity")
        return spec


DEFAULT_DECISION_ADAPTERS = DecisionCheckpointRegistry((JuliaCheckpointAdapter(), LayaCheckpointAdapter()))
