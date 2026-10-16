from abc import abstractmethod
from dataclasses import dataclass

import jax
from jaxtyping import Array, Float

from lalamo.module import LalamoConfig
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "GELU",
    "Activation",
    "Identity",
    "ReLU",
    "SiLU",
]


@dataclass(frozen=True)
class Activation(LalamoConfig, RegistryABC):
    @abstractmethod
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]: ...


@dataclass(frozen=True)
class SiLU(Activation):
    alpha: float = 1.0

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x * jax.nn.sigmoid(self.alpha * x)


@dataclass(frozen=True)
class GELU(Activation):
    approximate: bool = True
    # uzu's optional pinned PyTorch CPU BF16 compatibility operator.
    pytorch_bf16: bool = False

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        if self.pytorch_bf16 and x.dtype == jax.numpy.bfloat16:
            raise NotImplementedError("Pinned PyTorch BF16 GELU compatibility runs in uzu; lalamo reference forward supports F32")
        return jax.nn.gelu(x, approximate=self.approximate)


@dataclass(frozen=True)
class Identity(Activation):
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x


@dataclass(frozen=True)
class ReLU(Activation):
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jax.nn.relu(x)
