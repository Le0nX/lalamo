from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.activations import GELU
from lalamo.modules.embedding import EmbeddingBase, EmbeddingConfig, EmbeddingForwardPassConfig
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.transformer import Transformer, TransformerConfig, TransformerForwardPassConfig
from lalamo.modules.utils import call_vmapped_twice

__all__ = ["Decision", "DecisionConfig"]


@dataclass(frozen=True)
class DecisionConfig(LalamoConfig):
    embedding_config: EmbeddingConfig
    type_embedding_config: EmbeddingConfig
    embedding_norm_config: NormalizationConfig
    encoder_config: TransformerConfig
    head_config: TransformerConfig
    scorer_norm_config: NormalizationConfig
    scorer_dense_config: LinearConfig
    scorer_readout_config: LinearConfig
    vocab_size: int
    model_dim: int
    max_sequence_length: int
    max_head_tokens: int
    num_question_types: int = 3
    pytorch_bf16: bool = False

    def init(self, initializer: Initializer) -> "Decision":
        return Decision(
            config=self,
            sharding_config=initializer.sharding_config,
            embedding=self.embedding_config.init(initializer, self.model_dim, self.vocab_size),
            embedding_norm=self.embedding_norm_config.init(initializer, self.model_dim),
            encoder=self.encoder_config.init(initializer),
            type_embedding=self.type_embedding_config.init(initializer, self.model_dim, self.num_question_types),
            head=self.head_config.init(initializer),
            scorer_norm=self.scorer_norm_config.init(initializer, self.model_dim),
            scorer_dense=self.scorer_dense_config.init(
                initializer, self.model_dim, (self.model_dim,), has_biases=True
            ),
            scorer_readout=self.scorer_readout_config.init(initializer, self.model_dim, (1,), has_biases=True),
        )


class Decision(LalamoModule[DecisionConfig]):
    embedding: EmbeddingBase
    embedding_norm: Normalization
    encoder: Transformer
    type_embedding: EmbeddingBase
    head: Transformer
    scorer_norm: Normalization
    scorer_dense: Linear
    scorer_readout: Linear

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch tokens"],
        question_types: Int[Array, " batch"],
        markers: Int[Array, "batch options"],
        lengths_without_padding: Int[Array, " batch"] | None = None,
        embedding_forward_pass_config: EmbeddingForwardPassConfig = EmbeddingForwardPassConfig(),
        transformer_forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch options"]:
        if token_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("Decision input exceeds the model context window")
        embedding_key, encoder_key, type_key, head_key, dense_key, readout_key = keychain.split(6)
        positions = jnp.broadcast_to(jnp.arange(token_ids.shape[1], dtype=jnp.int32), token_ids.shape)
        features = self.embedding.embed(
            token_ids, keychain=embedding_key, forward_pass_config=embedding_forward_pass_config
        )
        if self.config.pytorch_bf16 and features.dtype == jnp.bfloat16:
            raise NotImplementedError("The pinned PyTorch BF16 compatibility profile runs in uzu; lalamo reference forward supports F32")
        features = call_vmapped_twice(self.embedding_norm, features)
        encoded = self.encoder(
            features,
            positions,
            state=None,
            return_updated_state=False,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=lengths_without_padding,
            forward_pass_config=transformer_forward_pass_config,
            keychain=encoder_key,
        ).outputs
        type_features = self.type_embedding.embed(
            question_types,
            keychain=type_key,
            forward_pass_config=embedding_forward_pass_config,
        )
        head_result = self.head(
            encoded + type_features[:, None, :],
            positions,
            state=None,
            return_updated_state=False,
            return_layer_results=True,
            return_positional_embeddings=False,
            lengths_without_padding=lengths_without_padding,
            forward_pass_config=transformer_forward_pass_config,
            keychain=head_key,
        )
        # The marker head has no final normalization; the Transformer container does.
        assert head_result.layer_results is not None
        head_features = head_result.layer_results[-1].outputs
        selected = jnp.take_along_axis(head_features, markers[:, :, None], axis=1)
        selected = call_vmapped_twice(self.scorer_norm, selected)
        (dense,) = call_vmapped_twice(self.scorer_dense, selected, keychain=dense_key)
        (scores,) = call_vmapped_twice(self.scorer_readout, GELU(approximate=False)(dense), keychain=readout_key)
        return scores.squeeze(-1).astype(jnp.float32)
