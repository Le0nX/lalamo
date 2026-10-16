from typing import Literal

import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.initializer import EmptyInitializer
from lalamo.model import BaseModelConfig
from lalamo.model_import.d3 import d3_decoder_config
from lalamo.model_import.loaders.huggingface import resolve_decoder_load_layout
from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config, Qwen35RopeParameters
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.code_decision import CodeDecisionModelConfig
from lalamo.modules.normalization import UpcastMode
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.deltanet import DeltaNetConfig
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig


@pytest.mark.parametrize("head_dtype", ["float32", "bfloat16"])
def test_code_readout_preserves_directions_and_independent_precision(
    head_dtype: Literal["float32", "bfloat16"],
) -> None:
    native = HFQwen35Config(
        model_type="qwen3_5",
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        vocab_size=512,
        rope_parameters=Qwen35RopeParameters(10000.0, 0.25),
        head_dim=64,
        attention_bias=False,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        intermediate_size=256,
    )
    decoder = d3_decoder_config(native)
    delta, attention = decoder.transformer_config.layer_configs
    assert isinstance(delta.mixer_config, DeltaNetConfig)
    assert isinstance(attention.mixer_config, AttentionConfig)
    assert attention.mixer_config.is_causal is False
    assert attention.mixer_config.query_norm_config is not None
    assert attention.mixer_config.query_norm_config.upcast_mode is UpcastMode.FULL_LAYER
    ordinary = native.to_decoder_config({}).transformer_config.layer_configs[1].mixer_config
    assert isinstance(ordinary, AttentionConfig)
    assert ordinary.is_causal is True

    codec = ChatCodecConfig("{{ messages }}", None, "system", "user", "assistant", None, None)
    config = CodeDecisionModelConfig(
        token_codec_config=codec,
        architecture_id="qwen_code_readout_v1",
        decoder_config=decoder,
        readout_rows=255,
        readout_dtype=head_dtype,
        max_sequence_length=1024,
    )
    tokenizer = Tokenizer(WordLevel({"unknown": 0}, unk_token="unknown"))
    model = config.init(tokenizer, EmptyInitializer(jnp.bfloat16, ShardingConfig.replicated()))
    assert model.readout.weights.dtype == jnp.dtype(head_dtype)
    assert model.readout.weights.shape == (255, 128)
    assert model.decoder.embedding.embedding_matrix.dtype == jnp.bfloat16
    assert model.decoder.config.vocab_size == 512
    assert BaseModelConfig.from_json(config.to_json()) == config


def test_native_top_level_language_model_layout() -> None:
    weights = {"language_model.embed_tokens.weight": jnp.zeros((2, 2))}
    layout = resolve_decoder_load_layout(weights)
    assert layout.embedding_path == ParameterPath("language_model.embed_tokens")
    assert layout.decoder_path == ParameterPath("language_model")
