import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import cattrs
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.decision_adapters import (
    DEFAULT_DECISION_ADAPTERS,
    DecisionCheckpointRegistry,
    JuliaInferencePolicy,
    LayaCheckpointConfig,
    UnsupportedDecisionArchitectureError,
)
from lalamo.model_import.loaders.huggingface import load_linear, load_tied_embedding
from lalamo.model_import.model_configs.huggingface.modern_bert import ModernBERTConfig
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.decision import DecisionModel, DecisionModelConfig
from lalamo.modules.activations import ReLU
from lalamo.modules.decision import DecisionConfig
from lalamo.modules.embedding import TiedEmbedding
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLP, FeedForwardMLP, FeedForwardMLPConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayer, TransformerLayerConfig
from lalamo.safetensors import safe_read
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig

__all__ = ["DecisionPolicy", "export_decision_model", "import_decision_model"]

_converter = cattrs.Converter()


@dataclass(frozen=True)
class TokenizerSpecialToken:
    content: str


@dataclass(frozen=True)
class DecisionTokenizerConfig:
    cls_token: str | TokenizerSpecialToken
    sep_token: str | TokenizerSpecialToken
    mask_token: str | TokenizerSpecialToken
    pad_token: str | TokenizerSpecialToken


_converter.register_structure_hook(
    str | TokenizerSpecialToken,
    lambda value, _: value if isinstance(value, str) else _converter.structure(value, TokenizerSpecialToken),
)


@dataclass(frozen=True)
class DecisionPolicy:
    family: Literal["julia", "laya"]
    max_length: int
    head_length: int
    strict_encoding: bool
    # Raw checkpoint calibration; native Laya serving clamps the selected temperature to [0.5, 5].
    temperatures: tuple[float, float, float]
    temperatures_by_options: dict[str, float]
    cls_token_id: int
    sep_token_id: int
    mask_token_id: int
    pad_token_id: int
    mask_token: str
    architecture_id: str = "modernbert_marker_v1"
    adapter_id: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.head_length <= self.max_length:
            raise ValueError("Decision token budgets must be positive and head_length <= max_length")
        values = (*self.temperatures, *self.temperatures_by_options.values())
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Decision temperatures must be finite and positive")


def _read_config[ConfigT](path: Path, config_type: type[ConfigT]) -> ConfigT:
    return _converter.structure(json.loads(path.read_text()), config_type)


def _normalization(module: Normalization, weights: Mapping[str, Array], path: str) -> Normalization:
    biases = weights[f"{path}.bias"].astype(jnp.float32) if module.config.has_biases else None
    return eqx.tree_at(
        lambda m: (m.scales, m.biases),
        module,
        (weights[f"{path}.weight"].astype(jnp.float32), biases),
        is_leaf=lambda x: x is None,
    )


def _load_layer(module: TransformerLayer, weights: Mapping[str, Array], path: str, *, head: bool) -> TransformerLayer:
    assert isinstance(module.mixer, Attention)
    attention_path = f"{path}.self_attn" if head else f"{path}.attn"
    linear_weights = weights
    if head:
        linear_weights = {
            **weights,
            f"{attention_path}.in_proj.weight": weights[f"{attention_path}.in_proj_weight"],
            f"{attention_path}.in_proj.bias": weights[f"{attention_path}.in_proj_bias"],
        }
    qkv = load_linear(
        module.mixer.qkvg_projection, linear_weights, ParameterPath(attention_path) / ("in_proj" if head else "Wqkv")
    )
    out = load_linear(
        module.mixer.out_projection, weights, ParameterPath(attention_path) / ("out_proj" if head else "Wo")
    )
    attention = eqx.tree_at(lambda m: (m.qkvg_projection, m.out_projection), module.mixer, (qkv, out))
    pre_norm = None
    if module.pre_mixer_norm is not None:
        pre_norm = _normalization(module.pre_mixer_norm, weights, f"{path}.{'norm1' if head else 'attn_norm'}")
    assert module.pre_mlp_norm is not None
    mlp_norm = _normalization(module.pre_mlp_norm, weights, f"{path}.{'norm2' if head else 'mlp_norm'}")
    assert isinstance(module.mlp, DenseMLP | FeedForwardMLP)
    if head:
        up = load_linear(module.mlp.up_projection, weights, ParameterPath(path) / "linear1")
    else:
        weight = weights[f"{path}.mlp.Wi.weight"]
        gate, value = jnp.split(weight, 2, axis=0)
        renamed = {f"{path}.mlp.Wi.weight": jnp.concatenate((value, gate), axis=0)}
        up = load_linear(module.mlp.up_projection, renamed, ParameterPath(path) / "mlp" / "Wi")
    down = load_linear(module.mlp.down_projection, weights, ParameterPath(path) / ("linear2" if head else "mlp.Wo"))
    mlp = eqx.tree_at(lambda m: (m.up_projection, m.down_projection), module.mlp, (up, down))
    return eqx.tree_at(
        lambda m: (m.mixer, m.pre_mixer_norm, m.pre_mlp_norm, m.mlp),
        module,
        (attention, pre_norm, mlp_norm, mlp),
        is_leaf=lambda x: x is None,
    )


def _head_config(model_dim: int, layers: int) -> TransformerConfig:
    if layers < 1 or model_dim % max(1, model_dim // 64):
        raise ValueError("Unsupported decision head dimensions")
    norm = NormalizationConfig(1e-5, None, UpcastMode.FULL_LAYER, subtract_mean=True, has_biases=True)
    linear = LinearConfig()
    attention = AttentionConfig(
        qkvg_projection_config=linear,
        out_projection_config=linear,
        query_norm_config=None,
        key_norm_config=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkvg_biases=True,
        has_out_biases=True,
        num_heads=max(1, model_dim // 64),
        num_groups=max(1, model_dim // 64),
        head_dim=model_dim // max(1, model_dim // 64),
        scale=None,
        is_causal=False,
        sliding_window_size=None,
    )
    layer = TransformerLayerConfig(
        pre_mixer_norm_config=norm,
        mixer_config=attention,
        post_mixer_norm_config=None,
        pre_mlp_norm_config=norm,
        mlp_config=FeedForwardMLPConfig(linear, ReLU(), has_up_biases=True, has_down_biases=True),
        post_mlp_norm_config=None,
        rope_config=None,
    )
    return TransformerConfig(
        layer_configs=(layer,) * layers,
        output_norm_config=NormalizationConfig(
            1e-5, None, UpcastMode.ONLY_NORMALIZATION, subtract_mean=True, has_scale=False
        ),
        model_dim=model_dim,
        hidden_dim=4 * model_dim,
    )


def _policy(
    folder: Path,
    tokenizer: Tokenizer,
    family: Literal["julia", "laya"],
    inference: JuliaInferencePolicy | LayaCheckpointConfig,
) -> DecisionPolicy:
    special = _read_config(folder / "tokenizer_config.json", DecisionTokenizerConfig)
    tokens = tuple(
        value if isinstance(value, str) else value.content
        for value in (special.cls_token, special.sep_token, special.mask_token, special.pad_token)
    )
    ids = tuple(tokenizer.token_to_id(token) for token in tokens)
    if any(token_id is None for token_id in ids):
        raise ValueError("Decision tokenizer is missing a required special token")
    cls_id, sep_id, mask_id, pad_id = ids
    assert cls_id is not None and sep_id is not None and mask_id is not None and pad_id is not None
    if isinstance(inference, JuliaInferencePolicy):
        length, head_length, strict = inference.max_length, inference.head_length, inference.strict_encoding
        temperatures, by_options = (1.0, 1.0, 1.0), {}
    else:
        length, head_length, strict = inference.max_len, inference.head_max_len, False
        temperatures, by_options = inference.temperature, inference.temperature_by_options
    return DecisionPolicy(
        family, length, head_length, strict, temperatures, by_options, cls_id, sep_id, mask_id, pad_id, tokens[2]
    )


def _validate_encoder_graph(encoder: ModernBERTConfig) -> None:
    if encoder.attention_bias or encoder.mlp_bias:
        raise UnsupportedDecisionArchitectureError(
            "modernbert_marker_v1 supports bias-free ModernBERT encoder attention/MLP projections only",
        )
    if (
        encoder.hidden_size <= 0
        or encoder.intermediate_size <= 0
        or encoder.num_attention_heads <= 0
        or encoder.num_hidden_layers <= 0
        or encoder.vocab_size <= 0
        or encoder.global_attn_every_n_layers <= 0
        or encoder.hidden_size % encoder.num_attention_heads
    ):
        raise ValueError("Invalid ModernBERT encoder dimensions")
    expected_layers = tuple(
        "full_attention" if index % encoder.global_attn_every_n_layers == 0 else "sliding_attention"
        for index in range(encoder.num_hidden_layers)
    )
    if encoder.layer_types is not None and encoder.layer_types != expected_layers:
        raise UnsupportedDecisionArchitectureError(
            "modernbert_marker_v1 requires the checkpoint's layer_types to match its global attention schedule",
        )


def _validate_marker_weights(weights: Mapping[str, Array], encoder: ModernBERTConfig, head_layers: int) -> None:
    dims, intermediate = encoder.hidden_size, encoder.intermediate_size
    expected: dict[str, tuple[int, ...]] = {
        "encoder.embeddings.tok_embeddings.weight": (encoder.vocab_size, dims),
        "type_emb.weight": (3, dims),
        "scorer.1.weight": (dims, dims),
        "scorer.1.bias": (dims,),
        "scorer.3.weight": (1, dims),
        "scorer.3.bias": (1,),
    }

    def add_norm(path: str, *, biases: bool) -> None:
        expected[f"{path}.weight"] = (dims,)
        if biases:
            expected[f"{path}.bias"] = (dims,)

    add_norm("encoder.embeddings.norm", biases=encoder.norm_bias)
    add_norm("encoder.final_norm", biases=encoder.norm_bias)
    add_norm("scorer.0", biases=True)
    for index in range(encoder.num_hidden_layers):
        prefix = f"encoder.layers.{index}"
        if index > 0:
            add_norm(f"{prefix}.attn_norm", biases=encoder.norm_bias)
        add_norm(f"{prefix}.mlp_norm", biases=encoder.norm_bias)
        expected.update(
            {
                f"{prefix}.attn.Wqkv.weight": (3 * dims, dims),
                f"{prefix}.attn.Wo.weight": (dims, dims),
                f"{prefix}.mlp.Wi.weight": (2 * intermediate, dims),
                f"{prefix}.mlp.Wo.weight": (dims, intermediate),
            }
        )
    for index in range(head_layers):
        prefix = f"head.layers.{index}"
        add_norm(f"{prefix}.norm1", biases=True)
        add_norm(f"{prefix}.norm2", biases=True)
        expected.update(
            {
                f"{prefix}.self_attn.in_proj_weight": (3 * dims, dims),
                f"{prefix}.self_attn.in_proj_bias": (3 * dims,),
                f"{prefix}.self_attn.out_proj.weight": (dims, dims),
                f"{prefix}.self_attn.out_proj.bias": (dims,),
                f"{prefix}.linear1.weight": (4 * dims, dims),
                f"{prefix}.linear1.bias": (4 * dims,),
                f"{prefix}.linear2.weight": (dims, 4 * dims),
                f"{prefix}.linear2.bias": (dims,),
            }
        )
    missing = set(expected) - weights.keys()
    unexpected = {
        name for name in weights.keys() - expected.keys() if not name.startswith("act_head.") and name != "temperature"
    }
    if missing or unexpected:
        raise ValueError(
            f"Incompatible modernbert_marker_v1 weights: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, shape in expected.items():
        value = weights[name]
        if value.shape != shape or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError(
                f"Incompatible tensor {name}: expected floating shape {shape}, got {value.shape} {value.dtype}"
            )


def import_decision_model(
    directory: Path | str,
    *,
    dtype: DTypeLike = jnp.bfloat16,
    sharding_config: ShardingConfig | None = None,
    adapters: DecisionCheckpointRegistry = DEFAULT_DECISION_ADAPTERS,
) -> tuple[DecisionModel, DecisionPolicy]:
    directory = Path(directory)
    checkpoint = adapters.resolve(directory)
    if checkpoint.architecture_id != "modernbert_marker_v1":
        raise UnsupportedDecisionArchitectureError(
            f"No graph converter registered for architecture {checkpoint.architecture_id!r}; "
            "this converter implements modernbert_marker_v1 only.",
        )
    layout = checkpoint.layout
    try:
        encoder = ModernBERTConfig.from_json(directory / layout.encoder_config_file)
    except (cattrs.errors.BaseValidationError, ValueError) as error:
        raise UnsupportedDecisionArchitectureError(
            f"Encoder configuration is incompatible with {checkpoint.architecture_id}: {directory}",
        ) from error
    _validate_encoder_graph(encoder)
    classifier = encoder.to_classifier_config()
    tokenizer_folder = directory / layout.tokenizer_directory
    tokenizer = Tokenizer.from_file(str(tokenizer_folder / "tokenizer.json"))
    policy = replace(
        _policy(tokenizer_folder, tokenizer, checkpoint.family, checkpoint.inference),
        architecture_id=checkpoint.architecture_id,
        adapter_id=checkpoint.adapter_id,
    )
    if (
        not 1 <= policy.max_length <= encoder.max_position_embeddings
        or not 1 <= policy.head_length <= policy.max_length
    ):
        raise ValueError("Decision policy exceeds the encoder context window")
    decision_config = DecisionConfig(
        embedding_config=classifier.embedding_config,
        type_embedding_config=classifier.embedding_config,
        embedding_norm_config=replace(classifier.embedding_norm_config, upcast_mode=UpcastMode.FULL_LAYER),
        encoder_config=replace(
            classifier.transformer_config,
            layer_configs=tuple(
                replace(
                    layer,
                    mixer_config=replace(layer.mixer_config, bf16_eager=True),
                    mlp_config=replace(
                        layer.mlp_config, activation=replace(layer.mlp_config.activation, pytorch_bf16=True)
                    ),
                    pre_mixer_norm_config=(
                        replace(layer.pre_mixer_norm_config, upcast_mode=UpcastMode.FULL_LAYER)
                        if layer.pre_mixer_norm_config is not None else None
                    ),
                    pre_mlp_norm_config=(
                        replace(layer.pre_mlp_norm_config, upcast_mode=UpcastMode.FULL_LAYER)
                        if layer.pre_mlp_norm_config is not None else None
                    ),
                )
                for layer in classifier.transformer_config.layer_configs
            ),
            output_norm_config=replace(
                classifier.transformer_config.output_norm_config, upcast_mode=UpcastMode.FULL_LAYER
            ),
        ),
        head_config=_head_config(encoder.hidden_size, checkpoint.head_layers),
        scorer_norm_config=NormalizationConfig(
            1e-5, None, UpcastMode.FULL_LAYER, subtract_mean=True, has_biases=True
        ),
        scorer_dense_config=LinearConfig(),
        scorer_readout_config=LinearConfig(),
        vocab_size=encoder.vocab_size,
        model_dim=encoder.hidden_size,
        max_sequence_length=policy.max_length,
        max_head_tokens=policy.head_length,
        pytorch_bf16=True,
    )
    codec = ChatCodecConfig(
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        None,
        "system",
        "user",
        "assistant",
        None,
        None,
    )
    config = DecisionModelConfig(token_codec_config=codec, decision_config=decision_config)
    if sharding_config is None:
        sharding_config = ShardingConfig.replicated()
    model = config.init(tokenizer, EmptyInitializer(dtype, sharding_config))
    with (directory / layout.weights_file).open("rb") as stream:
        _, raw_weights = safe_read(stream)
        source_weights = dict(raw_weights)
        _validate_marker_weights(source_weights, encoder, checkpoint.head_layers)
        weights = {key: value.astype(dtype) for key, value in source_weights.items()}
    decision = model.decision
    assert isinstance(decision.embedding, TiedEmbedding) and isinstance(decision.type_embedding, TiedEmbedding)
    embedding = load_tied_embedding(decision.embedding, weights, ParameterPath("encoder.embeddings.tok_embeddings"))
    type_embedding = load_tied_embedding(
        decision.type_embedding, {"type_emb.weight": weights["type_emb.weight"]}, ParameterPath("type_emb")
    )
    encoder_layers = tuple(
        _load_layer(layer, weights, f"encoder.layers.{index}", head=False)
        for index, layer in enumerate(decision.encoder.layers)
    )
    head_layers = tuple(
        _load_layer(layer, weights, f"head.layers.{index}", head=True)
        for index, layer in enumerate(decision.head.layers)
    )
    encoder_norm = _normalization(decision.encoder.output_norm, weights, "encoder.final_norm")
    encoder_module = eqx.tree_at(lambda m: (m.layers, m.output_norm), decision.encoder, (encoder_layers, encoder_norm))
    head_module = eqx.tree_at(lambda m: m.layers, decision.head, head_layers)
    decision = eqx.tree_at(
        lambda m: (
            m.embedding,
            m.embedding_norm,
            m.encoder,
            m.type_embedding,
            m.head,
            m.scorer_norm,
            m.scorer_dense,
            m.scorer_readout,
        ),
        decision,
        (
            embedding,
            _normalization(decision.embedding_norm, weights, "encoder.embeddings.norm"),
            encoder_module,
            type_embedding,
            head_module,
            _normalization(decision.scorer_norm, weights, "scorer.0"),
            load_linear(decision.scorer_dense, weights, ParameterPath("scorer.1")),
            load_linear(decision.scorer_readout, weights, ParameterPath("scorer.3")),
        ),
    )
    return eqx.tree_at(lambda m: m.decision, model, decision), policy


def export_decision_model(
    directory: Path | str,
    output_directory: Path | str,
    *,
    dtype: DTypeLike = jnp.bfloat16,
    adapters: DecisionCheckpointRegistry = DEFAULT_DECISION_ADAPTERS,
) -> DecisionPolicy:
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("Output directory must be empty")
    model, policy = import_decision_model(directory, dtype=dtype, adapters=adapters)
    model.save(output_directory)
    (output_directory / "decision_policy.json").write_text(json.dumps(_converter.unstructure(policy), indent=2) + "\n")
    return policy
