import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cattrs
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import ModernBertConfig, ModernBertModel

from lalamo.model import Model
from lalamo.model_import.decision import DecisionPolicy, export_decision_model, import_decision_model
from lalamo.model_import.decision_adapters import (
    DEFAULT_DECISION_ADAPTERS,
    DecisionCheckpointRegistry,
    DecisionCheckpointSpec,
    JuliaCheckpointAdapter,
    UnsupportedDecisionArchitectureError,
)
from lalamo.model_import.model_configs.huggingface.modern_bert import ModernBERTConfig, ModernBERTRoPEParameters
from lalamo.module import Keychain
from lalamo.modules.embedding import EmbeddingForwardPassConfig
from lalamo.modules.transformer import TransformerForwardPassConfig
from lalamo.utils.sharding import ShardingConfig


class TorchDecisionHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.type_emb = torch.nn.Embedding(3, 64)
        self.head = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                64,
                1,
                256,
                dropout=0,
                activation="relu",
                batch_first=True,
                norm_first=True,
            ),
            2,
            enable_nested_tensor=False,
        )
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
        )


def _checkpoint(directory: Path, *, family: str) -> tuple[ModernBertModel, TorchDecisionHead]:
    torch.manual_seed(73)
    config = ModernBertConfig(
        hidden_size=64,
        intermediate_size=96,
        num_attention_heads=1,
        num_hidden_layers=2,
        vocab_size=12,
        max_position_embeddings=64,
        local_attention=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    encoder = ModernBertModel(config).eval()
    head = TorchDecisionHead()
    head.eval()
    (directory / "encoder").mkdir(parents=True)
    config_dict = config.to_dict()
    config_dict["gradient_checkpointing"] = False
    config_dict["global_attn_every_n_layers"] = 3
    config_dict["architectures"] = ["ModernBertForMaskedLM"]
    (directory / "encoder/config.json").write_text(json.dumps(config_dict))
    tokenizer_folder = directory / "tokenizer"
    tokenizer_folder.mkdir()
    tokens = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "unknown": 4}
    tokens.update({f"token{index}": index for index in range(5, 12)})
    Tokenizer(WordLevel(tokens, unk_token="unknown")).save(str(tokenizer_folder / "tokenizer.json"))
    (tokenizer_folder / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": {"content": "[MASK]"},
                "pad_token": "[PAD]",
            }
        )
    )
    if family == "julia":
        (directory / "config.json").write_text("{}")
        (directory / "julia_config.json").write_text(
            json.dumps({"format_version": 1, "architecture": "JuliaDecisionModel", "head_layers": 2})
        )
        (directory / "inference-policy.json").write_text(
            json.dumps({"max_length": 64, "head_length": 32, "strict_encoding": True})
        )
    else:
        (directory / "rl_agent_config.json").write_text(
            json.dumps(
                {
                    "head_layers": 2,
                    "max_len": 64,
                    "head_max_len": 32,
                    "temperature": [1.1, 1.2, 1.3],
                    "temperature_by_options": {"choice:11+": 0.1},
                }
            )
        )
    weights = {f"encoder.{name}": weight for name, weight in encoder.state_dict().items()}
    weights.update(head.state_dict())
    # These training-only tensors must not be exported or required by runtime.
    weights["act_head.weight"] = torch.ones(2, 64)
    weights["temperature"] = torch.tensor([1.0, 1.0, 1.0])
    save_file(weights, directory / "model.safetensors")
    return encoder, head


@pytest.mark.parametrize("family", ["julia", "laya"])
def test_decision_import_matches_pytorch_and_roundtrips(tmp_path: Path, family: str) -> None:
    source = tmp_path / "source"
    encoder, head = _checkpoint(source, family=family)
    model, policy = import_decision_model(source, dtype=jnp.float32)
    token_ids = torch.tensor([[1, 5, 6, 7, 3, 8, 3, 9, 2]])
    markers = torch.tensor([[4, 6]])
    with torch.no_grad():
        encoded = encoder(token_ids).last_hidden_state
        hidden = head.head(encoded + head.type_emb(torch.tensor([2]))[:, None, :])
        expected = head.scorer(hidden[:, [4, 6], :]).squeeze(-1)
    actual = model.decision(
        jnp.asarray(token_ids.numpy()),
        jnp.array([2]),
        jnp.asarray(markers.numpy()),
        embedding_forward_pass_config=EmbeddingForwardPassConfig.for_tracer_tests(),
        transformer_forward_pass_config=TransformerForwardPassConfig.for_tracer_tests(),
        keychain=Keychain.init(42, sharding_config=ShardingConfig.replicated()),
    )
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), atol=2e-5, rtol=2e-5)
    output = tmp_path / "export"
    exported_policy = export_decision_model(source, output)
    loaded = Model.load(output, ShardingConfig.replicated())
    assert loaded.config == model.config
    exported = loaded.export().arrays
    assert not any("act_head" in name or "temperature" in name for name in exported)
    assert exported["decision.head.layers.0.pre_mixer_norm.scales"].dtype == jnp.float32
    assert exported["decision.head.layers.0.mixer.qkvg_projection.weights.weights"].dtype == jnp.bfloat16
    assert policy.family == exported_policy.family == family
    assert policy.adapter_id == f"{family}_v1"
    assert policy.architecture_id == "modernbert_marker_v1"
    stored_policy = json.loads((output / "decision_policy.json").read_text())
    stored_policy.pop("adapter_id")
    stored_policy.pop("architecture_id")
    legacy_policy = cattrs.Converter().structure(stored_policy, DecisionPolicy)
    assert legacy_policy.architecture_id == "modernbert_marker_v1"
    assert legacy_policy.adapter_id is None
    if family == "laya":
        assert exported_policy.temperatures_by_options["choice:11+"] == 0.1
    with pytest.raises(ValueError, match="empty"):
        export_decision_model(source, output)


def test_new_rope_schema_preserves_julia_local_theta(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _checkpoint(source, family="julia")
    encoder = ModernBERTConfig.from_json(source / "encoder/config.json")
    julia_encoder = replace(
        encoder,
        rope_parameters={
            "full_attention": ModernBERTRoPEParameters(160000.0),
            "sliding_attention": ModernBERTRoPEParameters(160000.0),
        },
    )
    encoded = julia_encoder.to_classifier_config().transformer_config
    assert encoded.layer_configs[0].rope_config is not None
    assert encoded.layer_configs[1].rope_config is not None
    assert encoded.layer_configs[0].rope_config.base == 160000.0
    assert encoded.layer_configs[1].rope_config.base == 160000.0
    legacy = replace(encoder, rope_parameters={}).to_classifier_config().transformer_config
    assert legacy.layer_configs[1].rope_config is not None
    assert legacy.layer_configs[1].rope_config.base == 10000.0


@dataclass(frozen=True)
class LocalFineTuneAdapter:
    adapter_id: str = "local_julia_finetune_v1"
    architecture_id: str = "modernbert_marker_v1"

    def matches(self, directory: Path) -> bool:
        return (directory / "finetune.safetensors").is_file()

    def load(self, directory: Path) -> DecisionCheckpointSpec:
        original = JuliaCheckpointAdapter().load(directory)
        return replace(
            original,
            adapter_id=self.adapter_id,
            architecture_id=self.architecture_id,
            layout=replace(original.layout, weights_file="finetune.safetensors"),
        )


def test_registered_adapter_reuses_graph_for_compatible_local_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _checkpoint(source, family="julia")
    copied = tmp_path / "arbitrary-hf-finetune-name"
    shutil.copytree(source, copied)
    (copied / "model.safetensors").rename(copied / "finetune.safetensors")
    registry = DecisionCheckpointRegistry().with_adapter(LocalFineTuneAdapter())
    original, _ = import_decision_model(source, dtype=jnp.float32)
    reused, policy = import_decision_model(copied, dtype=jnp.float32, adapters=registry)
    assert original.config == reused.config
    for name, value in original.export().arrays.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(reused.export().arrays[name]))
    assert policy.adapter_id == "local_julia_finetune_v1"
    assert policy.architecture_id == "modernbert_marker_v1"
    with pytest.raises(ValueError, match="already registered"):
        registry.with_adapter(LocalFineTuneAdapter())


def test_unknown_architecture_does_not_fall_back_to_laya(tmp_path: Path) -> None:
    _checkpoint(tmp_path, family="laya")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertForSequenceClassification"]})
    )
    with pytest.raises(UnsupportedDecisionArchitectureError, match="Unsupported decision checkpoint"):
        import_decision_model(tmp_path)


def test_adapter_rejects_incompatible_graph_and_weights(tmp_path: Path) -> None:
    _checkpoint(tmp_path, family="julia")
    (tmp_path / "model.safetensors").rename(tmp_path / "finetune.safetensors")
    unknown_graph = DecisionCheckpointRegistry().with_adapter(
        LocalFineTuneAdapter(architecture_id="different_graph_v1")
    )
    with pytest.raises(UnsupportedDecisionArchitectureError, match="No graph converter"):
        import_decision_model(tmp_path, adapters=unknown_graph)
    from safetensors.torch import load_file  # noqa: PLC0415

    weights = load_file(tmp_path / "finetune.safetensors")
    weights["type_emb.weight"] = torch.zeros(4, 64)
    save_file(weights, tmp_path / "finetune.safetensors")
    registry = DecisionCheckpointRegistry().with_adapter(LocalFineTuneAdapter())
    with pytest.raises(ValueError, match=r"Incompatible tensor type_emb\.weight"):
        import_decision_model(tmp_path, adapters=registry)
    weights["type_emb.weight"] = torch.zeros(3, 64)
    weights["another_head.weight"] = torch.zeros(2, 64)
    save_file(weights, tmp_path / "finetune.safetensors")
    with pytest.raises(ValueError, match=r"unexpected=.*another_head"):
        import_decision_model(tmp_path, adapters=registry)


def test_compatible_checkpoint_matches_layout_without_repository_identity(tmp_path: Path) -> None:
    _checkpoint(tmp_path, family="julia")
    spec = DEFAULT_DECISION_ADAPTERS.resolve(tmp_path)
    assert spec.adapter_id == "julia_v1"
    assert spec.architecture_id == "modernbert_marker_v1"
    assert spec.family == "julia"


def test_marker_adapter_rejects_incompatible_encoder_graph(tmp_path: Path) -> None:
    _checkpoint(tmp_path, family="julia")
    path = tmp_path / "encoder/config.json"
    config = json.loads(path.read_text())
    config["attention_bias"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(UnsupportedDecisionArchitectureError, match="bias-free"):
        import_decision_model(tmp_path)
    config["attention_bias"] = False
    config["layer_types"] = ["full_attention", "full_attention"]
    path.write_text(json.dumps(config))
    with pytest.raises(UnsupportedDecisionArchitectureError, match="layer_types"):
        import_decision_model(tmp_path)
    config["model_type"] = "bert"
    path.write_text(json.dumps(config))
    with pytest.raises(UnsupportedDecisionArchitectureError, match="Encoder configuration is incompatible"):
        import_decision_model(tmp_path)
