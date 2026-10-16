"""Native d3 code-readout checkpoint conversion, initially the 27B release."""

import ast
import hashlib
import importlib.metadata
import json
import math
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import cattrs
import equinox as eqx
import jax.numpy as jnp
from tokenizers import Tokenizer

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.decision_tokenizer import DecisionTokenizer
from lalamo.model_import.loaders.huggingface import load_huggingface_decoder, load_linear
from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.code_decision import CodeDecisionModelConfig
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.deltanet import DeltaNetConfig
from lalamo.safetensors import safe_read
from lalamo.utils.lazy_collections import LazyDict
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig

_converter = cattrs.Converter()


@dataclass(frozen=True)
class D3Contract:
    format_version: Literal[1]
    format_id: Literal["d3-code-readout-v1"]
    prompt: Literal["d3"]
    base_model: str
    revision: str
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    temperature: float
    attention_mode: Literal["noncausal_full_attention"]
    pooling: Literal["last"]
    readout_dtype: Literal["float32"]
    max_length: int | None = None


@dataclass(frozen=True)
class CodeReadoutPolicy:
    architecture_id: Literal["qwen_code_readout_v1"]
    adapter_id: Literal["d3_code_v1", "pplx_code_v1"]
    model_id: str
    revision: str
    max_sequence_length: int
    temperature: float
    readout_dtype: Literal["float32", "bfloat16"]
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    chat_prefix: str
    chat_suffix: str


D3Policy = CodeReadoutPolicy


@dataclass(frozen=True)
class ManifestHashes:
    files_sha256: dict[str, str]


@dataclass(frozen=True)
class D3Manifest:
    schema: Literal["d3-package-manifest/1"]
    repo_id: str
    identity: ManifestHashes
    runtime: ManifestHashes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_d3_contract(directory: Path | str) -> D3Contract:
    contract = _converter.structure(json.loads((Path(directory) / "decision_config.json").read_text()), D3Contract)
    if len(contract.codes) != 255 or len(contract.token_ids) != 255 or len(set(contract.token_ids)) != 255:
        raise ValueError("d3 requires its complete, unique 255-code readout mapping")
    if not math.isfinite(contract.temperature) or contract.temperature <= 0:
        raise ValueError("Invalid native calibration temperature")
    return contract


def build_d3_policy(
    directory: Path | str,
    tokenizer: DecisionTokenizer,
    model_revision: str,
    context: int,
) -> D3Policy:
    directory = Path(directory)
    contract = read_d3_contract(directory)
    manifest = _converter.structure(json.loads((directory / "MODEL_MANIFEST.json").read_text()), D3Manifest)
    for name in ("d3_format.py", "d3_runtime.py"):
        if sha256(directory / name) != manifest.runtime.files_sha256.get(name):
            raise ValueError(f"Native runtime source changed: {name}")
    tree = ast.parse((directory / "d3_format.py").read_text())
    definitions = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in n.targets)
    ]
    if len(definitions) != 1:
        raise ValueError("Missing literal native system prompt")
    prompt = ast.literal_eval(definitions[0].value)
    if not isinstance(prompt, str):
        raise TypeError("Invalid system prompt")
    marker = "__UZU_D3_CONTENT_BOUNDARY__"
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt}, {"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if text.count(marker) != 1:
        raise ValueError("Chat template changed the content boundary")
    prefix, suffix = text.split(marker)
    tail = tokenizer.encode(suffix, add_special_tokens=False)
    for code, token in zip(contract.codes, contract.token_ids, strict=True):
        if tokenizer.encode(code, add_special_tokens=False) != [token]:
            raise ValueError("Codebook does not match the tokenizer")
        if tokenizer.encode(suffix + code, add_special_tokens=False) != [*tail, token]:
            raise ValueError("Code is not one token at the assistant boundary")
    maximum = min(context, contract.max_length) if contract.max_length is not None else context
    return D3Policy(
        "qwen_code_readout_v1",
        "d3_code_v1",
        manifest.repo_id,
        model_revision,
        maximum,
        contract.temperature,
        contract.readout_dtype,
        contract.codes,
        contract.token_ids,
        prefix,
        suffix,
    )


def d3_decoder_config(config: HFQwen35Config) -> DecoderConfig:
    def full_layer(norm: NormalizationConfig | None) -> NormalizationConfig | None:
        if norm is None:
            return None
        return replace(norm, upcast_mode=UpcastMode.FULL_LAYER)

    decoder = config.to_decoder_config({})
    layers = []
    for layer in decoder.transformer_config.layer_configs:
        mixer = layer.mixer_config
        if isinstance(mixer, AttentionConfig):
            mixer = replace(
                mixer,
                is_causal=False,
                query_norm_config=full_layer(mixer.query_norm_config),
                key_norm_config=full_layer(mixer.key_norm_config),
            )
        elif isinstance(mixer, DeltaNetConfig):
            mixer = replace(mixer, norm_config=replace(mixer.norm_config, upcast_mode=UpcastMode.FULL_LAYER))
        else:
            raise TypeError("d3 requires Qwen full-attention/DeltaNet layers")
        layers.append(
            replace(
                layer,
                mixer_config=mixer,
                pre_mixer_norm_config=full_layer(layer.pre_mixer_norm_config),
                pre_mlp_norm_config=replace(layer.pre_mlp_norm_config, upcast_mode=UpcastMode.FULL_LAYER),
            )
        )
    transformer = replace(
        decoder.transformer_config,
        layer_configs=tuple(layers),
        output_norm_config=replace(decoder.transformer_config.output_norm_config, upcast_mode=UpcastMode.FULL_LAYER),
    )
    # Only input embeddings are used; the tied vocabulary readout is never run.
    # The independent 255-row readout is loaded separately below.
    return replace(decoder, embedding_config=TiedEmbeddingConfig(None, None), transformer_config=transformer)


def export_d3(directory: Path | str, output: Path | str, *, revision: str) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    contract = read_d3_contract(directory)
    source_config = HFQwen35Config.from_json(directory / "config.json")
    supported = {
        "Qwen/Qwen3.8-27B": (5120, 64),
        "Qwen/Qwen3.5-4B": (2560, 32),
        "Qwen/Qwen3.5-9B": (4096, 32),
    }
    if source_config.model_type != "qwen3_5" or supported.get(contract.base_model) != (
        source_config.hidden_size,
        source_config.num_hidden_layers,
    ):
        raise ValueError("Unsupported d3 backbone or dimensions")
    manifest = _converter.structure(json.loads((directory / "MODEL_MANIFEST.json").read_text()), D3Manifest)
    checked = {}
    for name, expected in manifest.identity.files_sha256.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Checkpoint identity mismatch: {name}")
        checked[name] = expected
    hf_tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    policy = build_d3_policy(
        directory, cast("DecisionTokenizer", hf_tokenizer), revision, source_config.max_position_embeddings
    )
    export_code_readout(
        directory, output, source_config, policy, checked, ("LICENSE", "NOTICE", "MODEL_MANIFEST.json")
    )


def export_code_readout(
    directory: Path,
    output: Path,
    source_config: HFQwen35Config,
    policy: CodeReadoutPolicy,
    checked: dict[str, str],
    provenance_files: tuple[str, ...],
) -> None:
    if output.exists():
        raise FileExistsError(output)
    codec = ChatCodecConfig(
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        None,
        "system",
        "user",
        "assistant",
        None,
        None,
    )
    config = CodeDecisionModelConfig(
        token_codec_config=codec,
        architecture_id="qwen_code_readout_v1",
        decoder_config=d3_decoder_config(source_config),
        readout_rows=255,
        readout_dtype=policy.readout_dtype,
        max_sequence_length=policy.max_sequence_length,
    )
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    model = config.init(tokenizer, EmptyInitializer(jnp.bfloat16, ShardingConfig.replicated()))
    with ExitStack() as stack:
        shards = [
            safe_read(stack.enter_context(path.open("rb")))[1] for path in sorted(directory.glob("model*.safetensors"))
        ]
        if not shards:
            raise ValueError("Missing backbone shards")
        owners = {key: shard for shard in shards for key in shard}
        weights = LazyDict(set(owners), lambda key: owners[key][key])
        decoder = load_huggingface_decoder(model.decoder, weights)
        _, readout = safe_read(stack.enter_context((directory / "readout.safetensors").open("rb")))
        weight = readout["weight"]
        if weight.shape != (255, source_config.hidden_size) or weight.dtype != jnp.dtype(policy.readout_dtype):
            raise ValueError("Readout shape or native precision does not match the contract")
        head = load_linear(model.readout, {"readout.weight": weight}, ParameterPath("readout"), None)
        model = eqx.tree_at(lambda m: (m.decoder, m.readout), model, (decoder, head))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d3-convert-", dir=output.parent) as temporary:
        staged = Path(temporary) / "uzu"
        model.save(staged)
        (staged / "decision_policy.json").write_text(json.dumps(asdict(policy), indent=2) + "\n")
        for name in provenance_files:
            shutil.copy2(directory / name, staged / name)
        provenance = {
            "model": policy.model_id,
            "revision": policy.revision,
            "files_sha256": checked,
            "architecture": "qwen_code_readout_v1",
            "backbone_dtype": "bfloat16",
            "readout_dtype": policy.readout_dtype,
            "attention_mode": "noncausal_full_attention",
            "temperature": policy.temperature,
            "weights_license": "Apache-2.0",
            "runtime_code_license": "Apache-2.0",
            "converter_code_license": "MIT",
            "vision": "not exported; text only",
            "versions": {name: importlib.metadata.version(name) for name in ("lalamo", "jax", "transformers")},
            "parity": "conversion is not proof of parity; verify native and uzu separately",
        }
        (staged / "conversion_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        staged.rename(output)
