"""Pinned Perplexity Decider text inference; images require another adapter."""

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import cattrs

from lalamo.model_import.d3 import CodeReadoutPolicy, export_code_readout, sha256
from lalamo.model_import.decision_tokenizer import DecisionTokenizer
from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config

_converter = cattrs.Converter()


@dataclass(frozen=True)
class PerplexityContract:
    format_version: Literal[1]
    base_model: Literal["Qwen/Qwen3.8-27B"]
    revision: str
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    temperature: float
    attention_mode: Literal["noncausal_full_attention"]
    pooling: Literal["last"]


@dataclass(frozen=True)
class ReleaseFile:
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PerplexityManifest:
    repo_id: Literal["perplexity-ai/pplx-decider-v1.1-27b"]
    files: dict[str, ReleaseFile]


def read_manifest(directory: Path) -> PerplexityManifest:
    return _converter.structure(json.loads((directory / "release-manifest.json").read_text()), PerplexityManifest)


def build_perplexity_policy(
    directory: Path, tokenizer: DecisionTokenizer, revision: str, context: int
) -> CodeReadoutPolicy:
    manifest = read_manifest(directory)
    contract = _converter.structure(json.loads((directory / "decision_config.json").read_text()), PerplexityContract)
    if len(contract.codes) != 255 or len(contract.token_ids) != 255 or len(set(contract.token_ids)) != 255:
        raise ValueError("Perplexity requires its complete, unique 255-code mapping")
    if not math.isfinite(contract.temperature) or contract.temperature <= 0:
        raise ValueError("Invalid native temperature")
    source_name = "source/src/autojev/model.py"
    if sha256(directory / source_name) != manifest.files[source_name].sha256:
        raise ValueError("Native prompt source differs from the release manifest")
    tree = ast.parse((directory / source_name).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decision_messages"]
    if len(functions) != 1:
        raise ValueError("Missing native decision_messages function")
    prompts = []
    for node in ast.walk(functions[0]):
        if isinstance(node, ast.Dict):
            literals = {
                key.value: value.value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
            }
            if literals.get("role") == "system":
                prompts.append(literals["content"])
    if len(prompts) != 1 or not isinstance(prompts[0], str):
        raise ValueError("Missing native literal system prompt")
    marker = "__UZU_PERPLEXITY_CONTENT_BOUNDARY__"
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": prompts[0]}, {"role": "user", "content": marker}],
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
            raise ValueError("Codebook differs from tokenizer")
        if tokenizer.encode(suffix + code, add_special_tokens=False) != [*tail, token]:
            raise ValueError("Answer code is not one token at the assistant boundary")
    return CodeReadoutPolicy(
        "qwen_code_readout_v1",
        "pplx_code_v1",
        manifest.repo_id,
        revision,
        context,
        contract.temperature,
        "bfloat16",
        contract.codes,
        contract.token_ids,
        prefix,
        suffix,
    )


def export_perplexity(directory: Path | str, output: Path | str, *, revision: str) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    directory, output = Path(directory), Path(output)
    config = HFQwen35Config.from_json(directory / "config.json")
    if (config.model_type, config.hidden_size, config.num_hidden_layers) != ("qwen3_5", 5120, 64):
        raise ValueError("This Perplexity adapter validates the 27B Qwen release only")
    manifest = read_manifest(directory)
    names = [
        name
        for name in manifest.files
        if name.endswith(".safetensors")
        or name
        in (
            "config.json",
            "decision_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "source/src/autojev/model.py",
            "source/src/autojev/types.py",
            "LICENSE",
            "NOTICE",
        )
    ]
    checked = {}
    for name in names:
        expected = manifest.files[name]
        path = directory / name
        if path.stat().st_size != expected.bytes or sha256(path) != expected.sha256:
            raise ValueError(f"Pinned checkpoint identity mismatch: {name}")
        checked[name] = expected.sha256
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    # 8192 in native prepare() is a caller default, not an architectural limit.
    policy = build_perplexity_policy(
        directory, cast("DecisionTokenizer", tokenizer), revision, config.max_position_embeddings
    )
    export_code_readout(directory, output, config, policy, checked, ("LICENSE", "NOTICE", "release-manifest.json"))
