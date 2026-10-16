"""Nimble v3's causal Qwen decoder plus its external prompt/codebook contract.

The release's prompt text remains in the exported, separately licensed artifact.
It is read as a literal after verifying the source hashes, never executed.
"""

import ast
import gc
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import cattrs
import jax.numpy as jnp

from lalamo.model_import.common import import_model
from lalamo.model_import.decision_tokenizer import DecisionTokenizer
from lalamo.model_import.huggingface_tokenizer_config import HFTokenizerConfig
from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config
from lalamo.model_import.model_spec import ConfigMap, LanguageModelSpec
from lalamo.model_import.origins import FileSpec, LocalOrigin
from lalamo.models.language_model import GenerationConfig
from lalamo.utils.sharding import ShardingConfig

_converter = cattrs.Converter()


@dataclass(frozen=True)
class NimbleContract:
    task: Literal["schema_candidate_classification_v2"]
    model: Literal["Qwen/Qwen3.5-9B"]
    revision: str
    max_choices: int
    candidate_encoding: Literal["uppercase_single_token_v1"]
    candidate_codes: tuple[str, ...]
    candidate_token_ids: tuple[int, ...]
    prompt_source_sha256: dict[str, str]


@dataclass(frozen=True)
class NimblePolicy:
    architecture_id: Literal["causal_token_readout_v1"]
    adapter_id: Literal["nimble_v3"]
    max_sequence_length: int
    temperature: float
    candidate_codes: tuple[str, ...]
    candidate_token_ids: tuple[int, ...]
    chat_prefix_letter: str
    chat_prefix_short: str
    chat_suffix: str


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_nimble_contract(adapter: Path | str, prompt_sources: Path | str) -> tuple[NimbleContract, str]:
    adapter, prompt_sources = Path(adapter), Path(prompt_sources)
    contract = _converter.structure(json.loads((adapter / "schema_config.json").read_text()), NimbleContract)
    if (
        contract.max_choices != 255
        or len(contract.candidate_codes) != 255
        or len(set(contract.candidate_codes)) != 255
    ):
        raise ValueError("Nimble v3 requires its complete 255-code contract")
    if len(contract.candidate_token_ids) != 255 or len(set(contract.candidate_token_ids)) != 255:
        raise ValueError("Nimble candidate token ids must be unique")
    if contract.revision != "c202236235762e1c871ad0ccb60c8ee5ba337b9a":
        raise ValueError("Unsupported Nimble v3 base revision; validate a new release policy first")
    for name in ("parallel_schema.py", "extended_schema.py"):
        if contract.prompt_source_sha256.get(name) != _sha(prompt_sources / name):
            raise ValueError(f"Prompt source does not match the checkpoint contract: {name}")
    tree = ast.parse((prompt_sources / "extended_schema.py").read_text())
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Expected one literal SYSTEM_PROMPT in the pinned release source")
    prompt = ast.literal_eval(assignments[0].value)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Invalid release system prompt")
    return contract, prompt


def build_nimble_policy(
    adapter: Path | str,
    prompt_sources: Path | str,
    tokenizer: DecisionTokenizer,
    max_sequence_length: int,
) -> NimblePolicy:
    contract, system_prompt = load_nimble_contract(adapter, prompt_sources)
    if max_sequence_length <= 0:
        raise ValueError("Decoder context must be positive")
    for code, token in zip(contract.candidate_codes, contract.candidate_token_ids, strict=True):
        if not code or not code.isascii() or not code.isupper() or not code.isalpha():
            raise ValueError("Candidate code must be uppercase ASCII letters")
        if tokenizer.encode(code, add_special_tokens=False) != [token]:
            raise ValueError("Candidate code/token mismatch")
    marker = "__UZU_NIMBLE_CONTENT_BOUNDARY__"
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt}, {"role": "user", "content": marker}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in (system_prompt, system_prompt.replace("one-letter", "short"))
    ]
    if any(not isinstance(text, str) or text.count(marker) != 1 for text in rendered):
        raise ValueError("Chat template did not preserve the content boundary")
    (letter, suffix), (short, other_suffix) = (text.split(marker) for text in rendered)
    if suffix != other_suffix:
        raise ValueError("Chat suffix depends on the system prompt")
    # Also validate the actual assistant boundary, not just isolated token ids.
    tail = suffix
    ids = tokenizer.encode(tail, add_special_tokens=False)
    for code, token in zip(contract.candidate_codes, contract.candidate_token_ids, strict=True):
        if tokenizer.encode(tail + code, add_special_tokens=False) != [*ids, token]:
            raise ValueError("Candidate code merges at the assistant boundary")
    return NimblePolicy(
        "causal_token_readout_v1",
        "nimble_v3",
        max_sequence_length,
        1.0,
        contract.candidate_codes,
        contract.candidate_token_ids,
        letter,
        short,
        suffix,
    )


def nimble_config_map(merged_checkpoint: Path | str) -> ConfigMap:
    checkpoint = Path(merged_checkpoint)
    tokenizer_config = HFTokenizerConfig.from_json(checkpoint / "tokenizer_config.json")
    if tokenizer_config.chat_template is not None:
        return ConfigMap(generation_config=GenerationConfig())
    template = checkpoint / "chat_template.jinja"
    if not template.is_file():
        raise ValueError("Merged Nimble tokenizer has neither an inline nor a standalone chat template")
    return ConfigMap(generation_config=GenerationConfig(), chat_template=FileSpec(template.name))


def export_merged_nimble(
    merged_checkpoint: Path | str,
    adapter: Path | str,
    prompt_sources: Path | str,
    output: Path | str,
) -> None:
    """Export a caller-supplied PEFT-merged BF16 Qwen3.5 checkpoint and policy."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    merged_checkpoint, adapter, output = Path(merged_checkpoint), Path(adapter), Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = HFQwen35Config.from_json(merged_checkpoint / "config.json")
    if config.model_type != "qwen3_5" or config.hidden_size != 4096 or config.num_hidden_layers != 32:
        raise ValueError("Nimble v3 requires the dense Qwen3.5-9B text backbone")
    tokenizer = AutoTokenizer.from_pretrained(merged_checkpoint, local_files_only=True, trust_remote_code=False)
    policy = build_nimble_policy(
        adapter, prompt_sources, cast("DecisionTokenizer", tokenizer), config.max_position_embeddings
    )
    weights = tuple(path.name for path in sorted(merged_checkpoint.glob("*.safetensors")))
    if not weights:
        raise ValueError("No merged model weights found")
    spec = LanguageModelSpec(
        vendor="Bespoke Labs",
        family="Nimble",
        name="Bespoke-Nimble-9B-v3",
        size="9B",
        origin=LocalOrigin(str(merged_checkpoint), weights),
        config_type=HFQwen35Config,
        configs=nimble_config_map(merged_checkpoint),
    )
    imported = import_model(spec, sharding_config=ShardingConfig.replicated(), dtype=jnp.bfloat16)
    imported.model.save(output)
    (output / "decision_policy.json").write_text(json.dumps(asdict(policy), indent=2) + "\n")
    shutil.copy2(adapter / "schema_config.json", output / "nimble_schema_config.json")


def merge_and_export_nimble(
    base: Path | str,
    adapter: Path | str,
    prompt_sources: Path | str,
    output: Path | str,
) -> None:
    """CPU preparation only. Native PEFT merges LoRA; existing lalamo exports it.

    A merged BF16 model is a separately recorded numerical policy. Compare it
    with unmerged native PEFT before claiming equivalence on a new checkpoint.
    """
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration  # noqa: PLC0415

    base, adapter, output = Path(base), Path(adapter), Path(output)
    if output.exists():
        raise FileExistsError(output)
    contract, _ = load_nimble_contract(adapter, prompt_sources)
    config = HFQwen35Config.from_json(base / "config.json")
    if config.hidden_size != 4096 or config.num_hidden_layers != 32 or config.model_type != "qwen3_5":
        raise ValueError("Expected original dense Qwen3.5-9B base")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if tokenizer is None:
        raise ValueError("Cannot load the pinned tokenizer")
    build_nimble_policy(adapter, prompt_sources, cast("DecisionTokenizer", tokenizer), config.max_position_embeddings)
    source_files = [*sorted(base.glob("*.safetensors")), adapter / "adapter_model.safetensors"]
    if len(source_files) < 2 or not all(path.is_file() for path in source_files):
        raise ValueError("Base and adapter weights are required")
    provenance = {
        "base_model": contract.model,
        "base_revision_required": contract.revision,
        "base_identity_note": "Caller supplies the pinned original base; hashes below identify actual local files.",
        "source_sha256": {str(path): _sha(path) for path in source_files},
        "weights_license": {"base": "Apache-2.0", "adapter": "CC-BY-NC-4.0"},
        "prompt_artifact_license": "CC-BY-NC-4.0",
        "converter_code_license": "MIT",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "lalamo")},
        "precision": "native PEFT CPU merge_and_unload(safe_merge=True), BF16 base; BF16 lalamo decoder",
        "parity": "not established by conversion; native unmerged vs merged vs uzu gates required",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nimble-convert-", dir=output.parent) as temporary:
        merged = Path(temporary) / "merged"
        original = Qwen3_5ForConditionalGeneration.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": "cpu"},
            attn_implementation="eager",
        )
        peft = PeftModel.from_pretrained(original, adapter, is_trainable=False, local_files_only=True)
        model = peft.merge_and_unload(safe_merge=True)
        model.save_pretrained(merged, safe_serialization=True)
        tokenizer.save_pretrained(merged)
        del model, peft, original
        gc.collect()
        staged = Path(temporary) / "uzu"
        export_merged_nimble(merged, adapter, prompt_sources, staged)
        (staged / "conversion_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        staged.rename(output)
