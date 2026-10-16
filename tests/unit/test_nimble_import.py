import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.model_import.common import _import_chat_codec
from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.nimble import nimble_config_map
from lalamo.model_import.origins import LocalOrigin


@pytest.mark.parametrize("inline", [True, False])
def test_merged_nimble_tokenizer_template_reaches_real_import(tmp_path: Path, inline: bool) -> None:
    template = "{{ messages }}"
    config = {"bos_token": "unknown", "eos_token": "unknown"}
    if inline:
        config["chat_template"] = template
    else:
        (tmp_path / "chat_template.jinja").write_text(template)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    Tokenizer(WordLevel({"unknown": 0}, unk_token="unknown")).save(str(tmp_path / "tokenizer.json"))
    spec = LanguageModelSpec(
        vendor="Bespoke Labs",
        family="Nimble",
        name="test",
        size="test",
        origin=LocalOrigin(str(tmp_path), ()),
        config_type=HFQwen35Config,
        configs=nimble_config_map(tmp_path),
    )
    _, codec = _import_chat_codec(spec)
    assert codec.prompt_template == template


def test_merged_nimble_missing_template_is_explicit(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="neither an inline nor a standalone"):
        nimble_config_map(tmp_path)
