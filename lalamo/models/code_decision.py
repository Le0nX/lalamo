from dataclasses import dataclass, replace
from typing import Literal

import jax.numpy as jnp
from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.modules.decoder import Decoder, DecoderConfig
from lalamo.modules.linear import Linear, LinearConfig


@dataclass(frozen=True)
class CodeDecisionModelConfig(ModelConfig[ChatCodecConfig]):
    architecture_id: Literal["qwen_code_readout_v1"]
    decoder_config: DecoderConfig
    readout_rows: int
    readout_dtype: Literal["float32", "bfloat16"]
    max_sequence_length: int

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "CodeDecisionModel":
        if self.readout_rows != 255 or self.max_sequence_length < 1:
            raise ValueError("Invalid Qwen code-readout dimensions")
        readout_initializer = replace(initializer, default_dtype=jnp.dtype(self.readout_dtype))
        return CodeDecisionModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=self.token_codec_config.init(tokenizer),
            decoder=self.decoder_config.init(initializer),
            readout=LinearConfig().init(
                readout_initializer,
                self.decoder_config.transformer_config.model_dim,
                (self.readout_rows,),
                has_biases=False,
            ),
        )


class CodeDecisionModel(Model[ChatCodecConfig, CodeDecisionModelConfig, ChatCodec]):
    decoder: Decoder
    readout: Linear
