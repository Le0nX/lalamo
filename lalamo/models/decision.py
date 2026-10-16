from dataclasses import dataclass

from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.modules.decision import Decision, DecisionConfig

__all__ = ["DecisionModel", "DecisionModelConfig"]


@dataclass(frozen=True)
class DecisionModelConfig(ModelConfig[ChatCodecConfig]):
    decision_config: DecisionConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "DecisionModel":
        return DecisionModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=self.token_codec_config.init(tokenizer),
            decision=self.decision_config.init(initializer),
        )


class DecisionModel(Model[ChatCodecConfig, DecisionModelConfig, ChatCodec]):
    decision: Decision
