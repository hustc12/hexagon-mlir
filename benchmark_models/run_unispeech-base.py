#!/usr/bin/env python3
"""UniSpeech-base full-structure Hexagon benchmark."""
from transformers import UniSpeechConfig, UniSpeechForCTC

from full_audio_encoder import make_parser, run_full_audio_encoder


if __name__ == "__main__":
    run_full_audio_encoder(
        make_parser(__doc__).parse_args(),
        display_name="UniSpeech-base",
        config=UniSpeechConfig(),
        model_cls=UniSpeechForCTC,
        root_name="unispeech",
        seed=377,
    )
