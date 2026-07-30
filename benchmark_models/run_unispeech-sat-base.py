#!/usr/bin/env python3
"""UniSpeech-SAT-base full-structure Hexagon benchmark."""
from transformers import UniSpeechSatConfig, UniSpeechSatForCTC

from full_audio_encoder import make_parser, run_full_audio_encoder


if __name__ == "__main__":
    run_full_audio_encoder(
        make_parser(__doc__).parse_args(),
        display_name="UniSpeech-SAT-base",
        config=UniSpeechSatConfig(),
        model_cls=UniSpeechSatForCTC,
        root_name="unispeech_sat",
        seed=991,
    )
