#!/usr/bin/env python3
"""Wav2Vec2-base full-structure Hexagon benchmark."""
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from full_audio_encoder import make_parser, run_full_audio_encoder


if __name__ == "__main__":
    run_full_audio_encoder(
        make_parser(__doc__).parse_args(),
        display_name="Wav2Vec2-base",
        config=Wav2Vec2Config(),
        model_cls=Wav2Vec2ForCTC,
        root_name="wav2vec2",
        seed=960,
    )
