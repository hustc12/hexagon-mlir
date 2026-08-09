#!/usr/bin/env python3
"""HuBERT-base full-structure Hexagon benchmark."""
from transformers import HubertConfig, HubertForCTC

from full_audio_encoder import make_parser, run_full_audio_encoder


if __name__ == "__main__":
    run_full_audio_encoder(
        make_parser(__doc__).parse_args(),
        display_name="HuBERT-base",
        config=HubertConfig(),
        model_cls=HubertForCTC,
        root_name="hubert",
        seed=961,
    )
