#!/usr/bin/env python3
from transformers import Data2VecAudioConfig, Data2VecAudioForCTC
from audio_encoder_candidate import parser, run_candidate
if __name__=="__main__":
    c=Data2VecAudioConfig(vocab_size=32,hidden_size=64,num_hidden_layers=2,num_attention_heads=2,
        intermediate_size=128,conv_dim=(32,32),conv_kernel=(10,3),conv_stride=(5,2),
        num_conv_pos_embeddings=2,num_conv_pos_embedding_groups=4,apply_spec_augment=False,
        hidden_act="gelu_new",feat_extract_activation="gelu_new")
    run_candidate(parser().parse_args(),"Data2Vec-Audio-base",c,Data2VecAudioForCTC,"data2vec_audio",220)
