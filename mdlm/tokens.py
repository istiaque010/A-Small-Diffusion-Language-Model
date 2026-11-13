"""Tokenizer: Project 1's BPE (8,192 tokens) plus one new special token, [MASK] = 8192.

All ids 0..8191 keep their meaning, so Project 1's tokenized data and its autoregressive models can be used
unchanged. Only the diffusion model's embedding/output layer has one extra row (vocabulary 8,193).
"""
import os

from tokenizers import Tokenizer

import mdlm
from src.tokenizer import StoryTokenizer

MASK = "[MASK]"
PAD_ID, UNK_ID, BOS_ID, EOS_ID, MASK_ID = 0, 1, 2, 3, 8192
VOCAB = 8193


def build_tokenizer(out_path: str = mdlm.TOKENIZER) -> str:
    """Write tokenizer/tokenizer.json = Project 1's tokenizer + [MASK]. Checks that no existing id moved."""
    tok = Tokenizer.from_file(mdlm.BASE_TOKENIZER)
    assert tok.get_vocab_size() == 8192
    tok.add_special_tokens([MASK])
    assert tok.token_to_id(MASK) == MASK_ID, tok.token_to_id(MASK)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tok.save(out_path)
    return out_path


def load_tokenizer() -> StoryTokenizer:
    if not os.path.exists(mdlm.TOKENIZER):
        build_tokenizer()
    tok = StoryTokenizer(mdlm.TOKENIZER)
    assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) == (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
    assert tok.tok.token_to_id(MASK) == MASK_ID
    return tok
