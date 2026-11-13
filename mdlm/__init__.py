"""Project 3: a small masked (absorbing-state) diffusion language model. Code package `mdlm`.

Importing `mdlm` makes Project 1's teacher package importable (its tokenizer wrapper, data loader, evaluation
helpers and the autoregressive GPT used as the baseline):
    import mdlm
    from src.tokenizer import StoryTokenizer

Project 1 (and Project 2) are only read, never changed. Bytecode caching is off so nothing is written there.
"""
import os
import sys

sys.dont_write_bytecode = True

PROJECT3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PROJECT3)
PROJECT1 = os.path.join(ROOT, "project1")
PROJECT2 = os.path.join(ROOT, "project2")
TEACHER_DIR = os.path.join(PROJECT1, "teacher")
DATA_DIR = os.path.join(PROJECT1, "data", "full")           # cleaned TinyStories as token ids (train/val/test .bin)
BASE_TOKENIZER = os.path.join(TEACHER_DIR, "tokenizer.json")  # Project 1 BPE, 8,192 tokens
TOKENIZER = os.path.join(PROJECT3, "tokenizer", "tokenizer.json")  # same + [MASK] (id 8192)
CACHE_DIR = os.path.join(PROJECT3, "data")                    # story-start indices (not in git)
RESULTS = os.path.join(PROJECT3, "results")

if TEACHER_DIR not in sys.path:
    sys.path.insert(0, TEACHER_DIR)
