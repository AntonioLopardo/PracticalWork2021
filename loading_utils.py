import numpy as np
import os
from termcolor import colored
import re
import sys
from io import StringIO
from contextlib import redirect_stdout

sys.path.append("/home/PracticalWork2021/CodeGen/")
from jaxformer.hf.sample import *


class model_args:
    def __init__(self):
        self.fp16 = True
        self.model = "codegen-16B-mono"
        self.device = "cuda:0"
        self.rng_seed = 42
        self.rng_deterministic = True
        self.pad = 50256


class codegen_gen_args:
    def __init__(self):
        self.k = 1
        self.do_sample = True
        self.top_k = 50
        self.temperature = 0.4
        self.top_p = 0.9
        self.min_length = 10
        self.max_length_after_input = 100
        self.num_return_sequences = 1


class gptj_gen_args:
    def __init__(self):
        self.k = 3
        self.do_sample = True
        self.top_k = 50
        self.temperature = 0.4
        self.top_p = 0.9
        self.min_length = 1
        self.max_length_after_input = 100
        self.num_return_sequences = 4


def load_CodeGen(args):
    """Load the CodeGen model and tokenizer

    :param model_args args: arguments for the model
    :return HF_model: the model
    :return HF_tokenizer: the tokenizer
    """

    models_nl = []
    models_pl = [
        "codegen-350M-mono",
        "codegen-2B-mono",
        "codegen-6B-mono",
        "codegen-16B-mono",
    ]
    models = models_nl + models_pl

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    device = torch.device(args.device)
    ckpt = f"/home/PracticalWork2021/CodeGen/checkpoints/{args.model}"

    if "cluster" in os.getcwd():
        ckpt = f"/cluster/scratch/alopardo/CodeGen/checkpoints/{args.model}"

    with print_time("loading parameters"):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)

    with print_time("loading tokenizer"):
        if args.model in models_pl:
            tokenizer = create_custom_gpt2_tokenizer()
        else:
            tokenizer = create_tokenizer()
        tokenizer.padding_side = "left"
        tokenizer.pad_token = args.pad

    return model, tokenizer
