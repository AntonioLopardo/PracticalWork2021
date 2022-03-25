import os
from xml.etree import ElementTree
import numpy as np
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from io import StringIO
from contextlib import redirect_stdout
from termcolor import colored

import dataset_handler as dh
import helper_func as hf

import wandb

torch.manual_seed(0)
np.random.seed(8)

gptj_model = "EleutherAI/gpt-j-6B"
codeparrot_model = "lvwerra/codeparrot"

# dataset_name = "gsm8k"
dataset_name = "asdiv"
# dataset_name = "SingleEq"

"""Load the priming text to add to the prompt and sample a question"""
# priming_text_path = "data/priming_texts/gsm8k/gsm8k_fewer_alt.txt"
# priming_text = read_string_from_file("data/priming_texts/singleEq.txt")
priming_text_path = "data/priming_texts/asdiv/asdiv.txt"

print(colored("Prompt from: " + priming_text_path + "\n", "yellow"))

current_dataset = dh.init_dataset_from_name(
    dataset_name, primingtext_path=priming_text_path
)

nr_samples = 100
sample_q_list, sample_a_list = current_dataset.sample_n_for_prompting(nr_samples)

print(colored(sample_q_list[0], "blue"))
print(colored(sample_a_list[0], "white"))

print(colored("\nMODEL LOADING (about 100 sec)", "yellow"))
tokenizer = AutoTokenizer.from_pretrained(gptj_model)
model = AutoModelForCausalLM.from_pretrained(gptj_model).half().eval().cuda()

print(colored("MODEL LOADED", "white"))

torch.manual_seed(42)
np.random.seed(42)

n = 4
k = 3
"""n = 3
k = 3"""

config = {
    "dataset": dataset_name,
    "nr_samples": nr_samples,
    "machine": "datacrunch",
    "model": "GPT-J(6B) Half",
    "n": n,
    "k": k,
}

wandb.init(
    project="PracticalWork",
    entity="antoniolopardo",
    config=config,
    name="baseline_asdiv",
)

pass_at_k = hf.testing_loop(
    n, k, current_dataset, tokenizer, model, sample_q_list, sample_a_list
)

wandb.log({"pass_at_k": pass_at_k})
artifact = wandb.Artifact(name="priming_text", type="txt")
artifact.add_file(priming_text_path)
wandb.log_artifact(artifact)
