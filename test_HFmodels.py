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

torch.manual_seed(0)
np.random.seed(8)

gptj_model = "EleutherAI/gpt-j-6B"
codeparrot_model = "lvwerra/codeparrot"

"""Load the priming text to add to the prompt and sample a question"""
# priming_text_path = "data/priming_texts/gsm8k_onlyfullanswer.txt"
# priming_text = read_string_from_file("data/priming_texts/singleEq.txt")
priming_text_path = "data/priming_texts/asdiv/asdiv.txt"

print(colored("Prompt from: " + priming_text_path + "\n", "yellow"))

current_dataset = dh.init_dataset_from_name("asdiv", primingtext_path=priming_text_path)

sample_q_list, sample_a_list = current_dataset.sample_n_for_prompting(500)

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

hf.testing_loop(n, k, current_dataset, tokenizer, model, sample_q_list, sample_a_list)
