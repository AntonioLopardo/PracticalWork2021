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

pass_k_list = []

cnt = 0
for sample_q, sample_a in zip(sample_q_list, sample_a_list):
    print(colored("TESTING STARTED", "yellow"))
    cnt += 1
    prompt = f"{current_dataset.priming_text}\n\n#{sample_q}"
    # print(colored(f"\n\nPrompt:\n{prompt}", "green"))

    tokens = tokenizer(prompt, return_tensors="pt").input_ids
    generated_tokens = model.generate(
        tokens.long().cuda(),
        use_cache=True,
        do_sample=True,
        top_k=50,
        temperature=0.4,
        top_p=0.9,
        min_length=1,
        max_length=len(tokens[0]) + 100,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )

    list_outputs = hf.preproc_gen_toks(generated_tokens, len(tokens[0]), tokenizer)

    is_correct_list = [
        current_dataset.verify_pred_from_output(output, sample_q, sample_a)
        for output in list_outputs
    ]

    c = is_correct_list.count(True)

    pass_k = hf.pass_at_k(n, c, k)
    pass_k_list.append(pass_k)

    if cnt % 10 == 0:
        print(
            colored(
                f"@sample {cnt} -> Pass@{k} = {np.mean(np.array(pass_k_list))}", "green"
            )
        )
