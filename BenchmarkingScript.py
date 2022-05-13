import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored
import wandb
import importlib
import argparse
import pickle

import dataset_handler as dh
import loading_utils as lu
import testing_utils as tu

gptj_model = "EleutherAI/gpt-j-6B"
codeparrot_model = "lvwerra/codeparrot"


def run_benchmark(model_name, func_impl_path, priming_text):

    # priming_text_path = "data/priming_texts/gsm8k/codegen/func_eq_short.txt"  # for codegen
    # wandb_run_name = "@100-codegen-0"
    # import exp_impl.func_def_eq_short as exp_impl

    priming_text_path = f"data/priming_texts/gsm8k/{model_name}/{priming_text}"

    exp_impl = importlib.import_module(f"exp_impl.{func_impl_path}")

    if model_name == "gpt-j":
        """GPT-J and codeparrot models run in HFTest venv"""
        tokenizer = AutoTokenizer.from_pretrained(gptj_model)
        model = AutoModelForCausalLM.from_pretrained(gptj_model).half().eval().cuda()
    elif model_name == "codegen":
        """CodeGen runs in the venv venv"""
        model_args = lu.model_args()
        # model_args.model = "codegen-350M-mono"
        model, tokenizer = lu.load_CodeGen(model_args)

    """Load gsm8k"""

    if model_name == "gpt-j":
        priming_text_path = (
            "data/priming_texts/gsm8k/gpt-j/gsm8k_fewer_alt.txt"  # for gpt-j
        )
        current_dataset = dh.init_dataset_from_name(
            "gsm8k", primingtext_path=priming_text_path
        )
    else:
        current_dataset = dh.init_dataset_from_name(
            "gsm8k",
            primingtext_path=priming_text_path,
            sample_func=exp_impl.sample_n_for_prompting,
            generate_prompt_func=exp_impl.generate_prompt,
        )

    tu.set_all_seeds()
    # tu.set_all_seeds_alt()

    sample_q_list, sample_a_list = current_dataset.sample_n_for_prompting(10)

    with open("test_prompt.txt", "w") as f:
        f.write(current_dataset.generate_prompt(sample_q_list[0]))

    print(colored(sample_q_list[0], "blue"))
    print(colored(sample_a_list[0], "green"))

    # Set up for CodeGen
    config = lu.codegen_gen_args()
    config.num_return_sequences = 4  # 4 for cluster
    # config.num_return_sequences = 6
    config.k = 3
    config.max_lenght_after_input = 250
    # config.top_p = 0.95
    config.top_p = 0.95
    config.top_k = 50
    # config.temperature = 0.7
    config.temperature = 0.61
    config.min_length = 3

    tu.set_all_seeds(model_name)
    _, general_pass_at_k = tu.testing_loop(
        current_dataset,
        tokenizer,
        model,
        sample_q_list,
        sample_a_list,
        config,
        func_def_mod=True,
        print_output=False,
    )

    with open(f"results_list/{priming_text}_{func_impl_path}.pkl", "wb") as f:
        pickle.dump(general_pass_at_k, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="codegen",
        nargs="?",
        help="Whihc model to train: tucker",
    )
    parser.add_argument(
        "--func_impl_path",
        type=str,
        default="func_def_eq_short",
        nargs="?",
        help="Which func_def_impl to use: func_def_eq_short",
    )
    parser.add_argument(
        "--priming_text",
        type=str,
        default="func_eq_short.txt",
        nargs="?",
        help="Which priming text to use: func_def_eq_short",
    )

    args = parser.parse_args()
    run_benchmark(
        args.model,
        args.func_impl_path,
        args.priming_text,
    )
