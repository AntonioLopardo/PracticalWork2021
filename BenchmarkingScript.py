import os
from pyexpat import model
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from termcolor import colored
import wandb
import importlib
import argparse
import pickle

import utils.dataset_handler as dh
import utils.loading_utils as lu
import utils.testing_utils as tu

gptj_model = "EleutherAI/gpt-j-6B"
codeparrot_model = "lvwerra/codeparrot"


def run_benchmark(
    model_name,
    func_impl_path,
    priming_text_path,
    results_path,
    model=None,
    tokenizer=None,
    full_test=False,
):

    # priming_text_path = "data/priming_texts/gsm8k/codegen/func_eq_short.txt"  # for codegen
    # wandb_run_name = "@100-codegen-0"
    # import exp_impl.func_def_eq_short as exp_impl

    exp_impl = importlib.import_module(f"exp_impl.{func_impl_path}")

    if model_name == "gpt-j":
        priming_text_path = (
            "data/priming_texts/gsm8k/gpt-j/gsm8k_fewer_alt.txt"  # for gpt-j
        )
        current_dataset = dh.init_dataset_from_name(
            "gsm8k", primingtext_path=priming_text_path
        )
    else:
        if full_test:
            current_dataset = dh.init_dataset_from_name(
                "gsm8k-test",
                primingtext_path=priming_text_path,
                sample_func=exp_impl.sample_n_for_prompting,
                generate_prompt_func=exp_impl.generate_prompt,
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

    if "_eq" in priming_text_path:
        sample_q_list, sample_a_list = current_dataset.sample_n_for_prompting(
            100, inc_eq=True
        )
    else:
        if full_test:
            test_set_size = len(current_dataset.data)
            sample_q_list, sample_a_list = current_dataset.sample_n_for_prompting(
                nr_entries=test_set_size
            )

        else:
            sample_q_list, sample_a_list = current_dataset.sample_n_for_prompting(100)

    with open("test_prompt.txt", "w") as f:
        f.write(current_dataset.generate_prompt(sample_q_list[0]))

    print(colored(sample_q_list[0], "blue"))
    print(colored(sample_a_list[0], "green"))

    if model:
        # Set up for CodeGen
        config = lu.codegen_gen_args()
        config.num_return_sequences = 4  # 4 for cluster
        # config.num_return_sequences = 6
        config.k = 1
        config.max_length_after_input = 250
        # config.top_p = 0.95
        config.top_p = 0.95
        config.top_k = 50
        # config.temperature = 0.7
        config.temperature = 0.61
        config.min_length = 3

        config.priming_text = priming_text_path
        config.func_impl_path = func_impl_path

        with wandb.init(
            project="PracticalWork",
            entity="antoniolopardo",
            config=config,
            name=f"{priming_text}_{func_impl_path}",
        ):

            tu.set_all_seeds(model_name)
            transformers.set_seed(5)
            pass_at_k, pass_at_k_list = tu.testing_loop(
                current_dataset,
                tokenizer,
                model,
                sample_q_list,
                sample_a_list,
                config,
                func_def_mod=True,
                print_output=False,
            )

            priming_text_save = priming_text_path.split("/")[-1]

            with open(
                f"{results_path}/{priming_text_save}_{func_impl_path}_pass@1.pkl", "wb"
            ) as f:
                pickle.dump(pass_at_k_list, f)
            with open(
                f"{results_path}/{priming_text_save}_{func_impl_path}_config_pass@1.pkl",
                "wb",
            ) as f:
                pickle.dump(config, f)

            wandb.log({"pass_at_k": pass_at_k})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="codegen",
        nargs="?",
        help="Which model to train: tucker",
    )
    parser.add_argument(
        "--func_impl_path",
        type=str,
        default="func_def_general",
        nargs="?",
        help="Which func_def_impl to use: func_def_eq_short",
    )
    parser.add_argument(
        "--priming_text",
        type=str,
        default="func_eq_short",
        nargs="?",
        help="Which priming text to use: func_def_eq_short",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results_lists_long",
        nargs="?",
        help="Where to save the results",
    )

    args = parser.parse_args()

    tu.set_all_seeds(args.model_name)
    transformers.set_seed(5)

    if args.model_name == "codegen":
        """CodeGen runs in the venv venv"""
        model_args = lu.model_args()
        # model_args.model = "codegen-350M-mono"
        model, tokenizer = lu.load_CodeGen(model_args)

    print(colored("Running Benchmark", "green"))

    priming_text_list = [
        # "data/priming_texts/gsm8k/codegen/func_eq_short.txt",
        "data/priming_texts/gsm8k/codegen/func_short.txt",
    ]

    func_impl_list = ["func_def_general" for _ in range(len(priming_text_list))]
    results_path_list = [
        os.path.join("results_lists", pt_dir.split("/")[-2], "full_test")
        for pt_dir in priming_text_list
    ]

    for priming_text, func_impl, results_path in zip(
        priming_text_list, func_impl_list, results_path_list
    ):
        run_benchmark(
            args.model_name,
            func_impl,
            priming_text,
            results_path,
            model=model,
            tokenizer=tokenizer,
            full_test=True,
        )
