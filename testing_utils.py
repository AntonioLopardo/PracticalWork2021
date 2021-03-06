import numpy as np
import os
from termcolor import colored
import re
import sys
from io import StringIO
from contextlib import redirect_stdout
from wrapt_timeout_decorator import *

sys.path.append("/home/PracticalWork2021/CodeGen/")
from jaxformer.hf.sample import *


@timeout(45)
def execute(c):
    exec(c, globals())


@timeout(45)
def solve_timeout(func):
    return func()


def set_all_seeds(model=None):
    """Set all seeds"""
    if model == "gpt-j":
        torch.manual_seed(42)
        np.random.seed(42)
    elif model == "codegen":
        # torch.manual_seed(0)
        # np.random.seed(8)
        os.environ["PYTHONHASHSEED"] = str(0)
        torch.backends.cudnn.deterministic = True
        random.seed(hash("setting random seeds") % 2**32 - 1)
        np.random.seed(hash("improves reproducibility") % 2**32 - 1)
        torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
        torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
    else:
        torch.manual_seed(0)
        np.random.seed(8)


def preproc_gen_toks(gen_toks, input_len, tokenizer, func_def_mod=False):
    """Process generated tokens from model keeping only up to \n\n

    :param list of list gen_toks: the output form the model
    :param int input_len: input lenght used to ignore the prompt
    :param HF_tokenizer tokenizer: tokenizer used for decoding
    :return list of str : list of generated outputs
    """
    list_out = []
    for gen_tok in gen_toks:
        last_tokens = gen_tok[input_len:]
        generated_text = tokenizer.decode(last_tokens)
        # print(generated_text)
        if func_def_mod:
            print_pattern = re.compile(r"float\(([^)]+)\)")
            # print_pattern = re.compile(r"return ([^\n]*)\n")
        else:
            print_pattern = re.compile(r"print\(([^)]+)\)")
        split_list = re.split(print_pattern, generated_text)
        if len(split_list) > 1:
            if func_def_mod:
                output = f"{split_list[0]}float({split_list[1]})\n"
                # output = f"{split_list[0]}return float({split_list[1]})\n"
            else:
                output = f"{split_list[0]}print({split_list[1]})\n"
        else:
            output = "INVALID OUTPUT"
        list_out.append(output)
    return list_out


def verify_pred_from_output(
    output, sample_a, preprocess_func, func_def_mod=False, print_output=False
):
    """Verify the the output generates the solution

    :param str output: output generated by the language model
    :param str sample_a: str of solution, should be castable to float otherwise it will be changed to default wrong value
    :return bool: True if the output generates the solution, False otherwise
    """
    if print_output:
        print(colored(f"Return Sequence:", "yellow"))

    s = 1111111111.0
    avoid_input = re.compile(r"input\(([^)]+)\)")
    if func_def_mod:
        if avoid_input.search(output):
            pass
        else:
            try:
                # exec(f"def solve_exercise():{output}\n", globals())
                execute(f"def solve_exercise():{output}\n")
                # s = solve_exercise()
                s = solve_timeout(solve_exercise)
                if print_output:
                    print(colored(f"{s}", "yellow"))
            except TimeoutError as e:
                print(colored(f"TimeoutError: {e}", "red"))
                print(f"def solve_exercise():{output}\n")
                s = 1111111111.0
            except Exception as e:
                s = 1111111111.0

    else:
        f = StringIO()
        with redirect_stdout(f):
            avoid_input = re.compile(r"input\(([^)]+)\)")
            if avoid_input.search(output):
                pass
            else:
                try:
                    exec(output)
                except Exception as e:
                    pass

        s = f.getvalue()
        try:
            s = float(s)
        except Exception as e:
            s = 1111111111.0

    is_correct = s == preprocess_func(sample_a)
    if print_output:
        print(colored(f"{output}", "green" if is_correct else "red"))
    return is_correct


def pass_at_k(n, c, k):
    """Implementaton of the pass at k metric

    :param int n: total number of samples
    :param int c: number of correct samples
    :param int k: k in pass at k
    :return float: result of the metric
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def testing_loop(
    current_dataset,
    tokenizer,
    model,
    sample_q_list,
    sample_a_list,
    gen_args,
    func_def_mod=False,
    print_output=False,
):
    """Perform full testing loop avoiding question with non float solutions

    :param int n: total number of samples in pass at k
    :param int k: k in pass at k
    :param math_dataset current_dataset: dataset to test
    :param HF_tokenizer tokenizer: tokenizer used to encode
    :param HF_model model: language model used to output solutions
    :param list of str sample_q_list: list of questions with instructions for the LM
    :param list of str sample_a_list: list of solutions
    :return float: the pass at k metric
    """
    pass_k_list = []
    cnt = 0
    print(colored("TESTING STARTED", "yellow"))
    for sample_q, sample_a in zip(sample_q_list, sample_a_list):
        cnt += 1

        prompt = current_dataset.generate_prompt(sample_q)

        if current_dataset.preprocess_sol(sample_a) == 22222222.0:
            pass
        else:
            tokens = tokenizer(prompt, return_tensors="pt").input_ids
            generated_tokens = model.generate(
                tokens.long().cuda(),
                use_cache=True,
                do_sample=True,
                top_k=gen_args.top_k,
                temperature=gen_args.temperature,
                top_p=gen_args.top_p,
                min_length=len(tokens[0]) + gen_args.min_length,
                max_length=len(tokens[0]) + gen_args.max_length_after_input,
                num_return_sequences=gen_args.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
            )

            list_outputs = preproc_gen_toks(
                generated_tokens, len(tokens[0]), tokenizer, func_def_mod=func_def_mod
            )

            if print_output:
                [print(colored(f"{len(i)}", "green")) for i in list_outputs]

            is_correct_list = [
                verify_pred_from_output(
                    output,
                    sample_a,
                    current_dataset.preprocess_sol,
                    func_def_mod=func_def_mod,
                    print_output=print_output,
                )
                for output in list_outputs
            ]

            c = is_correct_list.count(True)

            pass_k = pass_at_k(gen_args.num_return_sequences, c, gen_args.k)
            pass_k_list.append(pass_k)

            if cnt % 5 == 0:
                print(
                    colored(
                        f"@sample {cnt} -> Pass@{gen_args.k} = {np.mean(np.array(pass_k_list))}",
                        "white",
                    )
                )
    print(colored(f"\n\nPass@{gen_args.k} = {np.mean(np.array(pass_k_list))}", "green"))

    return np.mean(np.array(pass_k_list)), pass_k_list
