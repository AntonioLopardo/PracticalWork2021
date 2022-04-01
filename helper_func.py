import numpy as np
import os
from termcolor import colored
curr_dir = os.getcwd()
os.chdir(os.path.join(curr_dir, 'CodeGen'))
from CodeGen.jaxformer.hf.sample import *
os.chdir(curr_dir)


def preproc_gen_toks(gen_toks, input_len, tokenizer):
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
        output = generated_text.split("\n\n")[0]
        list_out.append(output)
    return list_out


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

def load_CodeGen():
    """Load the CodeGen model and tokenizer

    :return HF_model: the model
    :return HF_tokenizer: the tokenizer
    """
    curr_dir = os.getcwd()
    os.chdir(os.path.join(curr_dir, 'CodeGen'))
    class args:
        def __init__(self):
            self.fp16 = True
            self.model = 'codegen-16B-mono'
            self.device = 'cuda:0'
            self.rng_seed = 42
            self.rng_deterministic = True
            self.pad = 50256
            self.context = 'def helloworld():'
            self.batch_size = 1
            self.max_length = 128

    args = args()

    models_nl = []
    models_pl = ['codegen-350M-mono', 'codegen-2B-mono', 'codegen-6B-mono', 'codegen-16B-mono']
    models = models_nl + models_pl

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        if args.model in models_pl:
            tokenizer = create_custom_gpt2_tokenizer()
        else:
            tokenizer = create_tokenizer()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = args.pad

    os.chdir(curr_dir)
    return model, tokenizer


def testing_loop(n, k, current_dataset, tokenizer, model, sample_q_list, sample_a_list):
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
        prompt = f"{current_dataset.priming_text}\n\n#{sample_q}"
        # print(colored(f"\n\nPrompt:\n{prompt}", "green"))

        if current_dataset.preprocess_sol(sample_a) == 22222222.0:
            pass
        else:
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

            list_outputs = preproc_gen_toks(generated_tokens, len(tokens[0]), tokenizer)

            is_correct_list = [
                current_dataset.verify_pred_from_output(output, sample_a)
                for output in list_outputs
            ]

            c = is_correct_list.count(True)

            pass_k = pass_at_k(n, c, k)
            pass_k_list.append(pass_k)

            if cnt % 5 == 0:
                print(
                    colored(
                        f"@sample {cnt} -> Pass@{k} = {np.mean(np.array(pass_k_list))}",
                        "white",
                    )
                )
    print(colored(f"\n\nPass@{k} = {np.mean(np.array(pass_k_list))}", "green"))

    return np.mean(np.array(pass_k_list))
