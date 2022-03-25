import numpy as np
from termcolor import colored


def preproc_gen_toks(gen_toks, input_len, tokenizer):
    list_out = []
    for gen_tok in gen_toks:
        last_tokens = gen_tok[input_len:]
        generated_text = tokenizer.decode(last_tokens)
        output = generated_text.split("\n\n")[0]
        list_out.append(output)
    return list_out


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def testing_loop(n, k, current_dataset, tokenizer, model, sample_q_list, sample_a_list):
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
                current_dataset.verify_pred_from_output(output, sample_q, sample_a)
                for output in list_outputs
            ]

            c = is_correct_list.count(True)

            pass_k = pass_at_k(n, c, k)
            pass_k_list.append(pass_k)

            if cnt % 10 == 0:
                print(
                    colored(
                        f"@sample {cnt} -> Pass@{k} = {np.mean(np.array(pass_k_list))}",
                        "green",
                    )
                )
