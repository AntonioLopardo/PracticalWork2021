import numpy as np


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
