import numpy as np
import re
from termcolor import colored


def sample_n_for_prompting(self, nr_entries=1):
    rand_indexes = np.random.randint(0, len(self.data), nr_entries)

    sample_a_list = []
    sample_q_list = []
    for rand_index in rand_indexes:
        # sample_q_list.append("def exercise6():\n"+ '    """Write a program that returns the answer to the following question. ' + self.data[rand_index]["question"]+ '"""')
        sample_q_list.append(
            f"def exercise4():\n"
            + '    """\n    '
            + self.data[rand_index]["question"]
            + " Hint: use these equations"
            + extract_eq(self.data[rand_index]["answer"])
            + '\n    """'
        )
        sample_a_list.append(
            re.findall(r"#### \w+", self.data[rand_index]["answer"])[0][5:]
        )

    return sample_q_list, sample_a_list


def generate_prompt(self, entry_q):
    return f"{self.priming_text}\n\n\n{entry_q}"


def extract_eq(full_answer):
    eq_pattern = re.compile(r"<<([^>]+)>>")
    split_list = re.findall(eq_pattern, full_answer)

    eq_string = ""

    for i, eq in enumerate(split_list):
        eq_string += f" eq{i+1}: {eq}"

    return eq_string
