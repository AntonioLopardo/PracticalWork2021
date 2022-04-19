import numpy as np
import re


def sample_n_for_prompting(self, nr_entries=1):
    rand_indexes = np.random.randint(0, len(self.data), nr_entries)

    sample_a_list = []
    sample_q_list = []
    for rand_index in rand_indexes:
        # sample_q_list.append("def exercise6():\n"+ '    """Write a program that returns the answer to the following question. ' + self.data[rand_index]["question"]+ '"""')
        sample_q_list.append(
            "def exercise9():\n" + '    """' + self.data[rand_index]["question"] + '"""'
        )
        sample_a_list.append(
            re.findall(r"#### \w+", self.data[rand_index]["answer"])[0][5:]
        )

    return sample_q_list, sample_a_list


def generate_prompt(self, entry_q):
    return f"{self.priming_text}\n\n\n{entry_q}"
