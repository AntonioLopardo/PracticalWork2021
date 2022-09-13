import numpy as np
import re


def sample_n_for_prompting(self, nr_entries):
    rand_indexes = np.random.randint(0, len(self.data), nr_entries)

    sample_a_list = []
    sample_q_list = []
    for rand_index in rand_indexes:
        sample_q_list.append(
            "Write a program that prints the answer to the following question. "
            + self.data[rand_index]["question"]
        )
        # sample_a_list.append(self.data[rand_index]["answer"])
        sample_a_list.append(
            re.findall(r"#### \w+", self.data[rand_index]["answer"])[0][5:]
        )

    return sample_q_list, sample_a_list


def generate_prompt(self, entry_q):
    """Generates full prompt using the saved priming text and the entry's question
    :param str entry_q: entry question
    :return str: full prompt
    """
    return f"{self.priming_text}\n\n#{entry_q}"
