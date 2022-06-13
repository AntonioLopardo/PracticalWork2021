import numpy as np
import re
from termcolor import colored


def remove_last_question(sample_q):
    comma_list_senteces = sample_q.split(",")
    period_list_senteces = sample_q.split(".")

    if len(comma_list_senteces[-1]) > len(period_list_senteces[-1]):
        return sample_q[: -len(period_list_senteces[-1])]
    else:
        return sample_q[: -len(comma_list_senteces[-1])]


def get_num_operands_instep(step):
    eq_pattern = re.compile(r"<<([^>]+)>>")
    split_list = re.findall(eq_pattern, step)
    op_patter = re.compile(r"[\+\-\*\/\=]")

    num_operands = 0
    for i, eq in enumerate(split_list):
        num_operands += len(re.split(op_patter, eq))

    return num_operands


def sample_n_for_prompting(self, nr_entries=1, ex_number=4, inc_eq=False):
    rand_indexes = np.random.randint(0, len(self.data), nr_entries)

    sample_a_list = []
    sample_q_list = []
    sample_steps_list = []
    for rand_index in rand_indexes:
        if inc_eq:
            sample_q_list.append(
                f"def exercise{ex_number}():\n"
                + '    """\n    '
                + self.data[rand_index]["question"]
                + " Hint: use these equations"
                + extract_eq(self.data[rand_index]["answer"])
                + '\n    """'
            )
        else:
            sample_q_list.append(
                f"def exercise{ex_number}():\n"
                + '    """\n    '
                + remove_last_question(self.data[rand_index]["question"])
                + '\n    """'
            )
        sample_a_list.append(
            re.findall(r"#### \w+", self.data[rand_index]["answer"])[0][5:]
        )
        sample_steps = [
            f" {step.split(' **')[0]}\n    Use no more than {get_num_operands_instep(step)} variables\n    "
            for step in self.data[rand_index]["answer"].split("\n")
        ][:-1]
        sample_steps_list.append(sample_steps)

    return sample_q_list, sample_steps_list, sample_a_list


def generate_prompt(self, entry_q):
    return f"{self.priming_text}\n\n\n{entry_q}"


def extract_eq(full_answer):
    eq_pattern = re.compile(r"<<([^>]+)>>")
    split_list = re.findall(eq_pattern, full_answer)

    eq_string = ""

    for i, eq in enumerate(split_list):
        eq_string += f" eq{i+1}: {eq}"

    return eq_string
