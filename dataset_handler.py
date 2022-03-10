from abc import ABC, abstractmethod
from xml.etree import ElementTree
import numpy as np
from termcolor import colored


class math_dataset(ABC):
    @abstractmethod
    def __init__(self, dataset_path, priming_text_pth, dataset_name):
        self.data = self.load_dataset(dataset_path)
        self.priming_text = self.load_priming_text(priming_text_pth)
        self.dataset_name = dataset_name

    @abstractmethod
    def load_dataset(self, dataset_path):
        pass

    def load_priming_text(self, priming_text_pth):
        with open(priming_text_pth, "r") as f:
            return f.read()

    @abstractmethod
    def print_entry_from_idx(self, idx):
        pass

    @abstractmethod
    def sample_n_for_prompting(self, nr_entries):
        pass

    @abstractmethod
    def preprocess_sol(self, raw_sol):
        pass

    def generate_prompt(self, entry_q):
        return f"{self.priming_text}\n\n#{entry_q}"


class asdiv_dataset(math_dataset):
    def __init__(self, dataset_path, priming_text_pth, dataset_name):
        super().__init__(dataset_path, priming_text_pth, dataset_name)

    def load_dataset(self, dataset_path):
        data = {}
        dom = ElementTree.parse(dataset_path)

        # XML parsing
        body_list = dom.findall("ProblemSet/Problem/Body")
        answer_list = dom.findall("ProblemSet/Problem/Answer")
        question_list = dom.findall("ProblemSet/Problem/Question")
        formula_list = dom.findall("ProblemSet/Problem/Formula")
        stype_list = dom.findall("ProblemSet/Problem/Solution-Type")

        data["body_list"] = body_list
        data["answer_list"] = answer_list
        data["question_list"] = question_list
        data["formula_list"] = formula_list
        data["stype_list"] = stype_list

        return data

    def print_entry_from_idx(self, entry_idx):
        print(
            colored(
                f"{self.data['body_list'][entry_idx].text} {self.data['question_list'][entry_idx].text}",
                "yellow",
            )
        )
        print(colored(f"{self.data['formula_list'][entry_idx].text}", "green"))
        print(colored(f"{self.data['answer_list'][entry_idx].text}", "green"))
        print("\n" + "-" * 100 + "\n")

    def sample_n_for_prompting(self, nr_entries):
        rand_indexes = np.random.randint(0, len(self.data["body_list"]), nr_entries)

        sample_a_list = []
        sample_q_list = []
        for rand_index in rand_indexes:
            sample_q_list.append(
                f"{self.data['body_list'][rand_index].text} Write a program that prints {self.data['question_list'][rand_index].text}"
            )
            sample_a_list.append(self.data["answer_list"][rand_index].text)

        return sample_q_list, sample_a_list

    def preprocess_sol(self, raw_sol):
        sol = raw_sol.split(" ")[0]
        if ":" in sol:
            sol = int(sol.split(":")[0]) / int(sol.split(":")[1])
        try:
            sol = float(sol)
        except ValueError:
            sol = 22222222.0
        return float(sol)


# asdiv_path = "data/nlu-asdiv-dataset/dataset/ASDiv.xml"
# gsm8k_path = "data/grade-school-math/grade_school_math/data/train.jsonl"
# singleEq_path = "data/TACL2015/questions.json"

# dataset = asdiv_dataset(asdiv_path, "data/priming_texts/asdiv.txt", "asdiv")

# dataset.print_entry_from_idx(135)
