from abc import ABC, abstractmethod
from xml.etree import ElementTree
import numpy as np
from termcolor import colored
from io import StringIO
from contextlib import redirect_stdout
import json
import re


class math_dataset(ABC):
    @abstractmethod
    def __init__(
        self,
        dataset_path,
        priming_text_pth,
        dataset_name,
    ):
        """Initialize the dataset, loading the dataset priming text and setting the dataset name

        :param str dataset_path: path to the dataset
        :param str priming_text_pth: path to the priming text
        :param str dataset_name: name of the dataset
        """
        self.data = self.load_dataset(dataset_path)
        self.priming_text = self.load_priming_text(priming_text_pth)
        self.dataset_name = dataset_name

    @abstractmethod
    def load_dataset(self, dataset_path):
        """Load dataset from dataset path, specific for each dataset

        :param str dataset_path: path to the dataset
        """
        pass

    def load_priming_text(self, priming_text_pth):
        """Read priming text from file

        :param str priming_text_pth: path to priming text to load
        :return str: loaded priming text
        """
        with open(priming_text_pth, "r") as f:
            return f.read()

    @abstractmethod
    def print_entry_from_idx(self, idx):
        """Print entry of dataset form index

        :param int idx: index of the entry in the dataset to print
        """
        pass

    @abstractmethod
    def sample_n_for_prompting(self, nr_entries=1):
        """Sample nr_entries entries from the dataset, already adding the natural language instruction for the model

        :param int nr_entries: number of entries to return
        :return couple of lists: two lists of aligned questions and answers
        """
        pass

    @abstractmethod
    def preprocess_sol(self, raw_sol):
        """Function to correctly format solution, used to make sure it can be cast to float

        :param str raw_sol: solution straight form tha dataset
        :return flaot : solution cast to float
        """
        pass

    def generate_prompt(self, entry_q):
        """Generates full prompt using the saved priming text and the entry's question

        :param str entry_q: entry question
        :return str: full prompt
        """
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

    def sample_n_for_prompting(self, nr_entries=1):
        rand_indexes = np.random.randint(0, len(self.data["body_list"]), nr_entries)

        sample_a_list = []
        sample_q_list = []
        for rand_index in rand_indexes:
            sample_q_list.append(
                f"Write a program that prints the answer to the following question. {self.data['body_list'][rand_index].text} {self.data['question_list'][rand_index].text}"
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


class gsm8k_datatset(math_dataset):
    def __init__(self, dataset_path, priming_text_pth, dataset_name):
        super().__init__(dataset_path, priming_text_pth, dataset_name)

    def load_dataset(self, dataset_path):
        with open(dataset_path) as fh:
            data = [json.loads(line) for line in fh.readlines() if line]
        return data

    def print_entry_from_idx(self, entry_idx):
        problem = self.data[entry_idx]
        print(colored(problem["question"], "yellow"))
        print(colored(problem["answer"], "green"))
        print(colored(re.findall(r"#### \w+", problem["answer"])[0][5:], "green"))
        print("\n" + "-" * 100 + "\n")

    def sample_n_for_prompting(self, nr_entries=1):
        rand_indexes = np.random.randint(0, len(self.data), nr_entries)

        sample_a_list = []
        sample_q_list = []
        for rand_index in rand_indexes:
            sample_q_list.append(
                "Write a program that prints the answer to the following question. "
                + self.data[rand_index]["question"]
            )
            sample_a_list.append(
                re.findall(r"#### \w+", self.data[rand_index]["answer"])[0][5:]
            )

        return sample_q_list, sample_a_list

    def preprocess_sol(self, raw_sol):
        sol = raw_sol
        try:
            sol = float(sol)
        except ValueError:
            sol = 22222222.0
        return float(sol)


class singleEq_dataset(math_dataset):
    def __init__(self, dataset_path, priming_text_pth, dataset_name):
        super().__init__(
            dataset_path,
            priming_text_pth,
            dataset_name,
            sample_func,
            preprocess_sol_func,
        )

    def load_dataset(self, dataset_path):
        with open(dataset_path, "r") as f:
            data = json.load(f)
        return data

    def print_entry_from_idx(self, entry_idx):
        problem = self.data[entry_idx]
        print(colored(problem["sQuestion"], "yellow"))
        print(colored(problem["lSolutions"][0], "green"))
        print("\n" + "-" * 100 + "\n")

    def sample_n_for_prompting(self, nr_entries=1):

        rand_indexes = np.random.randint(0, len(self.data), nr_entries)

        sample_a_list = []
        sample_q_list = []
        for rand_index in rand_indexes:
            question = self.data[rand_index]["sQuestion"]
            proc_question = (
                question[: question.rfind(".") + 1]
                + " Write a program that prints"
                + question[question.rfind(".") + 1 :]
            )
            sample_q_list.append(proc_question)
            sample_a_list.append(self.data[rand_index]["lSolutions"][0])

        return sample_q_list, sample_a_list

    def preprocess_sol(self, raw_sol):
        sol = raw_sol
        try:
            sol = float(sol)
        except ValueError:
            sol = 22222222.0
        return float(sol)


asdiv_path = "data/nlu-asdiv-dataset/dataset/ASDiv.xml"
gsm8k_path = "data/grade-school-math/grade_school_math/data/train.jsonl"
singleEq_path = "data/TACL2015/questions.json"


def init_dataset_from_name(datatset_name, primingtext_path):
    """General factory function for the math_dataset classes

    :param str datatset_name: name of the dataset
    :param str primingtext_path: path to prompt section
    :raises ValueError: if given wrong dataset name
    :return math_dataset: return the math_dataset object specified in the dataset name
    """

    if datatset_name == "asdiv":
        dataset_path = asdiv_path
        dataset = asdiv_dataset(dataset_path, primingtext_path, "asdiv")
    elif datatset_name == "gsm8k":
        dataset_path = gsm8k_path
        dataset = gsm8k_datatset(dataset_path, primingtext_path, "gsm8k")
    elif datatset_name == "singleEq":
        dataset_path = singleEq_path
        dataset = singleEq_dataset(dataset_path, primingtext_path, "singleEq")
    else:
        raise ValueError("dataset_name not recognized")

    return dataset
