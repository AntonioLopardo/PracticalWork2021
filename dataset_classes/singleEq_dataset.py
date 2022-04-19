from abc import ABC, abstractmethod
from xml.etree import ElementTree
import numpy as np
from termcolor import colored
from io import StringIO
from contextlib import redirect_stdout
import json
import re
import types

from dataset_classes.abstract_math_dataset import math_dataset


class singleEq_dataset(math_dataset):
    def __init__(
        self,
        dataset_path,
        priming_text_pth,
        dataset_name,
        sample_func=None,
        preprocess_sol_func=None,
        generate_prompt_func=None,
    ):
        super().__init__(
            dataset_path,
            priming_text_pth,
            dataset_name,
            sample_func,
            preprocess_sol_func,
            generate_prompt_func,
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
