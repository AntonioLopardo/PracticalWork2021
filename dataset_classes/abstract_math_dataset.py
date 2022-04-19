from abc import ABC, abstractmethod
from xml.etree import ElementTree
import numpy as np
from termcolor import colored
from io import StringIO
from contextlib import redirect_stdout
import json
import re
import types


class math_dataset(ABC):
    @abstractmethod
    def __init__(
        self,
        dataset_path,
        priming_text_pth,
        dataset_name,
        sample_func=None,
        preprocess_sol_func=None,
        generate_prompt_func=None,
    ):
        """Initialize the dataset, loading the dataset priming text and setting the dataset name

        :param str dataset_path: path to the dataset
        :param str priming_text_pth: path to the priming text
        :param str dataset_name: name of the dataset
        """
        self.data = self.load_dataset(dataset_path)
        self.priming_text = self.load_priming_text(priming_text_pth)
        self.dataset_name = dataset_name
        if sample_func is not None:
            self.sample_n_for_prompting = types.MethodType(sample_func, self)
        if preprocess_sol_func is not None:
            self.preprocess_sol = types.MethodType(preprocess_sol_func, self)
        if generate_prompt_func is not None:
            self.generate_prompt = types.MethodType(generate_prompt_func, self)

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
