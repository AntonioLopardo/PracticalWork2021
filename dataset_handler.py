from abc import ABC, abstractmethod
from xml.etree import ElementTree
import numpy as np
from termcolor import colored
from io import StringIO
from contextlib import redirect_stdout
import json
import re
import types
import sys

from dataset_classes.asdiv_dataset import asdiv_dataset
from dataset_classes.gsm8k_dataset import gsm8k_dataset
from dataset_classes.singleEq_dataset import singleEq_dataset

asdiv_path = "data/nlu-asdiv-dataset/dataset/ASDiv.xml"
gsm8k_path = "data/grade-school-math/grade_school_math/data/train.jsonl"
gsm8k_socratic_path = (
    "data/grade-school-math/grade_school_math/data/train_socratic.jsonl"
)
gsm8k_test_path = "data/grade-school-math/grade_school_math/data/test.jsonl"
singleEq_path = "data/TACL2015/questions.json"


def init_dataset_from_name(
    datatset_name,
    primingtext_path,
    sample_func=None,
    preprocess_sol_func=None,
    generate_prompt_func=None,
):
    """General factory function for the math_dataset classes

    :param str datatset_name: name of the dataset
    :param str primingtext_path: path to prompt section
    :raises ValueError: if given wrong dataset name
    :return math_dataset: return the math_dataset object specified in the dataset name
    """

    if datatset_name == "asdiv":
        dataset_path = asdiv_path
        dataset = asdiv_dataset(
            dataset_path,
            primingtext_path,
            "asdiv",
            sample_func,
            preprocess_sol_func,
            generate_prompt_func,
        )
    elif datatset_name == "gsm8k":
        dataset_path = gsm8k_path
        dataset = gsm8k_dataset(
            dataset_path,
            primingtext_path,
            "gsm8k",
            sample_func,
            preprocess_sol_func,
            generate_prompt_func,
        )
    elif datatset_name == "gsm8k-test":
        dataset_path = gsm8k_test_path
        dataset = gsm8k_dataset(
            dataset_path,
            primingtext_path,
            "gsm8k",
            sample_func,
            preprocess_sol_func,
            generate_prompt_func,
        )
    elif datatset_name == "gsm8k-socratic":
        dataset_path = gsm8k_socratic_path
        dataset = gsm8k_dataset(
            dataset_path,
            primingtext_path,
            "gsm8k",
            sample_func,
            preprocess_sol_func,
            generate_prompt_func,
        )
    elif datatset_name == "singleEq":
        dataset_path = singleEq_path
        dataset = singleEq_dataset(
            dataset_path,
            primingtext_path,
            "singleEq",
            sample_func,
            preprocess_sol_func,
            generate_prompt_func,
        )
    else:
        raise ValueError("dataset_name not recognized")

    return dataset
