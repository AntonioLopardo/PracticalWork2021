from xml.etree import ElementTree
from collections import Counter
from random import sample
import random
import numpy as np
import json
import re
from termcolor import colored
import os


def load_gsm8k(dataset_path):
    with open(dataset_path) as fh:
        data = [json.loads(line) for line in fh.readlines() if line]

    return data


# 1 for Transfer
# 2 for Dimension Analysis
# 3 for Explicit Math
# 4 for Part-Whole Relations
# 0 for quit
# \ for uncertain

data = load_gsm8k("data/grade-school-math/grade_school_math/data/train.jsonl")

transfer_problem_list = []
dimension_problem_list = []
explicit_problem_list = []
part_whole_problem_list = []

for idx, problem in enumerate(data[:100]):
    print(f"\n{idx}---------------------\n" + problem["question"])
    label = input("------Classify the quesiton------\n")
    print(label)
    if label == "1":
        transfer_problem_list.append(idx)
        print(colored("Transfer", "green"))
    elif label == "2":
        dimension_problem_list.append(idx)
        print(colored("Dimension Analysis", "green"))
    elif label == "3":
        explicit_problem_list.append(idx)
        print(colored("Explicit Math", "green"))
    elif label == "4":
        part_whole_problem_list.append(idx)
        print(colored("Part-Whole Relations", "green"))
    elif label == "0":
        break
    elif label == "\\":
        print("Uncertain")
