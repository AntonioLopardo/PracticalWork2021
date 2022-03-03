from transformers import AutoTokenizer, AutoModelForCausalLM
from xml.etree import ElementTree
import numpy as np
import json
import re

genji_model = ""
gptj_model = "EleutherAI/gpt-j-6B"
codeparrot_model = ""

asdiv_path = "data/nlu-asdiv-dataset/dataset/ASDiv.xml"
gsm8k_path = ""
singleEq_path = "" 

def read_string_from_file(path):
    with open(path, 'r') as f:
        return f.read()

# Choose the dataset you want to test
dataset_path = asdiv_path

# Load the priming text to add to the prompt
priming_text = read_string_from_file("data/priming_texts/asdiv.txt")

def sample_asdiv(dataset_path):
    dom = ElementTree.parse(dataset_path)

    #XML parsing
    body_list = dom.findall('ProblemSet/Problem/Body')
    answer_list = dom.findall('ProblemSet/Problem/Answer')
    question_list = dom.findall('ProblemSet/Problem/Question')
    formula_list = dom.findall('ProblemSet/Problem/Formula')
    stype_list = dom.findall('ProblemSet/Problem/Solution-Type')

    #Randomly choose a problem
    rand_index = np.random.randint(0, len(body_list))

    return f"{body_list[rand_index].text} {question_list[rand_index].text}", formula_list[rand_index].text

def sample_gsm8k(dataset_path):
    with open(dataset_path) as fh:
        data = [json.loads(line) for line in fh.readlines() if line]

    # Randomly choose a problem
    rand_index = np.random.randint(0, len(data))
    problem = data[rand_index]
    return problem['question'], re.findall(r"#### \w+", problem["answer"])[0][5:]

def sample_singleEq(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Randomly choose a problem
    rand_index = np.random.randint(0, len(data))
    problem = data[rand_index]
    return problem['sQuestion'], problem['lSolutions']

#sample_q, sample_a = sample_asdiv(dataset_path)
#sample_q, sample_a = sample_asdiv(dataset_path)
sample_q, sample_a = sample_asdiv(dataset_path)

prompt = f"{priming_text}\n\n#{sample_q}"
print(prompt)
print("\n" + "-"*100 + "\n")

tokenizer = AutoTokenizer.from_pretrained(gptj_model)
model = AutoModelForCausalLM.from_pretrained(gptj_model).eval().cuda()

tokens = tokenizer(prompt, return_tensors="pt").input_ids
generated_tokens = model.generate(tokens.long().cuda(), 
                                  use_cache=True, 
                                  do_sample=True, 
                                  top_k=50, 
                                  temperature=0.3, 
                                  top_p=0.9, 
                                  repetition_penalty=1.125, 
                                  min_length=1, 
                                  max_length=len(tokens[0]) + 100, 
                                  pad_token_id=tokenizer.eos_token_id)

last_tokens = generated_tokens[0][len(tokens[0]):]
generated_text = tokenizer.decode(last_tokens)
print("Generation:\n" + generated_text)
