import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import dataset_handler as dh
import testing_utils as tu

# from termcolor import colored
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("\n------------------\n")
# load pickle from results_lists/codegen
with open("results_lists/codegen/func_eq_short.txt_func_def_general.pkl", "rb") as f:
    general_eq_pass_at_k = pickle.load(f)

with open("results_lists/codegen/func_short.txt_func_def_general.pkl", "rb") as f:
    general_pass_at_k = pickle.load(f)

print(f"General - Pass@{3} = {np.mean(np.array(general_pass_at_k))}")
print(f"General eq - Pass@{3} = {np.mean(np.array(general_eq_pass_at_k))}")


print("\n------------------\n")

with open(f"results_lists/concepts/transfer_3.txt_func_def_general.pkl", "rb") as f:
    transfer_pass_at_k = pickle.load(f)
with open(
    f"results_lists/concepts/dimension_analysis_3.txt_func_def_general.pkl", "rb"
) as f:
    dimension_analysis_pass_at_k = pickle.load(f)
with open(
    f"results_lists/concepts/explicit_math_3.txt_func_def_general.pkl", "rb"
) as f:
    explicit_math_pass_at_k = pickle.load(f)
with open(f"results_lists/concepts/part-whole_3.txt_func_def_general.pkl", "rb") as f:
    part_whole_pass_at_k = pickle.load(f)


print(f"\nTransfer prompts - Pass@{3} = {np.mean(np.array(transfer_pass_at_k))}")
print(
    f"Dimension analysis prompts - Pass@{3} = {np.mean(np.array(dimension_analysis_pass_at_k))}"
)
print(
    f"Explicit math prompts - Pass@{3} = {np.mean(np.array(explicit_math_pass_at_k))}"
)
print(f"Part-whole prompts - Pass@{3} = {np.mean(np.array(part_whole_pass_at_k))}")

max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []
max_dict[4] = []


agg_pass_at_k = []

for i in range(len(general_pass_at_k)):
    cand_list = [
        general_pass_at_k[i],
        transfer_pass_at_k[i],
        dimension_analysis_pass_at_k[i],
        explicit_math_pass_at_k[i],
        part_whole_pass_at_k[i],
    ]
    max_cand = max(cand_list)
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)

print(
    f"\nBest possible Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)


with open(f"results_lists/concepts_eq/transfer_3.txt_func_def_general.pkl", "rb") as f:
    transfer_eq_pass_at_k = pickle.load(f)
with open(
    f"results_lists/concepts_eq/dimension_analysis_3.txt_func_def_general.pkl", "rb"
) as f:
    dimension_analysis_eq_pass_at_k = pickle.load(f)
with open(
    f"results_lists/concepts_eq/explicit_math_3.txt_func_def_general.pkl", "rb"
) as f:
    explicit_math_eq_pass_at_k = pickle.load(f)
with open(
    f"results_lists/concepts_eq/part-whole_3.txt_func_def_general.pkl", "rb"
) as f:
    part_whole_eq_pass_at_k = pickle.load(f)


print(f"\nTransfer prompts eq - Pass@{3} = {np.mean(np.array(transfer_eq_pass_at_k))}")
print(
    f"Dimension analysis prompts eq - Pass@{3} = {np.mean(np.array(dimension_analysis_eq_pass_at_k))}"
)
print(
    f"Explicit math prompts eq - Pass@{3} = {np.mean(np.array(explicit_math_eq_pass_at_k))}"
)
print(
    f"Part-whole prompts eq - Pass@{3} = {np.mean(np.array(part_whole_eq_pass_at_k))}"
)

max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []
max_dict[4] = []


agg_pass_at_k = []

for i in range(len(general_pass_at_k)):
    cand_list = [
        general_eq_pass_at_k[i],
        transfer_eq_pass_at_k[i],
        dimension_analysis_eq_pass_at_k[i],
        explicit_math_eq_pass_at_k[i],
        part_whole_eq_pass_at_k[i],
    ]
    max_cand = max(cand_list)
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)

print(
    f"\nBest possible Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)

print("\n------------------\n")

with open(f"results_lists/3_clusters/cluster_0.txt_func_def_general.pkl", "rb") as f:
    cluster_0_pass_at_k = pickle.load(f)
with open(f"results_lists/3_clusters/cluster_1.txt_func_def_general.pkl", "rb") as f:
    cluster_1_pass_at_k = pickle.load(f)
with open(f"results_lists/3_clusters/cluster_2.txt_func_def_general.pkl", "rb") as f:
    cluster_2_pass_at_k = pickle.load(f)

print(f"\ncluster_0 prompts - Pass@{3} = {np.mean(np.array(cluster_0_pass_at_k))}")
print(f"cluster_1 prompts - Pass@{3} = {np.mean(np.array(cluster_1_pass_at_k))}")
print(f"cluster_2 prompts - Pass@{3} = {np.mean(np.array(cluster_2_pass_at_k))}")

max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []

agg_pass_at_k = []

for i in range(len(general_pass_at_k)):
    cand_list = [
        general_pass_at_k[i],
        cluster_0_pass_at_k[i],
        cluster_1_pass_at_k[i],
        cluster_2_pass_at_k[i],
    ]
    max_cand = max(cand_list)
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)

print(
    f"\nBest possible Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)

priming_text_path = "data/priming_texts/gsm8k/codegen/func_eq_short.txt"
gsm8k_path = "data/grade-school-math/grade_school_math/data/train.jsonl"

current_dataset = dh.init_dataset_from_name("gsm8k", primingtext_path=priming_text_path)

tu.set_all_seeds()

rand_indexes = np.random.randint(0, len(current_dataset.data), 100)
# print(rand_indexes)

test_data = [current_dataset.data[i] for i in rand_indexes]
test_data = [elem["question"] for elem in test_data]

clustering_model_3 = pickle.load(open("clustering_model_3.pkl", "rb"))

embedder = SentenceTransformer("all-mpnet-base-v2")

# Corpus with example sentences
corpus = test_data
corpus_embeddings = embedder.encode(corpus)

labels = list(clustering_model_3.predict(corpus_embeddings))

agg_pass_at_k = []
max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []


for i, label in enumerate(labels):
    cand_list = [
        cluster_0_pass_at_k[i],
        cluster_1_pass_at_k[i],
        cluster_2_pass_at_k[i],
    ]
    max_cand = cand_list[label]
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)


print(
    f"\nClustering 3 Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)

with open(f"results_lists/3_clusters_eq/cluster_0.txt_func_def_general.pkl", "rb") as f:
    cluster_0_eq_pass_at_k = pickle.load(f)
with open(f"results_lists/3_clusters_eq/cluster_1.txt_func_def_general.pkl", "rb") as f:
    cluster_1_eq_pass_at_k = pickle.load(f)
with open(f"results_lists/3_clusters_eq/cluster_2.txt_func_def_general.pkl", "rb") as f:
    cluster_2_eq_pass_at_k = pickle.load(f)

print(
    f"\ncluster_0_eq prompts - Pass@{3} = {np.mean(np.array(cluster_0_eq_pass_at_k))}"
)
print(f"cluster_1_eq prompts - Pass@{3} = {np.mean(np.array(cluster_1_eq_pass_at_k))}")
print(f"cluster_2_eq prompts - Pass@{3} = {np.mean(np.array(cluster_2_eq_pass_at_k))}")

max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []

agg_pass_at_k = []

for i in range(len(general_pass_at_k)):
    cand_list = [
        general_eq_pass_at_k[i],
        cluster_0_eq_pass_at_k[i],
        cluster_1_eq_pass_at_k[i],
        cluster_2_eq_pass_at_k[i],
    ]
    max_cand = max(cand_list)
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)

print(
    f"\nBest possible Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)

priming_text_path = "data/priming_texts/gsm8k/codegen/func_eq_short.txt"
gsm8k_path = "data/grade-school-math/grade_school_math/data/train.jsonl"

current_dataset = dh.init_dataset_from_name("gsm8k", primingtext_path=priming_text_path)

tu.set_all_seeds()

rand_indexes = np.random.randint(0, len(current_dataset.data), 100)
# print(rand_indexes)

test_data = [current_dataset.data[i] for i in rand_indexes]
test_data = [elem["question"] for elem in test_data]

clustering_model_3 = pickle.load(open("clustering_model_3.pkl", "rb"))

embedder = SentenceTransformer("all-mpnet-base-v2")

# Corpus with example sentences
corpus = test_data
corpus_embeddings = embedder.encode(corpus)

labels = list(clustering_model_3.predict(corpus_embeddings))

agg_pass_at_k = []
max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []


for i, label in enumerate(labels):
    cand_list = [
        cluster_0_eq_pass_at_k[i],
        cluster_1_eq_pass_at_k[i],
        cluster_2_eq_pass_at_k[i],
    ]
    max_cand = cand_list[label]
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)


print(
    f"\nClustering 3 Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)

print("\n------------------\n")

with open(f"results_lists/4_clusters/cluster_0.txt_func_def_general.pkl", "rb") as f:
    cluster_0_pass_at_k = pickle.load(f)
with open(f"results_lists/4_clusters/cluster_1.txt_func_def_general.pkl", "rb") as f:
    cluster_1_pass_at_k = pickle.load(f)
with open(f"results_lists/4_clusters/cluster_2.txt_func_def_general.pkl", "rb") as f:
    cluster_2_pass_at_k = pickle.load(f)
with open(f"results_lists/4_clusters/cluster_3.txt_func_def_general.pkl", "rb") as f:
    cluster_3_pass_at_k = pickle.load(f)

print(f"\ncluster_0 prompts - Pass@{3} = {np.mean(np.array(cluster_0_pass_at_k))}")
print(f"cluster_1 prompts - Pass@{3} = {np.mean(np.array(cluster_1_pass_at_k))}")
print(f"cluster_2 prompts - Pass@{3} = {np.mean(np.array(cluster_2_pass_at_k))}")
print(f"cluster_3 prompts - Pass@{3} = {np.mean(np.array(cluster_3_pass_at_k))}")

max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []
max_dict[4] = []

agg_pass_at_k = []

for i in range(len(general_pass_at_k)):
    cand_list = [
        general_pass_at_k[i],
        cluster_0_pass_at_k[i],
        cluster_1_pass_at_k[i],
        cluster_2_pass_at_k[i],
        cluster_3_pass_at_k[i],
    ]
    max_cand = max(cand_list)
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)

print(
    f"\nBest possible Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)

priming_text_path = "data/priming_texts/gsm8k/codegen/func_eq_short.txt"
gsm8k_path = "data/grade-school-math/grade_school_math/data/train.jsonl"

current_dataset = dh.init_dataset_from_name("gsm8k", primingtext_path=priming_text_path)

tu.set_all_seeds()

rand_indexes = np.random.randint(0, len(current_dataset.data), 100)
# print(rand_indexes)

test_data = [current_dataset.data[i] for i in rand_indexes]
test_data = [elem["question"] for elem in test_data]

clustering_model_4 = pickle.load(open("clustering_model_3.pkl", "rb"))

embedder = SentenceTransformer("all-mpnet-base-v2")

# Corpus with example sentences
corpus = test_data
corpus_embeddings = embedder.encode(corpus)

labels = list(clustering_model_4.predict(corpus_embeddings))

agg_pass_at_k = []
max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []


for i, label in enumerate(labels):
    cand_list = [
        cluster_0_pass_at_k[i],
        cluster_1_pass_at_k[i],
        cluster_2_pass_at_k[i],
        cluster_3_pass_at_k[i],
    ]
    max_cand = cand_list[label]
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)


print(
    f"\nClustering 4 Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)

with open(f"results_lists/4_clusters_eq/cluster_0.txt_func_def_general.pkl", "rb") as f:
    cluster_0_eq_pass_at_k = pickle.load(f)
with open(f"results_lists/4_clusters_eq/cluster_1.txt_func_def_general.pkl", "rb") as f:
    cluster_1_eq_pass_at_k = pickle.load(f)
with open(f"results_lists/4_clusters_eq/cluster_2.txt_func_def_general.pkl", "rb") as f:
    cluster_2_eq_pass_at_k = pickle.load(f)
with open(f"results_lists/4_clusters_eq/cluster_3.txt_func_def_general.pkl", "rb") as f:
    cluster_3_eq_pass_at_k = pickle.load(f)

print(
    f"\ncluster_0_eq prompts - Pass@{3} = {np.mean(np.array(cluster_0_eq_pass_at_k))}"
)
print(f"cluster_1_eq prompts - Pass@{3} = {np.mean(np.array(cluster_1_eq_pass_at_k))}")
print(f"cluster_2_eq prompts - Pass@{3} = {np.mean(np.array(cluster_2_eq_pass_at_k))}")
print(f"cluster_3_eq prompts - Pass@{3} = {np.mean(np.array(cluster_3_eq_pass_at_k))}")

max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []
max_dict[4] = []

agg_pass_at_k = []

for i in range(len(general_pass_at_k)):
    cand_list = [
        general_eq_pass_at_k[i],
        cluster_0_eq_pass_at_k[i],
        cluster_1_eq_pass_at_k[i],
        cluster_2_eq_pass_at_k[i],
        cluster_3_eq_pass_at_k[i],
    ]
    max_cand = max(cand_list)
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)

print(
    f"\nBest possible Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)

priming_text_path = "data/priming_texts/gsm8k/codegen/func_eq_short.txt"
gsm8k_path = "data/grade-school-math/grade_school_math/data/train.jsonl"

current_dataset = dh.init_dataset_from_name("gsm8k", primingtext_path=priming_text_path)

tu.set_all_seeds()

rand_indexes = np.random.randint(0, len(current_dataset.data), 100)
# print(rand_indexes)

test_data = [current_dataset.data[i] for i in rand_indexes]
test_data = [elem["question"] for elem in test_data]

clustering_model_4 = pickle.load(open("clustering_model_4.pkl", "rb"))

embedder = SentenceTransformer("all-mpnet-base-v2")

# Corpus with example sentences
corpus = test_data
corpus_embeddings = embedder.encode(corpus)

labels = list(clustering_model_4.predict(corpus_embeddings))

agg_pass_at_k = []
max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []


for i, label in enumerate(labels):
    cand_list = [
        cluster_0_eq_pass_at_k[i],
        cluster_1_eq_pass_at_k[i],
        cluster_2_eq_pass_at_k[i],
        cluster_3_eq_pass_at_k[i],
    ]
    max_cand = cand_list[label]
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)


print(
    f"\nClustering 4 Aggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n"
)
