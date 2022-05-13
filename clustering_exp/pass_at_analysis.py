import pickle
import numpy as np
from termcolor import colored

pkl_dir = "pass_at_list/"

with open(f"{pkl_dir}cluster_0_pass_at_k.pkl", "rb") as f:
    cluster_0_pass_at_k = pickle.load(f)
with open(f"{pkl_dir}cluster_1_pass_at_k.pkl", "rb") as f:
    cluster_1_pass_at_k = pickle.load(f)
with open(f"{pkl_dir}cluster_2_pass_at_k.pkl", "rb") as f:
    cluster_2_pass_at_k = pickle.load(f)
with open(f"{pkl_dir}cluster_3_pass_at_k.pkl", "rb") as f:
    cluster_3_pass_at_k = pickle.load(f)
with open(f"{pkl_dir}general_pass_at_k.pkl", "rb") as f:
    general_pass_at_k = pickle.load(f)

general_pass_at_k = [0.0 for _ in range(len(general_pass_at_k))]

print(f"\ncluster_0 prompts - Pass@{3} = {np.mean(np.array(cluster_0_pass_at_k))}")
print(f"cluster_1 prompts - Pass@{3} = {np.mean(np.array(cluster_1_pass_at_k))}")
print(f"cluster_2 prompts - Pass@{3} = {np.mean(np.array(cluster_2_pass_at_k))}")
print(f"cluster_3 prompts - Pass@{3} = {np.mean(np.array(cluster_3_pass_at_k))}")
print(f"General prompts - Pass@{3} = {np.mean(np.array(general_pass_at_k))}")

max_dict = {}
max_dict[0] = []
max_dict[1] = []
max_dict[2] = []
max_dict[3] = []
max_dict[4] = []

diff_from_best_dict = {}
diff_from_best_dict[0] = []
diff_from_best_dict[1] = []
diff_from_best_dict[2] = []
diff_from_best_dict[3] = []
diff_from_best_dict[4] = []

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
    for j in range(len(cand_list)):
        diff_from_best_dict[j].append(abs(cand_list[j] - max_cand))
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)

print(f"\nAggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n")
print(f"General: {len(max_dict[0])}")
print(f"cluster_0: {len(max_dict[1])}")
print(f"cluster_1: {len(max_dict[2])}")
print(f"cluster_2: {len(max_dict[3])}")
print(f"cluster_3: {len(max_dict[4])}")

print(
    f"\nGeneral Best - Pass@{3} = {np.mean(np.array(general_pass_at_k)[max_dict[0]])}"
)
print(
    f"cluster_0 Best - Pass@{3} = {np.mean(np.array(cluster_0_pass_at_k)[max_dict[1]])}"
)
print(
    f"cluster_1 Best - Pass@{3} = {np.mean(np.array(cluster_1_pass_at_k)[max_dict[2]])}"
)
print(
    f"cluster_2 Best - Pass@{3} = {np.mean(np.array(cluster_2_pass_at_k)[max_dict[3]])}"
)
print(
    f"cluster_3 Best - Pass@{3} = {np.mean(np.array(cluster_3_pass_at_k)[max_dict[4]])}"
)

not_general_list = max_dict[1] + max_dict[2] + max_dict[3] + max_dict[4]
not_transfer_list = max_dict[0] + max_dict[2] + max_dict[3] + max_dict[4]
not_dimension_list = max_dict[0] + max_dict[1] + max_dict[3] + max_dict[4]
not_explicit_list = max_dict[0] + max_dict[1] + max_dict[2] + max_dict[4]
not_part_whole_list = max_dict[0] + max_dict[1] + max_dict[2] + max_dict[3]

print(
    f"\nGeneral worse - Pass@{3} = {np.mean(np.array(general_pass_at_k)[not_general_list])}"
)
print(
    f"cluster_0 worse - Pass@{3} = {np.mean(np.array(cluster_0_pass_at_k)[not_transfer_list])}"
)
print(
    f"cluster_1 worse - Pass@{3} = {np.mean(np.array(cluster_0_pass_at_k)[not_dimension_list])}"
)
print(
    f"cluster_2 worse - Pass@{3} = {np.mean(np.array(cluster_0_pass_at_k)[not_explicit_list])}"
)
print(
    f"cluster_3 worse - Pass@{3} = {np.mean(np.array(cluster_0_pass_at_k)[not_part_whole_list])}"
)

print(
    f"\nGeneral Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[0]))}"
)
print(
    f"cluster_0 Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[1]))}"
)
print(
    f"cluster_1 Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[2]))}"
)
print(
    f"cluster_2 Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[3]))}"
)
print(
    f"cluster_3 Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[4]))}"
)
