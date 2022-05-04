import pickle
import numpy as np
from termcolor import colored

pkl_dir = "pass_at_list/"

with open(f"{pkl_dir}transfer_pass_at_k.pkl", "rb") as f:
    transfer_pass_at_k = pickle.load(f)
with open(f"{pkl_dir}dimension_analysis_pass_at_k.pkl", "rb") as f:
    dimension_analysis_pass_at_k = pickle.load(f)
with open(f"{pkl_dir}explicit_math_pass_at_k.pkl", "rb") as f:
    explicit_math_pass_at_k = pickle.load(f)
with open(f"{pkl_dir}part_whole_pass_at_k.pkl", "rb") as f:
    part_whole_pass_at_k = pickle.load(f)
with open(f"{pkl_dir}general_pass_at_k.pkl", "rb") as f:
    general_pass_at_k = pickle.load(f)

print(f"\nTransfer prompts - Pass@{3} = {np.mean(np.array(transfer_pass_at_k))}")
print(
    f"Dimension analysis prompts - Pass@{3} = {np.mean(np.array(dimension_analysis_pass_at_k))}"
)
print(
    f"Explicit math prompts - Pass@{3} = {np.mean(np.array(explicit_math_pass_at_k))}"
)
print(f"Part-whole prompts - Pass@{3} = {np.mean(np.array(part_whole_pass_at_k))}")
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
        transfer_pass_at_k[i],
        dimension_analysis_pass_at_k[i],
        explicit_math_pass_at_k[i],
        part_whole_pass_at_k[i],
    ]
    max_cand = max(cand_list)
    for j in range(len(cand_list)):
        diff_from_best_dict[j].append(abs(cand_list[j] - max_cand))
    max_dict[cand_list.index(max_cand)].append(i)
    agg_pass_at_k.append(max_cand)

print(f"\nAggregate prompts - Pass@{3} = {np.mean(np.array(agg_pass_at_k))}\n")
print(f"General: {len(max_dict[0])}")
print(f"Transfer: {len(max_dict[1])}")
print(f"Dimension: {len(max_dict[2])}")
print(f"Explicit: {len(max_dict[3])}")
print(f"Part-Whole: {len(max_dict[4])}")

print(
    f"\nGeneral Best - Pass@{3} = {np.mean(np.array(general_pass_at_k)[max_dict[0]])}"
)
print(
    f"Transfer Best - Pass@{3} = {np.mean(np.array(transfer_pass_at_k)[max_dict[1]])}"
)
print(
    f"Dimension Best - Pass@{3} = {np.mean(np.array(dimension_analysis_pass_at_k)[max_dict[2]])}"
)
print(
    f"Explicit Best - Pass@{3} = {np.mean(np.array(explicit_math_pass_at_k)[max_dict[3]])}"
)
print(
    f"Part-Whole Best - Pass@{3} = {np.mean(np.array(part_whole_pass_at_k)[max_dict[4]])}"
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
    f"Transfer worse - Pass@{3} = {np.mean(np.array(transfer_pass_at_k)[not_transfer_list])}"
)
print(
    f"Dimension worse - Pass@{3} = {np.mean(np.array(dimension_analysis_pass_at_k)[not_dimension_list])}"
)
print(
    f"Explicit worse - Pass@{3} = {np.mean(np.array(explicit_math_pass_at_k)[not_explicit_list])}"
)
print(
    f"Part-Whole worse - Pass@{3} = {np.mean(np.array(part_whole_pass_at_k)[not_part_whole_list])}"
)

print(
    f"\nGeneral Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[0]))}"
)
print(
    f"Transfer Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[1]))}"
)
print(
    f"Dimension Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[2]))}"
)
print(
    f"Explicit Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[3]))}"
)
print(
    f"Part-Whole Diff from Best - Pass@{3} = {np.mean(np.array(diff_from_best_dict[4]))}"
)
