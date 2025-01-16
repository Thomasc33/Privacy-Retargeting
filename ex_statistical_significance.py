#!/usr/bin/env python3
"""
combined_significance_script.py

A single, self-contained example that:
1) Loads NTU skeleton data (X_full.pkl).
2) Loads PMR (val_model) and DMR models + weights.
3) Performs multiple-run retargeting (with different seeds).
4) Evaluates action recognition and re-ID using SGN.
5) Does T-tests and prints a LaTeX table snippet for your paper.

You may need to adapt certain paths, class imports, or shapes
depending on your local setup and data organization.
"""

import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import stats
from SGN.model import SGN
from SGN.data import NTUDataLoaders, AverageMeter

# -----------------------------
# 1) Some of your SGN code / placeholders
# -----------------------------
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)



# The run_sgn_eval function from your snippet.
def test(test_loader, model, k=3):
    acces = AverageMeter()
    topk_acces = AverageMeter()
    model.eval()

    label_output = list()
    pred_output = list()

    for i, t in enumerate(test_loader):
        inputs = t[0]
        target = t[1]
        with torch.no_grad():
            output = model(inputs.cuda())
            # [batch, time, classes], or shape specifics depends on your SGN definition
            # We'll assume shape: (batch, seg, classes)
            # Flatten out to average across seg dimension:
            output = output.view(
                (-1, inputs.size(0)//target.size(0), output.size(1))
            )
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy(output.data, target.cuda())
        acces.update(acc[0], inputs.size(0))

        topk_acc = top_k_accuracy(output.data, target.cuda(), k=k)
        topk_acces.update(topk_acc[0], inputs.size(0))

    label_output = np.concatenate(label_output, axis=0)
    pred_output = np.concatenate(pred_output, axis=0)

    label_index = np.argmax(label_output, axis=1)
    pred_index = np.argmax(pred_output, axis=1)

    f1 = f1_score(label_index, pred_index, average='macro', zero_division=0)
    precision = precision_score(label_index, pred_index, average='macro', zero_division=0)
    recall = recall_score(label_index, pred_index, average='macro', zero_division=0)

    return acces.avg, f1, precision, recall, topk_acces.avg

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    # Convert one-hot targets to indices:
    target = torch.argmax(target, dim=1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    return correct.mul_(100.0 / batch_size)

def top_k_accuracy(output, target, k=3):
    batch_size = target.size(0)
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    target = torch.argmax(target, dim=1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

def run_sgn_eval(train_x, train_y, test_x, test_y, val_x, val_y, case, model, k=3):
    # This relies on a real NTUDataLoaders definition to produce a test_loader:
    ntu_loaders = NTUDataLoaders("NTU", case, seg=20,
                                 train_X=train_x, train_Y=train_y,
                                 test_X=test_x,  test_Y=test_y,
                                 val_X=val_x,    val_Y=val_y,
                                 aug=0)
    test_loader = ntu_loaders.get_test_loader(64, 16)
    return test(test_loader, model, k=k)

# -----------------------------
# 2) Other user variables
# -----------------------------
batch_size = 64
utility_classes = 60  # NTU-60
privacy_classes = 40
T = 75
k = 5  # for top-5 accuracy
ntu_120 = False
only_ntu_120 = False
only_use_pos = True

# Dummy placeholders for SGN-related arrays
sgn_train_x = np.zeros((batch_size, 300, 150), dtype=np.float32)
sgn_train_y = np.zeros((batch_size, 60), dtype=np.float32)  # one-hot for action
sgn_val_x   = np.zeros((batch_size, 300, 150), dtype=np.float32)
sgn_val_y   = np.zeros((batch_size, 60), dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sgn_ar = SGN(utility_classes, None, 20, batch_size, 0).to(device)
sgn_priv = SGN(privacy_classes, None, 20, batch_size, 0).to(device)

sgn_ar.load_state_dict(torch.load('SGN/pretrained/action_60_sgnpt.pt')['state_dict'])
sgn_priv.load_state_dict(torch.load('SGN/pretrained/privacy_60_sgnpt.pt')['state_dict'])


# -----------------------------
# 3) Models: PMR vs. DMR
#    (from model_ae import AutoEncoder, from dmr_model import DMR)
# -----------------------------
from model_ae import AutoEncoder
from dmr_model import DMR

# Instantiate PMR:
val_model = AutoEncoder(use_adv=False).to(device)
weights = torch.load('pretrained/MR.pt', map_location=device)
val_model.load_state_dict(weights, strict=False)

# Instantiate DMR:
dmr = DMR().to(device)
dmr_weights = torch.load('pretrained/DMR.pt', map_location=device)
dmr.load_state_dict(dmr_weights, strict=False)

# -----------------------------
# 4) Load X (the dictionary of skeleton data)
#    Key = filename (bytes or str), Value = torch.Tensor shape (T, J, 3?), etc.
# -----------------------------
with open('ntu/SGN/X_full.pkl', 'rb') as f:
    X = pickle.load(f)

# pad/trim data to T frames and convert to tensor
for file, value in X.items():
    # If SGN preprocessing, remove zero padding
    if only_use_pos:
        first_zero_index = value.shape[0]
        for i in range(value.shape[0]):
            if np.all(value[i] == 0):
                first_zero_index = i
                break
        value = value[:first_zero_index]

    num_frames = value.shape[0]

    # Pad or trim
    if num_frames < T:
        if only_use_pos: padding = np.repeat(value[-1][np.newaxis, :], T - num_frames, axis=0)
        else: padding = np.repeat(value[-1][np.newaxis, :, :], T - num_frames, axis=0)
        value = np.concatenate((value, padding), axis=0)
    elif num_frames > T:
        # Randomly sample T frames
        start = random.randint(0, num_frames - T)
        value = value[start:start+T]
    
    # Convert to tensor and store back
    X[file] = torch.from_numpy(value).float()

if not ntu_120:
    to_rem = []
    for file in X.keys():
        if int(str(file).split('A')[1][:3]) > 60:
            to_rem.append(file)
    for file in to_rem:
        del X[file]

if only_ntu_120:
    to_rem = []
    for file in X.keys():
        if int(str(file).split('A')[1][:3]) <= 60:
            to_rem.append(file)
    for file in to_rem:
        del X[file]

# chop off second actor and convert to 3d
if only_use_pos:
    for file, value in X.items():
        value = value[:, :75]
        value = value.view(-1, 25, 3)
        X[file] = value

# remove two actor actions
two_action_files = set([50,51,52,53,54,55,56,57,58,59,60,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120])
if True:
    to_rem = []
    for file in X.keys():
        if int(str(file).split('A')[1][:3]) in two_action_files:
            to_rem.append(file)
    for file in to_rem:
        del X[file]

# -----------------------------
# 5) parse_file_name function
# -----------------------------
def parse_file_name(file_name):
    file_name = str(file_name)
    if file_name[0] == 'b':  # SGN preprocessing (bytes-literal)
        S = int(file_name[3:6])
        C = int(file_name[7:10])
        P = int(file_name[11:14])
        R = int(file_name[15:18])
        A = int(file_name[19:22])
    else:
        S = int(file_name[1:4])
        C = int(file_name[5:8])
        P = int(file_name[9:12])
        R = int(file_name[13:16])
        A = int(file_name[17:20])
    return {'S': S, 'C': C, 'P': P, 'R': R, 'A': A}

# -----------------------------
# 6) Retargeting Functions (Random Action, Constant Action)
#    We'll remove rendering calls.
# -----------------------------
def retarget_random_action(
    model,
    out_random_path='results/X_hat_random_RA.pkl',
    out_constant_path='results/X_hat_constant_RA.pkl'
):
    """
    Similar to your original function but without rendering.
    """
    os.makedirs(os.path.dirname(out_random_path), exist_ok=True)

    X_hat_random = {}
    X_hat_constant = {}

    # pick a "constant" skeleton
    const_key = 'S007C001P025R001A045'.encode('utf-8')
    x2_const = X[const_key].float().cuda().unsqueeze(0)

    times = []
    with torch.no_grad():
        for file in tqdm(X, desc="Retarget random_action"):
            x1 = X[file].unsqueeze(0).float().cuda()

            # pick a random skeleton with different actor
            while True:
                sample = random.sample(list(X.keys()), 1)[0]
                if sample.decode('utf-8')[9:12] != file.decode('utf-8')[9:12]:
                    break
            x2_random = X[sample].unsqueeze(0).float().cuda()

            start = time.time()
            X_hat_random[file] = model.eval(x1, x2_random).cpu().numpy().squeeze()
            times.append(time.time() - start)

            start = time.time()
            X_hat_constant[file] = model.eval(x1, x2_const).cpu().numpy().squeeze()
            times.append(time.time() - start)

    print(f"[retarget_random_action] Mean time: {np.mean(times):.4f} s")

    # Save
    with open(out_random_path, 'wb') as f:
        pickle.dump(X_hat_random, f)
    with open(out_constant_path, 'wb') as f:
        pickle.dump(X_hat_constant, f)

def retarget_constant_action(
    model,
    out_random_path='results/X_hat_random_CA.pkl',
    out_constant_path='results/X_hat_constant_CA.pkl'
):
    """
    Similar to your original function but without rendering.
    """
    os.makedirs(os.path.dirname(out_random_path), exist_ok=True)

    X_hat_random = {}
    X_hat_constant = {}

    const_actor_id = 8
    times = []

    # dictionary of {action: skeleton} for that constant actor
    const_dict = {}
    for file in X:
        info = parse_file_name(file)
        if info['P'] == const_actor_id and info['A'] not in const_dict:
            const_dict[info['A']] = X[file].float().cuda().unsqueeze(0)

    with torch.no_grad():
        for file in tqdm(X, desc="Retarget constant_action"):
            x1 = X[file].unsqueeze(0).float().cuda()
            info = parse_file_name(file)

            # pick random skeleton with same action but different actor
            while True:
                sample = random.sample(list(X.keys()), 1)[0]
                info_ = parse_file_name(sample)
                if info_['P'] != info['P'] and info_['A'] == info['A']:
                    break
            x2_random = X[sample].unsqueeze(0).float().cuda()

            start = time.time()
            X_hat_random[file] = model.eval(x1, x2_random).cpu().numpy().squeeze()
            times.append(time.time() - start)

            # retarget to the "constant" actor for that action
            if info['A'] in const_dict:
                x2_const = const_dict[info['A']]
            else:
                # fallback if missing (rare)
                x2_const = list(const_dict.values())[0]

            start = time.time()
            X_hat_constant[file] = model.eval(x1, x2_const).cpu().numpy().squeeze()
            times.append(time.time() - start)

    print(f"[retarget_constant_action] Mean time: {np.mean(times):.4f} s")

    with open(out_random_path, 'wb') as f:
        pickle.dump(X_hat_random, f)
    with open(out_constant_path, 'wb') as f:
        pickle.dump(X_hat_constant, f)

# -----------------------------
# 7) Multiple runs
#    We do 5 runs for each scenario (PMR random, PMR constant, DMR random, DMR constant)
#    Then we evaluate & store results for T-tests
# -----------------------------

def create_multiple_runs_random_action(model, tag="PMR", num_runs=5):
    """
    Creates multiple runs for random_action retargeting
    with different seeds, storing each run in unique files.
    """
    for run_idx in range(1, num_runs+1):
        seed_val = 100 + run_idx
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        out_random_path   = f"results/{tag}_X_hat_random_RA_run{run_idx}.pkl"
        out_constant_path = f"results/{tag}_X_hat_constant_RA_run{run_idx}.pkl"

        print(f"\n--- {tag} RANDOM ACTION: RUN {run_idx}, SEED={seed_val} ---")
        retarget_random_action(model, out_random_path, out_constant_path)

def create_multiple_runs_constant_action(model, tag="PMR", num_runs=5):
    """
    Creates multiple runs for constant_action retargeting
    with different seeds, storing each run in unique files.
    """
    for run_idx in range(1, num_runs+1):
        seed_val = 200 + run_idx
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        out_random_path   = f"results/{tag}_X_hat_random_CA_run{run_idx}.pkl"
        out_constant_path = f"results/{tag}_X_hat_constant_CA_run{run_idx}.pkl"

        print(f"\n--- {tag} CONSTANT ACTION: RUN {run_idx}, SEED={seed_val} ---")
        retarget_constant_action(model, out_random_path, out_constant_path)

# -----------------------------
# 8) Evaluate each run with SGN (Action + ReID).
#    We'll store top-1, top-5 for each run, then do T-tests and
#    print out a final table snippet.
# -----------------------------

def evaluate_pkl_file(pkl_path, name="PMR"):
    """
    Evaluate a given .pkl retargeted dictionary with SGN for action & reID.
    Return (top1_action, top5_action, top1_reid, top5_reid).
    We'll just do placeholders if needed.
    """
    if not os.path.exists(pkl_path):
        print(f"[Warning] file not found: {pkl_path}")
        return None

    with open(pkl_path, 'rb') as f:
        anonym_dict = pickle.load(f)

    # TODO: Actually feed anonym_dict into run_sgn_eval with sgn_ar, etc.
    # For now, we'll do placeholders or minimal logic.
    # If you prefer the 'eval(X_dict, ...)' approach, you could call it here instead.
    # But let's do a dummy approach with run_sgn_eval:

    x = np.zeros((len(anonym_dict), 300, 150), dtype=np.float32)
    y_util = np.zeros(len(anonym_dict))
    y_priv = np.zeros(len(anonym_dict))

    for i, file in enumerate(anonym_dict):
        if anonym_dict[file].shape[1] == 75:
            anonym_dict[file] = np.pad(anonym_dict[file], ((0, 0), (0, 75)), 'constant')

        if anonym_dict[file].shape[0] < 300:
            padding = np.zeros((300 - anonym_dict[file].shape[0], 150))
            anonym_dict[file] = np.concatenate((anonym_dict[file], padding), axis=0)

        x[i] = np.array(anonym_dict[file], dtype=np.float32)
        y_util[i] = int(file[17:20])
        y_priv[i] = int(file[9:12])

    y_util = y_util - 1
    y_priv = y_priv - 1
    y_util = np.eye(utility_classes)[y_util.astype(int)]
    y_priv = np.eye(privacy_classes)[y_priv.astype(int)]

    # Just call run_sgn_eval for action
    acc_action, f1_action, prec_action, rec_action, topk_action = run_sgn_eval(
        sgn_train_x, sgn_train_y, x, y_util, sgn_val_x, sgn_val_y, 1, sgn_ar, k=5
    )
    # For reID
    acc_reid, f1_reid, prec_reid, rec_reid, topk_reid = run_sgn_eval(
        sgn_train_x, sgn_train_y, x, y_priv, sgn_val_x, sgn_val_y, 1, sgn_priv, k=5
    )

    return acc_action, topk_action, acc_reid, topk_reid


def main():
    # 1) Create multiple runs (uncomment if you actually want to generate them)
    # create_multiple_runs_random_action(dmr, tag="DMR", num_runs=5)
    # create_multiple_runs_constant_action(dmr, tag="DMR", num_runs=5)

    # create_multiple_runs_random_action(val_model, tag="PMR", num_runs=5)
    # create_multiple_runs_constant_action(val_model, tag="PMR", num_runs=5)

    # 2) Evaluate each run. We'll gather 4 sets of runs:
    #    - DMR_random_RA_runX
    #    - DMR_random_CA_runX
    #    - PMR_random_RA_runX
    #    - PMR_random_CA_runX
    # For simplicity, let's just do the random_RA scenario in detail.

    # We'll store results in a dictionary: method_group -> lists of (action_top1, action_top5, reid_top1, reid_top5).
    method_group_results = {}

    # Helper to accumulate results
    def add_result(group, r):
        if r is None:
            return
        if group not in method_group_results:
            method_group_results[group] = []
        method_group_results[group].append(r)  # (acc_action, topk_action, acc_reid, topk_reid)

    # Evaluate 5 runs of DMR random RA
    for run_idx in range(1, 6):
        path_random = f"results/DMR_X_hat_random_RA_run{run_idx}.pkl"
        r_random = evaluate_pkl_file(path_random, name="DMR_random_RA")
        add_result("DMR_random_RA", r_random)

    # Evaluate 5 runs of DMR constant RA
    for run_idx in range(1, 6):
        path_constant = f"results/DMR_X_hat_constant_RA_run{run_idx}.pkl"
        r_constant = evaluate_pkl_file(path_constant, name="DMR_constant_RA")
        add_result("DMR_constant_RA", r_constant)

    # Evaluate 5 runs of DMR random CA
    for run_idx in range(1, 6):
        path_random = f"results/DMR_X_hat_random_CA_run{run_idx}.pkl"
        r_random = evaluate_pkl_file(path_random, name="DMR_random_CA")
        add_result("DMR_random_CA", r_random)

    # Evaluate 5 runs of DMR constant CA
    for run_idx in range(1, 6):
        path_constant = f"results/DMR_X_hat_constant_CA_run{run_idx}.pkl"
        r_constant = evaluate_pkl_file(path_constant, name="DMR_constant_CA")
        add_result("DMR_constant_CA", r_constant)

    # Evaluate 5 runs of PMR random RA
    for run_idx in range(1, 6):
        path_random = f"results/PMR_X_hat_random_RA_run{run_idx}.pkl"
        r_random = evaluate_pkl_file(path_random, name="PMR_random_RA")
        add_result("PMR_random_RA", r_random)

    # Evaluate 5 runs of PMR constant RA
    for run_idx in range(1, 6):
        path_constant = f"results/PMR_X_hat_constant_RA_run{run_idx}.pkl"
        r_constant = evaluate_pkl_file(path_constant, name="PMR_constant_RA")
        add_result("PMR_constant_RA", r_constant)

    # Evaluate 5 runs of PMR random CA
    for run_idx in range(1, 6):
        path_random = f"results/PMR_X_hat_random_CA_run{run_idx}.pkl"
        r_random = evaluate_pkl_file(path_random, name="PMR_random_CA")
        add_result("PMR_random_CA", r_random)

    # Evaluate 5 runs of PMR constant CA
    for run_idx in range(1, 6):
        path_constant = f"results/PMR_X_hat_constant_CA_run{run_idx}.pkl"
        r_constant = evaluate_pkl_file(path_constant, name="PMR_constant_CA")
        add_result("PMR_constant_CA", r_constant)

    # (Similarly you can do it for constant_RA, constant_CA, random_CA, etc. as needed.)

    # 3) Calculate mean ± std, do T-tests, and produce a LaTeX table snippet.

    # Suppose we only compare "DMR_random_RA" vs. "PMR_random_RA".
    # We can do the t-test on top-1 action and top-1 reID. 
    # If you want top-5 as well, do the same.

    def compute_stats(arr):
        """
        arr: list of tuples (action_top1, action_top5, reid_top1, reid_top5)
        Return: means, stds for each
        """
        arr = np.array(arr)  # shape [num_runs, 4]
        mean_top1_action = arr[:, 0].mean()
        std_top1_action  = arr[:, 0].std()
        mean_top5_action = arr[:, 1].mean()
        std_top5_action  = arr[:, 1].std()
        mean_top1_reid   = arr[:, 2].mean()
        std_top1_reid    = arr[:, 2].std()
        mean_top5_reid   = arr[:, 3].mean()
        std_top5_reid    = arr[:, 3].std()
        return (mean_top1_action, std_top1_action,
                mean_top5_action, std_top5_action,
                mean_top1_reid,  std_top1_reid,
                mean_top5_reid,  std_top5_reid)
    
    # 3) Print results for each method group so you can place them in your main table
    print("\n=== Aggregate Results (Mean ± Std) ===")
    for group_name in sorted(method_group_results.keys()):
        stats = compute_stats(method_group_results[group_name])
        (mA1, sA1, mA5, sA5, mR1, sR1, mR5, sR5) = stats
        print(f"Method: {group_name}")
        print(f"  Action Top-1: {mA1:.4f} ± {sA1:.4f}")
        print(f"  Action Top-5: {mA5:.4f} ± {sA5:.4f}")
        print(f"  Re-ID  Top-1: {mR1:.4f} ± {sR1:.4f}")
        print(f"  Re-ID  Top-5: {mR5:.4f} ± {sR5:.4f}\n")

    # def print_table_line(method_name, stats_tuple):
    #     (mA1, sA1, mA5, sA5, mR1, sR1, mR5, sR5) = stats_tuple
    #     print(f"{method_name:<10} & ${mA1:.3f} \\pm {sA1:.3f}$ & ${mA5:.3f} \\pm {sA5:.3f}$ & ${mR1:.3f} \\pm {sR1:.3f}$ & ${mR5:.3f} \\pm {sR5:.3f}$ \\\\")

    # Gather stats
    # dmr_stats = compute_stats(method_group_results["DMR_random_RA"])
    # pmr_stats = compute_stats(method_group_results["PMR_random_RA"])

    # # T-test on top-1 action
    # dmr_action_top1 = np.array([r[0] for r in method_group_results["DMR_random_RA"]])
    # pmr_action_top1 = np.array([r[0] for r in method_group_results["PMR_random_RA"]])
    # t_stat_action, p_val_action = stats.ttest_ind(dmr_action_top1, pmr_action_top1, equal_var=False)

    # # T-test on top-1 reID
    # dmr_reid_top1 = np.array([r[2] for r in method_group_results["DMR_random_RA"]])
    # pmr_reid_top1 = np.array([r[2] for r in method_group_results["PMR_random_RA"]])
    # t_stat_reid, p_val_reid = stats.ttest_ind(dmr_reid_top1, pmr_reid_top1, equal_var=False)

    # print("\n=== Final LaTeX Table: ===")
    # print("\\begin{table*}[!t]")
    # print("\\small")
    # print("\\centering")
    # print("\\begin{tabular}{|l|cc|cc|}")
    # print("\\hline")
    # print("\\multirow{2}{*}{Method} & \\multicolumn{2}{c|}{Action Rec. (mean $\\pm$ std)} & \\multicolumn{2}{c|}{Re-ID (mean $\\pm$ std)} \\\\ \\cline{2-5}")
    # print(" & Top-1 & Top-5 & Top-1 & Top-5 \\\\ \\hline")

    # print_table_line("DMR", dmr_stats)
    # print_table_line("\\textbf{PMR}", pmr_stats)

    # print("\\hline")
    # print("\\end{tabular}")
    # print("\\caption{Statistical significance of PMR vs. DMR (random RA).")
    # print(f"A T-test on top-1 action yields p={p_val_action:.4f}, on top-1 reID yields p={p_val_reid:.4f}")
    # print("\\label{tab:statsig}")
    # print("\\end{table*}")

    # print("\n=== T-test Results ===")
    # print(f" Top-1 Action: t={t_stat_action:.4f}, p={p_val_action:.6f}")
    # print(f" Top-1 ReID  : t={t_stat_reid:.4f}, p={p_val_reid:.6f}")

    # Done.
    print("\n** Done. **")


if __name__ == "__main__":
    main()