import torch
import numpy as np
import pickle
import traceback
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from model.ske_mixf import Model as MixFormerModel
from SGN.data import NTUDataLoaders, AverageMeter

# ------------------ Hyperparameters ------------------
batch_size = 32
num_workers = 0
utility_classes = 60  # Number of utility classes (actions)
privacy_classes = 40  # Number of privacy classes (subjects)
num_point = 25  # Number of joints
num_person = 2  # Number of persons
graph = 'graph.ntu_rgb_d.Graph'
T = 64  # number of frames per sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class {} cannot be found ({})'.format(class_str, traceback.format_exc()))

def parse_file_name(file_name):
    """Parses the filename into a dictionary of parts."""
    file_name = str(file_name)
    if file_name[0] == 'b': # SGN preprocessing
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


class AnonymizedDataset(Dataset):
    """
    A PyTorch dataset that handles:
      - reading the skeleton tensor
      - reshaping to (C, T, V, M)
      - storing action, retargeted_actor, original_actor
    """
    def __init__(self, data_list):
        """
        data_list is a list of dicts:
        [{
            'skeleton': tensor(T, D),  # T frames, D = 25*3
            'action': int,
            'retargeted_actor': int,
            'original_actor': int,
        }, ...]
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        skeleton = sample['skeleton']  # shape (T, 75) if 25*3
        action_label = sample['action']
        retargeted_actor = sample['retargeted_actor']
        original_actor = sample['original_actor']

        # Reshape to (T, V, C)
        T_frames, D = skeleton.shape
        V = 25
        C = 3
        M = 2  # model expects 2 persons
        assert D == V*C, f"Dimension mismatch: D={D}, expected {V*C}"

        # If not a tensor yet, convert
        if not isinstance(skeleton, torch.Tensor):
            skeleton = torch.tensor(skeleton, dtype=torch.float32)

        skeleton = skeleton.view(T_frames, V, C)  # (T, V, C)

        # Expand to have two persons: (T, 2, V, C)
        skeleton = skeleton.unsqueeze(1)  # (T, 1, V, C)
        zeros_tensor = torch.zeros_like(skeleton)  # (T, 1, V, C)
        skeleton = torch.cat([skeleton, zeros_tensor], dim=1)  # (T, 2, V, C)

        # Permute to (C, T, V, M)
        skeleton = skeleton.permute(3, 0, 2, 1).contiguous()

        return skeleton, action_label, retargeted_actor, original_actor


def eval_skeleton_mixformer(data_list):
    """
    data_list is a list of dicts, each containing:
    {
      'skeleton': (T, D),
      'action': int,
      'retargeted_actor': int,
      'original_actor': int
    }
    """
    # ------------- Load the action recognition model -------------
    ar_model_weights = "E:\\LocalCode\\Transformer Retargeting\\eval\\mixformer\\pretrained\\ntu\\ar_cv.pth"
    ar_model_class = 'model.ske_mixf.Model'
    AR_Model = import_class(ar_model_class)
    ar_model_args = {
        'num_class': 60,
        'num_point': 25,
        'num_person': 2,
        'graph': graph,
    }
    ar_model = AR_Model(**ar_model_args)
    ar_state_dict = torch.load(ar_model_weights, weights_only=False)
    ar_state_dict = {k.replace('module.', ''): v for k, v in ar_state_dict.items()}
    ar_model.load_state_dict(ar_state_dict)
    ar_model = ar_model.to(device)
    ar_model.eval()

    # ------------- Load the re-identification model -------------
    ri_model_weights = "E:\\LocalCode\\Transformer Retargeting\\eval\\mixformer\\pretrained\\ntu\\ri.pth"
    ri_model_class = 'model.ske_mixf.Model'
    RI_Model = import_class(ri_model_class)
    ri_model_args = {
        'num_class': 60,  # same in checkpoint
        'num_point': 25,
        'num_person': 2,  # expecting 2 persons
        'graph': graph,
    }
    ri_model = RI_Model(**ri_model_args)
    ri_state_dict = torch.load(ri_model_weights, weights_only=False)
    ri_state_dict = {k.replace('module.', ''): v for k, v in ri_state_dict.items()}
    ri_model.load_state_dict(ri_state_dict)
    ri_model = ri_model.to(device)
    ri_model.eval()

    # ------------- Use DataLoader for batching -------------
    dataset = AnonymizedDataset(data_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total_samples = 0
    correct_action = 0
    correct_actor_ret = 0
    correct_actor_orig = 0

    # ------------- Evaluate in batches -------------
    for skeletons, action_labels, ret_actors, orig_actors in tqdm(loader):
        skeletons = skeletons.to(device)          # (B, C, T, V, M)
        action_labels = action_labels.to(device)  
        ret_actors = ret_actors.to(device)
        orig_actors = orig_actors.to(device)

        batch_size_curr = skeletons.shape[0]
        total_samples += batch_size_curr

        with torch.no_grad():
            # Action recognition
            ar_output = ar_model(skeletons)
            _, ar_predicted = torch.max(ar_output, 1)
            correct_action += (ar_predicted == action_labels).sum().item()

            # Re-identification
            ri_output = ri_model(skeletons)
            _, ri_predicted = torch.max(ri_output, 1)
            correct_actor_ret += (ri_predicted == ret_actors).sum().item()
            correct_actor_orig += (ri_predicted == orig_actors).sum().item()

    # ------------- Summarize results -------------
    action_accuracy = (correct_action / total_samples) * 100
    print(f'Action Recognition Accuracy: {action_accuracy:.2f}%')

    print('Re-identification Results:')
    print(f'Predicted Retargeted Actor: {correct_actor_ret}/{total_samples} '
          f'({(correct_actor_ret/total_samples)*100:.2f}%)')
    print(f'Predicted Original Actor: {correct_actor_orig}/{total_samples} '
          f'({(correct_actor_orig/total_samples)*100:.2f}%)')
    neither = total_samples - correct_actor_ret - correct_actor_orig
    print(f'Predicted Neither Actor: {neither}/{total_samples} '
          f'({(neither/total_samples)*100:.2f}%)')


# ------------------ MAIN SCRIPT EXAMPLE ------------------
datasets = {
    'pmr_random_RA': ('results/X_hat_random_RA.pkl', False, True),
    'pmr_constant_RA': ('results/X_hat_constant_RA.pkl', False, True),
    'pmr_random_CA': ('results/X_hat_random_CA.pkl', False, True),
    'pmr_constant_CA': ('results/X_hat_constant_CA.pkl', False, True),
}

for dataset in datasets:
    data_path, _, _ = datasets[dataset]
    with open(data_path, 'rb') as f:
        anonymized_data = pickle.load(f)
    print(f'Evaluating {dataset}...')

    # Trim/pad each sample to T=64, then convert to tensors
    for file in anonymized_data:
        skeleton = anonymized_data[file]
        if skeleton.shape[0] < T:
            pad_frames = T - skeleton.shape[0]
            pad_skeleton = skeleton[-1].repeat(pad_frames, 1)
            anonymized_data[file] = torch.cat([skeleton, pad_skeleton], dim=0)
        elif skeleton.shape[0] > T:
            anonymized_data[file] = skeleton[:T]

        # Convert to torch.tensor if needed
        anonymized_data[file] = torch.tensor(anonymized_data[file], dtype=torch.float32)

    # Build a list of dicts for the dataset
    data_list = []
    for file in anonymized_data:
        info = parse_file_name(file)
        data_list.append({
            'skeleton': anonymized_data[file],
            'action': info['A'],
            'retargeted_actor': info['P'],
            'original_actor': info['P'],
        })

    # Now evaluate in batches
    eval_skeleton_mixformer(data_list)
