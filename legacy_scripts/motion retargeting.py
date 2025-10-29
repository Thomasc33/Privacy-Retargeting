# %% [markdown]
# # Imports and Hyperparameters

# %%
import pickle
import random
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import glob
from PIL import Image
from tqdm import tqdm
import plotly
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from SGN.model import SGN
from SGN.data import NTUDataLoaders, AverageMeter
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
import time
_=mlflow.set_experiment("Privacy Retargeting")

# %%
# Hyper Parameters
only_use_pos = True # True uses SGN preprocessing, False uses my preprocessing
remove_two_actor_actions = True
one_dimension_conv = False
ntu_120 = True
only_ntu_120 = True
seperate_train_test = True
sgn_eval_after_each_stage = False
binary_data = False
train_cameras = [2, 3]
test_cameras = [1]
train_actors = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103] #https://ar5iv.labs.arxiv.org/html/1905.04757
T = 75
k = 5
setting = 'cv'
dataset = 'NTU'
metric = 'val_utility_acc_coop'
matric_minimize = False
device = torch.device('cuda:0')
seg = 20
lr = 1e-5
adv_lr = 1e-5
util_classifier_alpha = 10
priv_classifier_alpha = .1
if ntu_120:
    utility_classes = 120
    privacy_classes = 106
else:
    utility_classes = 60
    privacy_classes = 40
validation_acc_freq = -1 #-1 to disable
emb_clf_update_per_epoch_paired = 1
emb_clf_update_per_epoch_unpaired = 3
encoded_channels = (128, 16) # default
dmr_encoded_channels = (256, 32) # dmr
# encoded_channels = dmr_encoded_channels # use this for dmr
batch_size = 32
workers=0
cross_samples_train = 50000
cross_samples_test = 5000

# %%
# Validate parameters
assert len(train_cameras) > 0 and len(test_cameras) > 0
assert emb_clf_update_per_epoch_paired > 0 and emb_clf_update_per_epoch_unpaired > 0

# %% [markdown]
# # Data

# %% [markdown]
# ## Import and Organize
# 
# X = (frames, joints, pos + orientation)
#     
#     (frames, 25, 7)

# %%
# load data
if only_use_pos:
    with open('ntu/SGN/X_full.pkl', 'rb') as f:
        X = pickle.load(f)
else:
    with open('ntu/X.pkl', 'rb') as f:
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
if remove_two_actor_actions:
    to_rem = []
    for file in X.keys():
        if int(str(file).split('A')[1][:3]) in two_action_files:
            to_rem.append(file)
    for file in to_rem:
        del X[file]

# %%
# only allow two actors and two actions for testing out training stages
if binary_data:
    actors = set([8,11])
    actions = set([1,2])
    to_rem = []
    for file in X.keys():
        if int(str(file.decode('utf-8')).split('P')[1][:3]) not in actors or int(str(file.decode('utf-8')).split('A')[1][:3]) not in actions:
            to_rem.append(file)
    for file in to_rem:
        del X[file]

# %% [markdown]
# ## Visualization Function

# %%
def render_frame(d):
    reshaped_data = d.reshape(-1, 3)
    x = reshaped_data[:, 0]
    y = reshaped_data[:, 1]
    z = reshaped_data[:, 2]

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    fig = px.scatter_3d(df, x='x', y='y', z='z', color=np.linspace(1, 25, len(x)),
                        color_continuous_scale='Rainbow', title='Interactive 3D Scatter Plot')

    fig.update_traces(marker=dict(size=2))

    cons = [[0, 1], [1, 20], [20, 2], [2, 3], [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24], [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22], [0, 16], [16, 17], [17, 18], [18, 19], [0, 12], [12, 13], [13, 14], [14, 15]]

    for con in cons:
        lx = [x[con[0]], x[con[1]]]
        ly = [y[con[0]], y[con[1]]]
        lz = [z[con[0]], z[con[1]]]
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=2)))

    fig.show()

def render_video(d, gif=None, show_render=True, duration=100):
    cons = [[0, 1], [1, 20], [20, 2], [2, 3], [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24], [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22], [0, 16], [16, 17], [17, 18], [18, 19], [0, 12], [12, 13], [13, 14], [14, 15]]

    frame_data = d[0].reshape(-1, 3)
    x = frame_data[:, 0]
    y = frame_data[:, 1]
    z = frame_data[:, 2]

    # Flatten the tensor to a 2D tensor with shape [frames * points, coordinates]
    d_flattened = d.reshape(-1, 3)

    # Calculate global bounds for x, y, and z
    x_min, x_max = d_flattened[:, 0].min().item(), d_flattened[:, 0].max().item()
    y_min, y_max = d_flattened[:, 1].min().item(), d_flattened[:, 1].max().item()
    z_min, z_max = d_flattened[:, 2].min().item(), d_flattened[:, 2].max().item()

    # Expand the bounds a bit for better visualization
    padding = 0.5
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]
    z_range = [z_min - padding, z_max + padding]

    # Set the fixed range for each axis
    scene = dict(
        xaxis=dict(range=x_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=y_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(range=z_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False),
        camera=dict(
                eye=dict(x=0, y=0, z=-.9),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0)
            ),
        aspectmode='cube',
        bgcolor='rgba(255,255,255,1)'
    )

    layout = go.Layout(updatemenus=[dict(type='buttons', showactive=False,
                                        buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])],
                    sliders=[dict(steps=[])],
                    title="Animated 3D Scatter Plot with Connections",
                    scene=scene,
                    autosize=False,
            )

    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                        marker=dict(size=2, color=np.linspace(1, 25, 25), colorscale='Rainbow'))

    traces = [scatter]

    for con in cons:
        lx = [x[con[0]], x[con[1]]]
        ly = [y[con[0]], y[con[1]]]
        lz = [z[con[0]], z[con[1]]]
        line_trace = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=2))
        traces.append(line_trace)

    fig = go.Figure(data=traces, layout=layout)

    frame_list = []

    for i in range(d.shape[0]):
        frame_data = d[i].reshape(-1, 3)
        x, y, z = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    
        fig.data[0].x = x
        fig.data[0].y = y
        fig.data[0].z = z

        frame_traces = []

        frame_scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                    marker=dict(size=2, color=np.linspace(1, 25, 25), colorscale='Rainbow'))
        frame_traces.append(frame_scatter)

        for con in cons:
            lx = [x[con[0]], x[con[1]]]
            ly = [y[con[0]], y[con[1]]]
            lz = [z[con[0]], z[con[1]]]
            line_trace = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=2))
            frame_traces.append(line_trace)

        frame = go.Frame(data=frame_traces, name=f'Frame {i}')
        frame_list.append(frame)

    fig.frames = frame_list

    if show_render: fig.show()

    if gif is not None:
        # Create a directory to save frames
        frame_dir = f'results/gif/{gif}'
        os.makedirs(frame_dir, exist_ok=True)

        layout = go.Layout(scene=scene, width=800, height=600, showlegend=False, margin=dict(l=0, r=0, b=0, t=0), autosize=False, )#paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        # Save each frame as an image
        for i in tqdm(range(len(frame_list))):
            frame = frame_list[i]
            fig = go.Figure(data=frame.data, layout=layout)
            plotly.io.write_image(fig, f'{frame_dir}/frame_{i}.png', width=800, height=600, scale=1)

        frames = glob.glob(f'{frame_dir}/frame_*.png')
        frames.sort()  # Ensure the frames are in order

        # Create an image object from the first frame
        img, *imgs = [Image.open(f) for f in frames]

        # Convert to GIF and save
        img.save(fp=f'results/gif/{gif}.gif', format='GIF', append_images=imgs,
                 save_all=True, duration=duration, loop=0)

        # Delete the frames after creating the GIF
        # for f in frames:
        #     os.remove(f)

# %% [markdown]
# ## Cross Data Logic

# %%
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

def organize_data(data):
    train_data = defaultdict(list)
    test_data = defaultdict(list)

    organized_data = defaultdict(list)
    for file_name, content in data.items():
        parts = parse_file_name(file_name)
        if not ntu_120: # NTU 60
            if int(parts['A']) > 60:
                continue
        organized_data[parts['C']].append((parts['P'], parts['A'], content))
        if setting == 'cs':
            if parts['P'] in train_actors:
                train_data[parts['C']].append((parts['P'], parts['A'], content))
            else:
                test_data[parts['C']].append((parts['P'], parts['A'], content))

        
    if setting == 'cv':
        for camera in train_cameras:
            train_data[camera].extend(organized_data[camera])

        for camera in test_cameras:
            test_data[camera].extend(organized_data[camera])

    return train_data, test_data

def sample_data(organized_data):
    # Pick a random C pair
    C = random.choice(list(organized_data.keys()))

    # Get all (P, A, content) tuples for this C 
    pa_list = organized_data[C]

    # Pick 2 unique P values, find two overlapping A's
    

    # Pick 2 unique P values and 2 unique A values
    random.shuffle(pa_list)
    unique_p = set()
    unique_a = set()
    for p, a, _ in pa_list:
        if len(unique_p) < 2:
            unique_p.add(p)
        if len(unique_a) < 2:
            unique_a.add(a)
        if len(unique_p) == 2 and len(unique_a) == 2:
            break

    if len(unique_p) < 2 or len(unique_a) < 2:
        raise Exception(f'Not enough unique P or A values for C pair {C}')

    # Form all four (P, A) pairs and get the corresponding content
    sampled_data = [] #(p1, a1) (p1, a2) (p2, a1) (p2, a2)
    for p in unique_p:
        for a in unique_a:
            for pa_content in pa_list:
                if pa_content[0] == p and pa_content[1] == a:
                    sampled_data.append(pa_content)
                    break

    return sampled_data

def gen_samples(samples, data):
    d = []
    unique_samples = set()  # Set to track unique samples
    for _ in range(samples):
        failed = 0
        while True:
            d_ = sample_data(data)
            d_tuple = tuple(tuple(x) for x in d_)
            if d_tuple not in unique_samples and len(d_tuple) == 4:
                unique_samples.add(d_tuple)  # Add the unique sample to the set
                d.append(d_)  # Add the unique sample to the dataset
                break
            failed += 1
            if failed > 100:
                print('failed to sample data')
                break
    return np.array(d)

# %% [markdown]
# ## Reconstruction Sampling

# %%
def sample_rec_data(X):
    # Remove NTU 120 if needed
    if not ntu_120:
        X = {k: v for k, v in X.items() if int(parse_file_name(k)['A']) <= 60}

    # Split data into train and test
    # X_train_keys, X_test_keys = train_test_split(list(X.keys()), test_size=0.2, random_state=42)

    # Split by camera views
    X_train_keys = []
    X_test_keys = []
    if setting == 'cs':
        for key in X.keys():
            if parse_file_name(key)['P'] in train_actors:
                X_train_keys.append(key)
            else:
                X_test_keys.append(key)
    elif setting == 'cv':
        for key in X.keys():
            if parse_file_name(key)['C'] in train_cameras:
                X_train_keys.append(key)
            else:
                X_test_keys.append(key)
    
    # Create train and test sets
    X_train = np.zeros((len(X_train_keys), T, 25, 3 if only_use_pos else 7))
    X_test = np.zeros((len(X_test_keys), T, 25, 3 if only_use_pos else 7))
    for i, key in enumerate(X_train_keys):
        X_train[i] = X[key]
    for i, key in enumerate(X_test_keys):
        X_test[i] = X[key]

    # Get actor and action names
    train_actors = [parse_file_name(key)['P'] for key in X_train_keys]
    test_actors = [parse_file_name(key)['P'] for key in X_test_keys]
    train_actions = [parse_file_name(key)['A'] for key in X_train_keys]
    test_actions = [parse_file_name(key)['A'] for key in X_test_keys]
    
    return X_train, X_test, train_actors, train_actions, test_actors, test_actions

# %% [markdown]
# ## Sample Data

# %%
class Cross_Data(Dataset):
    def __init__(self, sampled_data):
        self.data = sampled_data # the tuple is actor, action, frames
        self.x1 = sampled_data[:, 0, 2] # P1, A1
        self.x2 = sampled_data[:, 3, 2] # P2, A2
        self.y1 = sampled_data[:, 1, 2] # P1, A2
        self.y2 = sampled_data[:, 2, 2] # P2, A1
        self.none = torch.zeros(1)

    def __getitem__(self, index): # data, actors, actions
        if only_use_pos:
            return self.x1[index], self.none,\
                    self.x2[index], self.none,\
                    self.y1[index], self.none,\
                    self.y2[index], self.none,\
                    [float(self.data[index][0][0]), float(self.data[index][3][0])], [float(self.data[index][0][1]), float(self.data[index][3][1])]
        return self.x1[index][:, :, 0:3], self.x1[index][:, :, 3:7],\
                self.x2[index][:, :, 0:3], self.x2[index][:, :, 3:7],\
                self.y1[index][:, :, 0:3], self.y1[index][:, :, 3:7],\
                self.y2[index][:, :, 0:3], self.y2[index][:, :, 3:7],\
                [float(self.data[index][0][0]), float(self.data[index][3][0])], [float(self.data[index][0][1]), float(self.data[index][3][1])]
    
    def __len__(self):
        return len(self.data)

class Rec_Data(Dataset):
    def __init__(self, X, Actor, Action):
        self.X = X
        self.Actor = Actor
        self.Action = Action
    
    def __getitem__(self, index):
        return self.X[index], float(self.Actor[index]), float(self.Action[index])
    
    def __len__(self):
        return len(self.X)


# Cross Data
# organized_data_train, organized_data_test = organize_data(X)
# train_data = gen_samples(cross_samples_train, organized_data_train)
# if seperate_train_test: val_data = gen_samples(cross_samples_test, organized_data_test)
# else: val_data = train_data
# train_dataset = Cross_Data(train_data)
# val_dataset = Cross_Data(val_data)
# train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# # Rec Data
# rec_train_data, rec_val_data, t_actors, t_actions, v_actors, v_actions = sample_rec_data(X)
# rec_train_dataset = Rec_Data(rec_train_data, t_actors, t_actions)
# rec_val_dataset = Rec_Data(rec_val_data, v_actors, v_actions)
# rec_train_dl = DataLoader(rec_train_dataset, batch_size=batch_size, shuffle=True)
# rec_val_dl = DataLoader(rec_val_dataset, batch_size=batch_size, shuffle=True)

# %%
# # Gather stats on data
# def print_data(d):
#     print(f'Number of samples: {len(d)}')
#     unique_actors = set()
#     unique_actions = set()
#     for d in d:
#         for i in range(4):
#             unique_actors.add(d[i][0])
#             unique_actions.add(d[i][1])
#     print(f'Number of unique actors: {len(unique_actors)}')
#     print(f'Number of unique actions: {len(unique_actions)}')
# print('Train Data:')
# print_data(train_data)
# print('Val Data:')
# print_data(val_data)
# print('Rec Train Data:')
# print(len(rec_train_data))
# print('Rec Val Data:')
# print(len(rec_val_data))

# %% [markdown]
# # Model

# %% [markdown]
# ## Adversary

# %%
# Input is size of latent space
class Adversary_Emb(nn.Module):
    def __init__(self, num_classes):
        super(Adversary_Emb, self).__init__()
        self.channels = [encoded_channels[0], 128, 256, 512]
        self.conv1 = nn.ConvTranspose1d(self.channels[0], self.channels[1], 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose1d(self.channels[1], self.channels[2], 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose1d(self.channels[2], self.channels[3], 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm1d(self.channels[1])
        self.bn2 = nn.BatchNorm1d(self.channels[2])
        self.bn3 = nn.BatchNorm1d(self.channels[3])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(self.channels[3], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = F.softmax(self.fc3(x), dim=1)
        # x = self.fc3(x)
        return x
    
class Discriminator(nn.Module): # 1 = real, 0 = fake
    def __init__(self):
        super(Discriminator, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=T, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
        self.fc1 = nn.Linear(80, 32)
        self.fc2 = nn.Linear(32, 1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)
        
        x = self.ref2(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref3(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref4(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        #flatten
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

# %% [markdown]
# ## Motion Retargeting

# %%
class Encoder1D(nn.Module):
    def __init__(self):
        super(Encoder1D, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=T, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=512, out_channels=encoded_channels[0], kernel_size=3, stride=1, padding=1)
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(encoded_channels[0], encoded_channels[0] * encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)
        
        x = self.ref2(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref3(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref4(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1) 
        x = self.fc1(x)
        x = x.view(-1, *encoded_channels)

        return x

class Decoder1D(nn.Module):
    def __init__(self):
        super(Decoder1D, self).__init__()

        self.dec1 = nn.ConvTranspose1d(in_channels=encoded_channels[0]*2, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose1d(in_channels=96, out_channels=T, kernel_size=3, stride=1, padding=1)

        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
 
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest') 

        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)

        x = self.ref2(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)

        x = self.ref3(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)

        x = self.ref4(x)
        x = self.acti(self.dec4(x))
        x = self.up75(x)
        return x
    
class Encoder2D(nn.Module):
    def __init__(self):
        super(Encoder2D, self).__init__()

        self.enc1 = nn.Conv2d(in_channels=T, out_channels=12, kernel_size=(3,3), stride=1, padding=1)
        self.enc2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), stride=1, padding=1)
        self.enc3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.enc4 = nn.Conv2d(in_channels=32, out_channels=encoded_channels[0], kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(encoded_channels[0], encoded_channels[0] * encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(-1, *encoded_channels)

        return x

class Decoder2D(nn.Module):
    def __init__(self):
        super(Decoder2D, self).__init__()

        self.dec1 = nn.ConvTranspose2d(in_channels=encoded_channels[0]*2, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=(3,3), stride=1, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=96, out_channels=75, kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(3)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest') 
        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec4(x))
        x = self.up75(x)
        
        return x


# %%
class DMR_Encoder1D(nn.Module):
    def __init__(self):
        super(DMR_Encoder1D, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=T, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=512, out_channels=dmr_encoded_channels[0], kernel_size=3, stride=1, padding=1)
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(dmr_encoded_channels[0], dmr_encoded_channels[0] * dmr_encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)
        
        x = self.ref2(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref3(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref4(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1) 
        x = self.fc1(x)
        x = x.view(-1, *dmr_encoded_channels)

        return x

class DMR_Decoder1D(nn.Module):
    def __init__(self):
        super(DMR_Decoder1D, self).__init__()

        self.dec1 = nn.ConvTranspose1d(in_channels=dmr_encoded_channels[0]*2, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose1d(in_channels=96, out_channels=T, kernel_size=3, stride=1, padding=1)

        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
 
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest') 

        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)

        x = self.ref2(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)

        x = self.ref3(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)

        x = self.ref4(x)
        x = self.acti(self.dec4(x))
        x = self.up75(x)
        return x
    
class DMR_Encoder2D(nn.Module):
    def __init__(self):
        super(DMR_Encoder2D, self).__init__()

        self.enc1 = nn.Conv2d(in_channels=T, out_channels=12, kernel_size=(3,3), stride=1, padding=1)
        self.enc2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), stride=1, padding=1)
        self.enc3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.enc4 = nn.Conv2d(in_channels=32, out_channels=dmr_encoded_channels[0], kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(dmr_encoded_channels[0], dmr_encoded_channels[0] * dmr_encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(-1, *dmr_encoded_channels)

        return x

class DMR_Decoder2D(nn.Module):
    def __init__(self):
        super(DMR_Decoder2D, self).__init__()

        self.dec1 = nn.ConvTranspose2d(in_channels=dmr_encoded_channels[0]*2, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=(3,3), stride=1, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=96, out_channels=75, kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(3)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest') 
        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec4(x))
        x = self.up75(x)
        
        return x
    
class DMR(nn.Module):
    def __init__(self, adv_lr=1e-4, use_adv=False):
        super(DMR, self).__init__()

        # AutoEncoder Models
        if one_dimension_conv:
            self.static_encoder = DMR_Encoder1D()
            self.advamic_encoder = DMR_Encoder1D()
            self.decoder = DMR_Decoder1D()
        else:
            self.static_encoder = DMR_Encoder2D()
            self.dynamic_encoder = DMR_Encoder2D()
            self.decoder = DMR_Decoder1D()

        # Adversarial Models
        self.use_adv = use_adv
        if use_adv:
            self.priv_adv = Adversary_Emb(privacy_classes).to(device) # input = dynamic embedding, output = privacy class
            self.priv_coop = Adversary_Emb(privacy_classes).to(device) # input = static embedding, output = privacy class
            self.util_adv = Adversary_Emb(utility_classes).to(device) # input = static embedding, output = utility class
            self.util_coop = Adversary_Emb(utility_classes).to(device) # input = dynamic embedding, output = utility class
            self.discriminator = Discriminator().to(device)

            self.priv_optim = torch.optim.AdamW(self.priv_adv.parameters(), lr=adv_lr)
            self.priv_coop_optim = torch.optim.AdamW(self.priv_coop.parameters(), lr=adv_lr)
            self.util_optim = torch.optim.AdamW(self.util_adv.parameters(), lr=adv_lr)
            self.util_coop_optim = torch.optim.AdamW(self.util_coop.parameters(), lr=adv_lr)
            self.discriminator_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=adv_lr)

            # Freeze Adversarial Models
            self.priv_adv.eval()
            self.priv_coop.eval()
            self.util_adv.eval()
            self.util_coop.eval()
            self.discriminator.eval()

        # Loss Functions
        self.triplet_loss = nn.TripletMarginLoss()
        self.bce_loss = nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # Info for loss functions
        self.end_effectors = torch.tensor([19, 15, 23, 24, 21, 22, 3]).to(device) * 3
        self.chain_lengths = torch.tensor([5, 5, 8, 8, 8, 8, 5]).to(device)

        # Lambdas for discounted loss
        self.lambda_rec = 2
        self.lambda_cross = 0.1
        self.lambda_ee = 1
        self.lambda_smoothing = 3
        self.lambda_trip = 1
        self.lambda_latent = 10
        self.lambda_adv_util_coop = util_classifier_alpha
        self.lambda_adv_priv_coop = priv_classifier_alpha
        self.lambda_adv_util_adv = util_classifier_alpha
        self.lambda_adv_priv_adv = priv_classifier_alpha
        self.lambda_adv_disc = 1

        # Loss Toggles
        self.use_rec_loss = True
        self.use_cross_loss = True
        self.use_ee_loss = True 
        self.use_trip_loss_paired = True 
        self.use_trip_loss_unpaired = True
        self.use_smoothing_loss = True
        self.use_latent_consistency = True

    def get_loss_params(self):
        return {
            'lambda_rec': self.lambda_rec,
            'lambda_cross': self.lambda_cross,
            'lambda_ee': self.lambda_ee,
            'lambda_trip': self.lambda_trip,
            'lambda_latent': self.lambda_latent,
            'lambda_adv_util_coop': self.lambda_adv_util_coop,
            'lambda_adv_priv_coop': self.lambda_adv_priv_coop,
            'lambda_adv_util_adv': self.lambda_adv_util_adv,
            'lambda_adv_priv_adv': self.lambda_adv_priv_adv,
            'lambda_adv_disc': self.lambda_adv_disc,
            'use_rec_loss': self.use_rec_loss,
            'use_cross_loss': self.use_cross_loss,
            'use_ee_loss': self.use_ee_loss,
            'use_trip_loss_paired': self.use_trip_loss_paired,
            'use_trip_loss_unpaired': self.use_trip_loss_unpaired,
            'use_smoothing_loss': self.use_smoothing_loss,
            'use_latent_consistency': self.use_latent_consistency
        }

    def cross(self, x1, x1_rot, x2, x2_rot):
        d1 = self.dynamic_encoder(x1_rot)
        d2 = self.dynamic_encoder(x2_rot)
        s1 = self.static_encoder(x1)
        s2 = self.static_encoder(x2)
        
        x1_hat = self.decoder(torch.cat((d1, s1), dim=1))
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1))
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1))
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1))

        return x1_hat, x2_hat, y1_hat, y2_hat
    
    def eval(self, x1_rot, x2):
        dynamic = self.dynamic_encoder(x1_rot)
        static = self.static_encoder(x2)
        return self.decoder(torch.cat((dynamic, static), dim=1))

    def rec_loss(self, x, x_rot):
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x)
        x_hat = self.decoder(torch.cat((d, s), dim=1))
        if not one_dimension_conv:
            x_ = x.reshape(x.size(0), T, -1)
        return self.reconstruction_loss(x_, x_hat)
    
    def loss_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, cross = True, reconstruction = True, emb_adv = True, discrim_adv = True, verbose = False):
        d1 = self.dynamic_encoder(x1_rot) # A1
        d2 = self.dynamic_encoder(x2_rot) # A2
        s1 = self.static_encoder(x1) # P1
        s2 = self.static_encoder(x2) # P2

        x1_hat = self.decoder(torch.cat((d1, s1), dim=1)) # P1, A1
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1)) # P2, A2
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1)) # P2, A1
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1)) # P1, A2

        d12 = self.dynamic_encoder(y1_rot) # A1
        d21 = self.dynamic_encoder(y2_rot) # A2
        s12 = self.static_encoder(y1) # P2
        s21 = self.static_encoder(y2) # P1

        x1_hat_ = self.decoder(torch.cat((d12, s21), dim=1)) # P1, A1
        x2_hat_ = self.decoder(torch.cat((d21, s12), dim=1)) # P2, A2
        y1_hat_ = self.decoder(torch.cat((d12, s12), dim=1)) # P2, A1
        y2_hat_ = self.decoder(torch.cat((d21, s21), dim=1)) # P1, A2

        # x1_hat is reconstruction of x1
        # x2_hat is reconstruction of x2
        # y1_hat is cross reconstruction from x1 and x2
        # y2_hat is cross reconstruction from x2 and x1
        # x1_hat_ is cross reconstruction from y1 and y2
        # x2_hat_ is cross reconstruction from y2 and y1
        # y1_hat_ is reconstruction of y1
        # y2_hat_ is reconstruction of y2
        # d1 = A1
        # d2 = A2
        # d12 = A1
        # d21 = A2
        # s1 = P1
        # s2 = P2
        # s12 = P2
        # s21 = P1

        # flatten data if 2D
        if not one_dimension_conv:
            x1 = x1.view(x1.size(0), T, -1)
            x2 = x2.view(x2.size(0), T, -1)
            y1 = y1.view(y1.size(0), T, -1)
            y2 = y2.view(y2.size(0), T, -1)
        
        # initialize all losses to 0 tensor
        rec_loss = torch.zeros(1).to(device)
        cross_loss = torch.zeros(1).to(device)
        end_effector_loss = torch.zeros(1).to(device)
        triplet_loss = torch.zeros(1).to(device)
        smoothing_loss = torch.zeros(1).to(device)
        latent_consistency_loss = torch.zeros(1).to(device)
        privacy_loss = torch.zeros(1).to(device)
        privacy_loss_adv = torch.zeros(1).to(device)
        privacy_loss_coop = torch.zeros(1).to(device)
        utility_loss = torch.zeros(1).to(device)
        utility_loss_adv = torch.zeros(1).to(device)
        utility_loss_coop = torch.zeros(1).to(device)
        privacy_acc_adv = torch.zeros(1).to(device)
        privacy_acc_coop = torch.zeros(1).to(device)
        utility_acc_adv = torch.zeros(1).to(device)
        utility_acc_coop = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)
                        
        # reconstruction loss
        if self.use_rec_loss and reconstruction:
            rec_loss = (self.reconstruction_loss(x1, x1_hat) + self.reconstruction_loss(x2, x2_hat) + self.reconstruction_loss(y1, y1_hat_) + self.reconstruction_loss(y2, y2_hat_)) / 4
            if verbose: print('Reconstruction Loss: ', rec_loss.item())
        
        # cross reconstruction loss
        if self.use_cross_loss and cross:
            # could move this to its own function, but since cross is basically reconstruction, its fine like this
            cross_loss = (self.reconstruction_loss(y1, y1_hat) + self.reconstruction_loss(y2, y2_hat) + self.reconstruction_loss(x1, x1_hat_) + self.reconstruction_loss(x2, x2_hat_)) / 4
            if verbose: print('Cross Reconstruction Loss: ', cross_loss.item())
        
        # end effector loss
        if self.use_ee_loss:
            if reconstruction:
                end_effector_loss += (self.end_effector_loss(x1_hat, x1) + self.end_effector_loss(x2_hat, x2)) / 2
            if cross:
                end_effector_loss += (self.end_effector_loss(y1_hat, y1) + self.end_effector_loss(y2_hat, y2)) / 2
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # triplet loss
        if self.use_trip_loss_paired: # anchor, positive, negative
            # d1 = A1, d2 = A2, d12 = A1, d21 = A2
            # s1 = P1, s2 = P2, s12 = P2, s21 = P1
            # d12,s12 = y1, d21,s21 = y2
            # y1 = jk, y2 = il
            triplet_loss = self.triplet_loss(d12, d1, d2) \
                            + self.triplet_loss(d21, d2, d1) \
                            + self.triplet_loss(s12, s2, s1) \
                            + self.triplet_loss(s21, s1, s2) 
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        if self.use_smoothing_loss:
            smoothing_loss = (self.smoothing_loss(x1, x1_hat) + self.smoothing_loss(x2, x2_hat) + self.smoothing_loss(y1, y1_hat_) + self.smoothing_loss(y2, y2_hat_) + \
                                self.smoothing_loss(x1, x1_hat_) + self.smoothing_loss(x2, x2_hat_) + self.smoothing_loss(y1, y1_hat) + self.smoothing_loss(y2, y2_hat)) / 8
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())

        # latent consistency loss
        if self.use_latent_consistency:
            latent_consistency_loss = (self.latent_consistency_loss(d1, d12) + self.latent_consistency_loss(d2, d21) + self.latent_consistency_loss(s1, s21) + self.latent_consistency_loss(s2, s12)) / 4
            if verbose: print('Latent Consistency Loss: ', latent_consistency_loss.item())

        # adversarial loss
        if self.use_adv and emb_adv:
            actor_y1, actor_y2 = actors[0] - 1, actors[1] - 1
            actor_y1, actor_y2 = torch.eye(privacy_classes)[actor_y1.long()].to(device), torch.eye(privacy_classes)[actor_y2.long()].to(device)
            action_y1, action_y2 = actions[0] - 1, actions[1] - 1
            action_y1, action_y2 = torch.eye(utility_classes)[action_y1.long()].to(device), torch.eye(utility_classes)[action_y2.long()].to(device)

            # x1 => d1 s1
            # x2 => d2 s2

            # d1 => p1
            # d2 => p2
            # s1 => a1
            # s2 => a2
            
            # actor_y1 = p1
            # actor_y2 = p2

            # action_y1 = a1
            # action_y2 = a2


            # privacy loss (adversarial)
            privacy_loss_adv = (-self.adv_loss(self.priv_adv, d1, actor_y1) -self.adv_loss(self.priv_adv, d2, actor_y2))/2
            privacy_acc_adv = (self.adv_accuracy(self.priv_adv, d1, actor_y1) + self.adv_accuracy(self.priv_adv, d2, actor_y2))/2

            # privacy loss (coop)
            privacy_loss_coop = (self.adv_loss(self.priv_coop, s1, actor_y1) + self.adv_loss(self.priv_coop, s2, actor_y2))/2
            privacy_acc_coop = (self.adv_accuracy(self.priv_coop, s1, actor_y1) + self.adv_accuracy(self.priv_coop, s2, actor_y2))/2

            # utility loss (adversarial)
            utility_loss_adv = (-self.adv_loss(self.util_adv, s1, action_y1) -self.adv_loss(self.util_adv, s2, action_y2))/2
            utility_acc_adv = (self.adv_accuracy(self.util_adv, s1, action_y1) + self.adv_accuracy(self.util_adv, s2, action_y2))/2

            # utility loss (coop)
            utility_loss_coop = (self.adv_loss(self.util_coop, d1, action_y1) + self.adv_loss(self.util_coop, d2, action_y2))/2
            utility_acc_coop = (self.adv_accuracy(self.util_coop, d1, action_y1) + self.adv_accuracy(self.util_coop, d2, action_y2))/2

            privacy_loss = privacy_loss_adv * self.lambda_adv_priv_adv + privacy_loss_coop * self.lambda_adv_priv_coop
            utility_loss = utility_loss_adv * self.lambda_adv_util_adv + utility_loss_coop * self.lambda_adv_util_coop

            if verbose: 
                print('Privacy Loss (Adversarial): ', privacy_loss_adv.item(), '\tPrivacy Loss (Coop): ', privacy_loss_coop.item())
                print('Utility Loss (Adversarial): ', utility_loss_adv.item(), '\tUtility Loss (Coop): ', utility_loss_coop.item())
                print('Privacy Accuracy (Adversarial): ', privacy_acc_adv.item(), '\tPrivacy Accuracy (Coop): ', privacy_acc_coop.item())
                print('Utility Accuracy (Adversarial): ', utility_acc_adv.item(), '\tUtility Accuracy (Coop): ', utility_acc_coop.item())
            

        if self.use_adv and discrim_adv:
            # discrimnator (adversarial)
            discrim_out_fake = self.discriminator(torch.cat((x1_hat, x2_hat, y1_hat, y2_hat, x1_hat_, x2_hat_, y1_hat_, y2_hat_)))
            discriminator_loss = self.bce_loss(discrim_out_fake, torch.ones_like(discrim_out_fake))
            discriminator_acc = torch.sum(torch.round(discrim_out_fake) == 0).float() / (8 * batch_size)
            if verbose: print('Discriminator Loss: ', discriminator_loss.item(), '\tDiscriminator Accuracy: ', discriminator_acc.item())

        losses = {
            'rec_loss': rec_loss.item(),
            'cross_loss': cross_loss.item(),
            'end_effector_loss': end_effector_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'smoothing_loss': smoothing_loss.item(),
            'latent_consistency_loss': latent_consistency_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'privacy_loss_adv': privacy_loss_adv.item(),
            'privacy_loss_coop': privacy_loss_coop.item(),
            'privacy_acc_adv': privacy_acc_adv.item(),
            'privacy_acc_coop': privacy_acc_coop.item(),
            'utility_loss': utility_loss.item(),
            'utility_loss_adv': utility_loss_adv.item(),
            'utility_loss_coop': utility_loss_coop.item(),
            'utility_acc_adv': utility_acc_adv.item(),
            'utility_acc_coop': utility_acc_coop.item(),
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_acc': discriminator_acc.item()
        }

        return rec_loss * self.lambda_rec \
                + cross_loss * self.lambda_cross \
                + end_effector_loss * self.lambda_ee \
                + triplet_loss * self.lambda_trip \
                + latent_consistency_loss * self.lambda_latent \
                + privacy_loss \
                + utility_loss \
                + discriminator_loss * self.lambda_adv_disc \
                + smoothing_loss * self.lambda_smoothing, \
                x1_hat, x2_hat, y1_hat, y2_hat, losses

    def loss_unpaired(self, x_pos, x_rot, actors, actions, reconstruction = True, emb_adv = False, discrim_adv = False, ee = False, triplet = False, verbose = False):
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_pos)
        x_hat = self.decoder(torch.cat((d, s), dim=1))

        if not one_dimension_conv:
            x = x_pos.reshape(x_pos.size(0), T, -1)

        # initialize all losses to 0 tensor
        rec_loss = torch.zeros(1).to(device)
        end_effector_loss = torch.zeros(1).to(device)
        triplet_loss = torch.zeros(1).to(device)
        smoothing_loss = torch.zeros(1).to(device)
        privacy_loss = torch.zeros(1).to(device)
        privacy_loss_adv = torch.zeros(1).to(device)
        privacy_loss_coop = torch.zeros(1).to(device)
        utility_loss = torch.zeros(1).to(device)
        utility_loss_adv = torch.zeros(1).to(device)
        utility_loss_coop = torch.zeros(1).to(device)
        privacy_acc_adv = torch.zeros(1).to(device)
        privacy_acc_coop = torch.zeros(1).to(device)
        utility_acc_adv = torch.zeros(1).to(device)
        utility_acc_coop = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        # Reconstruction Loss
        if self.use_rec_loss and reconstruction:
            rec_loss = self.reconstruction_loss(x, x_hat)
            if verbose: print('Reconstruction Loss: ', rec_loss.item())

        # End Effector Loss
        if self.use_ee_loss and ee:
            end_effector_loss = self.end_effector_loss(x_hat, x)
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # Triplet Loss
        if self.use_trip_loss_unpaired and triplet: # anchor, positive, negative
            triplet_loss = (self.triplet_loss(d, d, s) + self.triplet_loss(s, s, d)) / 2
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        # Smoothing Loss
        if self.use_smoothing_loss:
            smoothing_loss = self.smoothing_loss(x, x_hat)
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())

        # Adversarial Loss
        if self.use_adv and emb_adv:
            actor_y = actors - 1
            actor_y = torch.eye(privacy_classes)[actor_y.long()].to(device)
            action_y = actions - 1
            action_y = torch.eye(utility_classes)[action_y.long()].to(device)

            # latent privacy loss (adv)
            privacy_loss_adv = -self.adv_loss(self.priv_adv, d, actor_y)
            privacy_acc_adv = self.adv_accuracy(self.priv_adv, d, actor_y)

            # latent privacy loss (coop)
            privacy_loss_coop = self.adv_loss(self.priv_coop, s, actor_y)
            privacy_acc_coop = self.adv_accuracy(self.priv_coop, s, actor_y)

            # latent utility loss (adv)
            utility_loss_adv = -self.adv_loss(self.util_adv, s, action_y)
            utility_acc_adv = self.adv_accuracy(self.util_adv, s, action_y)

            # latent utility loss (coop)
            utility_loss_coop = self.adv_loss(self.util_coop, d, action_y)
            utility_acc_coop = self.adv_accuracy(self.util_coop, d, action_y)

            privacy_loss = privacy_loss_adv * self.lambda_adv_priv_adv + privacy_loss_coop * self.lambda_adv_priv_coop
            utility_loss = utility_loss_adv * self.lambda_adv_util_adv + utility_loss_coop * self.lambda_adv_util_coop

            if verbose: 
                print('Privacy Loss Adv: ', privacy_loss_adv.item(), '\tPrivacy Loss Coop: ', privacy_loss_coop.item(), '\tPrivacy Loss: ', privacy_loss.item())
                print('Utility Loss Adv: ', utility_loss_adv.item(), '\tUtility Loss Coop: ', utility_loss_coop.item(), '\tUtility Loss: ', utility_loss.item())
                print('Privacy Accuracy Adv: ', privacy_acc_adv.item(), '\tPrivacy Accuracy Coop: ', privacy_acc_coop.item())
                print('Utility Accuracy Adv: ', utility_acc_adv.item(), '\tUtility Accuracy Coop: ', utility_acc_coop.item())


        if self.use_adv and discrim_adv:
            # discrimnator (adversarial)
            discrim_out_fake = self.discriminator(x_hat)
            discriminator_loss = self.bce_loss(discrim_out_fake, torch.ones_like(discrim_out_fake))
            discriminator_acc = torch.sum(torch.round(discrim_out_fake) == 0).float() / (batch_size)
            if verbose: print('Discriminator Loss: ', discriminator_loss.item(), '\tDiscriminator Accuracy: ', discriminator_acc.item())

        losses = {
            'rec_loss': rec_loss.item(),
            'end_effector_loss': end_effector_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'smoothing_loss': smoothing_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'privacy_loss_adv': privacy_loss_adv.item(),
            'privacy_loss_coop': privacy_loss_coop.item(),
            'privacy_acc_adv': privacy_acc_adv.item(),
            'privacy_acc_coop': privacy_acc_coop.item(),
            'utility_loss': utility_loss.item(),
            'utility_loss_adv': utility_loss_adv.item(),
            'utility_loss_coop': utility_loss_coop.item(),
            'utility_acc_adv': utility_acc_adv.item(),
            'utility_acc_coop': utility_acc_coop.item(),
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_acc': discriminator_acc.item()
        }

        return rec_loss * self.lambda_rec \
                + end_effector_loss * self.lambda_ee \
                + triplet_loss * self.lambda_trip \
                + privacy_loss \
                + utility_loss \
                + discriminator_loss * self.lambda_adv_disc \
                + smoothing_loss * self.lambda_smoothing, \
                x_hat, losses

    def reconstruction_loss(self, x, y):
        # return F.mse_loss(x, y)
        return torch.square(torch.norm(x - y, dim=1)).mean()
    
    def latent_consistency_loss(self, x, y):
        return F.mse_loss(x, y)
    
    def end_effector_loss(self, x, y):
        # slice to get the end effector joints
        x_ee = x[:, :, self.end_effectors.unsqueeze(-1) + torch.arange(3).to(device)] 
        y_ee = y[:, :, self.end_effectors.unsqueeze(-1) + torch.arange(3).to(device)]

        # calculate velocities
        x_vel = torch.norm(x_ee[:, 1:] - x_ee[:, :-1], dim=-1) / self.chain_lengths.unsqueeze(0)
        y_vel = torch.norm(y_ee[:, 1:] - y_ee[:, :-1], dim=-1) / self.chain_lengths.unsqueeze(0)
        
        # compute mse loss for each joint
        losses = F.mse_loss(x_vel, y_vel, reduction='none')

        # take sum over end effectors
        loss = losses.sum(dim=1)

        # take mean over batch
        loss = loss.mean()
        
        return loss
    
    def smoothing_loss(self, y, y_pred):
        # (batch, T, 75)
        # Calculate the squared sum of differences for y and y_pred
        diff_y = torch.sum(y[:, :-1] - y[:, 1:], dim=2) ** 2
        diff_y_pred = torch.sum(y_pred[:, :-1] - y_pred[:, 1:], dim=2) ** 2

        # Calculate the absolute difference
        abs_diff = torch.abs(diff_y - diff_y_pred)

        # Sum over all batches and sequence elements
        loss = torch.sum(abs_diff)

        # Normalize by the total number of elements (batch_size * sequence_length)
        total_loss = torch.sqrt(loss) / (y.size(0) * y.size(1))

        return total_loss

    def adv_loss(self, model, x, y):
        return self.cross_entropy(model(x), y)#.long().to(device))
    
    def adv_accuracy(self, model, x, y):
        return (model(x).argmax(dim=1) == y.argmax(dim=1).to(device)).float().mean()

    def forward(self, x, x_rot):
        dyn = self.dynamic_encoder(x_rot)
        sta = self.static_encoder(x)
        x = self.decoder(torch.cat((dyn, sta), dim=1))
        return x
    
    def set_eval(self, eval=True):
        if eval:
            self.static_encoder.eval()
            self.dynamic_encoder.eval()
            self.decoder.eval()
            self.priv_adv.eval()
            self.priv_coop.eval()
            self.util_adv.eval()
            self.util_coop.eval()
            self.discriminator.eval()
        else:
            self.static_encoder.train()
            self.dynamic_encoder.train()
            self.decoder.train()

# %%
class AutoEncoder(nn.Module):
    def __init__(self, adv_lr=1e-4, use_adv=True):
        super(AutoEncoder, self).__init__()

        # AutoEncoder Models
        if one_dimension_conv:
            self.static_encoder = Encoder1D()
            self.advamic_encoder = Encoder1D()
            self.decoder = Decoder1D()
        else:
            self.static_encoder = Encoder2D()
            self.dynamic_encoder = Encoder2D()
            self.decoder = Decoder1D()

        # Adversarial Models
        self.use_adv = use_adv
        if use_adv:
            self.priv_adv = Adversary_Emb(privacy_classes).to(device) # input = dynamic embedding, output = privacy class
            self.priv_coop = Adversary_Emb(privacy_classes).to(device) # input = static embedding, output = privacy class
            self.util_adv = Adversary_Emb(utility_classes).to(device) # input = static embedding, output = utility class
            self.util_coop = Adversary_Emb(utility_classes).to(device) # input = dynamic embedding, output = utility class
            self.discriminator = Discriminator().to(device)

            self.priv_optim = torch.optim.AdamW(self.priv_adv.parameters(), lr=adv_lr)
            self.priv_coop_optim = torch.optim.AdamW(self.priv_coop.parameters(), lr=adv_lr)
            self.util_optim = torch.optim.AdamW(self.util_adv.parameters(), lr=adv_lr)
            self.util_coop_optim = torch.optim.AdamW(self.util_coop.parameters(), lr=adv_lr)
            self.discriminator_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=adv_lr)

            # Freeze Adversarial Models
            self.priv_adv.eval()
            self.priv_coop.eval()
            self.util_adv.eval()
            self.util_coop.eval()
            self.discriminator.eval()

        # Loss Functions
        self.triplet_loss = nn.TripletMarginLoss()
        self.bce_loss = nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # Info for loss functions
        self.end_effectors = torch.tensor([19, 15, 23, 24, 21, 22, 3]).to(device) * 3
        self.chain_lengths = torch.tensor([5, 5, 8, 8, 8, 8, 5]).to(device)

        # Lambdas for discounted loss
        self.lambda_rec = 2
        self.lambda_cross = 0.1
        self.lambda_ee = 1
        self.lambda_smoothing = 3
        self.lambda_trip = 1
        self.lambda_latent = 10
        self.lambda_adv_util_coop = util_classifier_alpha
        self.lambda_adv_priv_coop = priv_classifier_alpha
        self.lambda_adv_util_adv = util_classifier_alpha
        self.lambda_adv_priv_adv = priv_classifier_alpha
        self.lambda_adv_disc = 1

        # Loss Toggles
        self.use_rec_loss = True
        self.use_cross_loss = True
        self.use_ee_loss = True 
        self.use_trip_loss_paired = True 
        self.use_trip_loss_unpaired = True
        self.use_smoothing_loss = True
        self.use_latent_consistency = True

    def get_loss_params(self):
        return {
            'lambda_rec': self.lambda_rec,
            'lambda_cross': self.lambda_cross,
            'lambda_ee': self.lambda_ee,
            'lambda_trip': self.lambda_trip,
            'lambda_latent': self.lambda_latent,
            'lambda_adv_util_coop': self.lambda_adv_util_coop,
            'lambda_adv_priv_coop': self.lambda_adv_priv_coop,
            'lambda_adv_util_adv': self.lambda_adv_util_adv,
            'lambda_adv_priv_adv': self.lambda_adv_priv_adv,
            'lambda_adv_disc': self.lambda_adv_disc,
            'use_rec_loss': self.use_rec_loss,
            'use_cross_loss': self.use_cross_loss,
            'use_ee_loss': self.use_ee_loss,
            'use_trip_loss_paired': self.use_trip_loss_paired,
            'use_trip_loss_unpaired': self.use_trip_loss_unpaired,
            'use_smoothing_loss': self.use_smoothing_loss,
            'use_latent_consistency': self.use_latent_consistency
        }

    def cross(self, x1, x1_rot, x2, x2_rot):
        d1 = self.dynamic_encoder(x1_rot)
        d2 = self.dynamic_encoder(x2_rot)
        s1 = self.static_encoder(x1)
        s2 = self.static_encoder(x2)
        
        x1_hat = self.decoder(torch.cat((d1, s1), dim=1))
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1))
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1))
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1))

        return x1_hat, x2_hat, y1_hat, y2_hat
    
    def eval(self, x1_rot, x2):
        dynamic = self.dynamic_encoder(x1_rot)
        static = self.static_encoder(x2)
        return self.decoder(torch.cat((dynamic, static), dim=1))

    def rec_loss(self, x, x_rot):
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x)
        x_hat = self.decoder(torch.cat((d, s), dim=1))
        if not one_dimension_conv:
            x_ = x.reshape(x.size(0), T, -1)
        return self.reconstruction_loss(x_, x_hat)
    
    def loss_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, cross = True, reconstruction = True, emb_adv = True, discrim_adv = True, verbose = False):
        d1 = self.dynamic_encoder(x1_rot) # A1
        d2 = self.dynamic_encoder(x2_rot) # A2
        s1 = self.static_encoder(x1) # P1
        s2 = self.static_encoder(x2) # P2

        x1_hat = self.decoder(torch.cat((d1, s1), dim=1)) # P1, A1
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1)) # P2, A2
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1)) # P2, A1
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1)) # P1, A2

        d12 = self.dynamic_encoder(y1_rot) # A1
        d21 = self.dynamic_encoder(y2_rot) # A2
        s12 = self.static_encoder(y1) # P2
        s21 = self.static_encoder(y2) # P1

        x1_hat_ = self.decoder(torch.cat((d12, s21), dim=1)) # P1, A1
        x2_hat_ = self.decoder(torch.cat((d21, s12), dim=1)) # P2, A2
        y1_hat_ = self.decoder(torch.cat((d12, s12), dim=1)) # P2, A1
        y2_hat_ = self.decoder(torch.cat((d21, s21), dim=1)) # P1, A2

        # x1_hat is reconstruction of x1
        # x2_hat is reconstruction of x2
        # y1_hat is cross reconstruction from x1 and x2
        # y2_hat is cross reconstruction from x2 and x1
        # x1_hat_ is cross reconstruction from y1 and y2
        # x2_hat_ is cross reconstruction from y2 and y1
        # y1_hat_ is reconstruction of y1
        # y2_hat_ is reconstruction of y2
        # d1 = A1
        # d2 = A2
        # d12 = A1
        # d21 = A2
        # s1 = P1
        # s2 = P2
        # s12 = P2
        # s21 = P1

        # flatten data if 2D
        if not one_dimension_conv:
            x1 = x1.view(x1.size(0), T, -1)
            x2 = x2.view(x2.size(0), T, -1)
            y1 = y1.view(y1.size(0), T, -1)
            y2 = y2.view(y2.size(0), T, -1)
        
        # initialize all losses to 0 tensor
        rec_loss = torch.zeros(1).to(device)
        cross_loss = torch.zeros(1).to(device)
        end_effector_loss = torch.zeros(1).to(device)
        triplet_loss = torch.zeros(1).to(device)
        smoothing_loss = torch.zeros(1).to(device)
        latent_consistency_loss = torch.zeros(1).to(device)
        privacy_loss = torch.zeros(1).to(device)
        privacy_loss_adv = torch.zeros(1).to(device)
        privacy_loss_coop = torch.zeros(1).to(device)
        utility_loss = torch.zeros(1).to(device)
        utility_loss_adv = torch.zeros(1).to(device)
        utility_loss_coop = torch.zeros(1).to(device)
        privacy_acc_adv = torch.zeros(1).to(device)
        privacy_acc_coop = torch.zeros(1).to(device)
        utility_acc_adv = torch.zeros(1).to(device)
        utility_acc_coop = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)
                        
        # reconstruction loss
        if self.use_rec_loss and reconstruction:
            rec_loss = (self.reconstruction_loss(x1, x1_hat) + self.reconstruction_loss(x2, x2_hat) + self.reconstruction_loss(y1, y1_hat_) + self.reconstruction_loss(y2, y2_hat_)) / 4
            if verbose: print('Reconstruction Loss: ', rec_loss.item())
        
        # cross reconstruction loss
        if self.use_cross_loss and cross:
            # could move this to its own function, but since cross is basically reconstruction, its fine like this
            cross_loss = (self.reconstruction_loss(y1, y1_hat) + self.reconstruction_loss(y2, y2_hat) + self.reconstruction_loss(x1, x1_hat_) + self.reconstruction_loss(x2, x2_hat_)) / 4
            if verbose: print('Cross Reconstruction Loss: ', cross_loss.item())
        
        # end effector loss
        if self.use_ee_loss:
            if reconstruction:
                end_effector_loss += (self.end_effector_loss(x1_hat, x1) + self.end_effector_loss(x2_hat, x2)) / 2
            if cross:
                end_effector_loss += (self.end_effector_loss(y1_hat, y1) + self.end_effector_loss(y2_hat, y2)) / 2
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # triplet loss
        if self.use_trip_loss_paired: # anchor, positive, negative
            # d1 = A1, d2 = A2, d12 = A1, d21 = A2
            # s1 = P1, s2 = P2, s12 = P2, s21 = P1
            # d12,s12 = y1, d21,s21 = y2
            # y1 = jk, y2 = il
            triplet_loss = self.triplet_loss(d12, d1, d2) \
                            + self.triplet_loss(d21, d2, d1) \
                            + self.triplet_loss(s12, s2, s1) \
                            + self.triplet_loss(s21, s1, s2) 
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        if self.use_smoothing_loss:
            smoothing_loss = (self.smoothing_loss(x1, x1_hat) + self.smoothing_loss(x2, x2_hat) + self.smoothing_loss(y1, y1_hat_) + self.smoothing_loss(y2, y2_hat_) + \
                                self.smoothing_loss(x1, x1_hat_) + self.smoothing_loss(x2, x2_hat_) + self.smoothing_loss(y1, y1_hat) + self.smoothing_loss(y2, y2_hat)) / 8
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())

        # latent consistency loss
        if self.use_latent_consistency:
            latent_consistency_loss = (self.latent_consistency_loss(d1, d12) + self.latent_consistency_loss(d2, d21) + self.latent_consistency_loss(s1, s21) + self.latent_consistency_loss(s2, s12)) / 4
            if verbose: print('Latent Consistency Loss: ', latent_consistency_loss.item())

        # adversarial loss
        if self.use_adv and emb_adv:
            actor_y1, actor_y2 = actors[0] - 1, actors[1] - 1
            actor_y1, actor_y2 = torch.eye(privacy_classes)[actor_y1.long()].to(device), torch.eye(privacy_classes)[actor_y2.long()].to(device)
            action_y1, action_y2 = actions[0] - 1, actions[1] - 1
            action_y1, action_y2 = torch.eye(utility_classes)[action_y1.long()].to(device), torch.eye(utility_classes)[action_y2.long()].to(device)

            # x1 => d1 s1
            # x2 => d2 s2

            # d1 => p1
            # d2 => p2
            # s1 => a1
            # s2 => a2
            
            # actor_y1 = p1
            # actor_y2 = p2

            # action_y1 = a1
            # action_y2 = a2


            # privacy loss (adversarial)
            privacy_loss_adv = (-self.adv_loss(self.priv_adv, d1, actor_y1) -self.adv_loss(self.priv_adv, d2, actor_y2))/2
            privacy_acc_adv = (self.adv_accuracy(self.priv_adv, d1, actor_y1) + self.adv_accuracy(self.priv_adv, d2, actor_y2))/2

            # privacy loss (coop)
            privacy_loss_coop = (self.adv_loss(self.priv_coop, s1, actor_y1) + self.adv_loss(self.priv_coop, s2, actor_y2))/2
            privacy_acc_coop = (self.adv_accuracy(self.priv_coop, s1, actor_y1) + self.adv_accuracy(self.priv_coop, s2, actor_y2))/2

            # utility loss (adversarial)
            utility_loss_adv = (-self.adv_loss(self.util_adv, s1, action_y1) -self.adv_loss(self.util_adv, s2, action_y2))/2
            utility_acc_adv = (self.adv_accuracy(self.util_adv, s1, action_y1) + self.adv_accuracy(self.util_adv, s2, action_y2))/2

            # utility loss (coop)
            utility_loss_coop = (self.adv_loss(self.util_coop, d1, action_y1) + self.adv_loss(self.util_coop, d2, action_y2))/2
            utility_acc_coop = (self.adv_accuracy(self.util_coop, d1, action_y1) + self.adv_accuracy(self.util_coop, d2, action_y2))/2

            privacy_loss = privacy_loss_adv * self.lambda_adv_priv_adv + privacy_loss_coop * self.lambda_adv_priv_coop
            utility_loss = utility_loss_adv * self.lambda_adv_util_adv + utility_loss_coop * self.lambda_adv_util_coop

            if verbose: 
                print('Privacy Loss (Adversarial): ', privacy_loss_adv.item(), '\tPrivacy Loss (Coop): ', privacy_loss_coop.item())
                print('Utility Loss (Adversarial): ', utility_loss_adv.item(), '\tUtility Loss (Coop): ', utility_loss_coop.item())
                print('Privacy Accuracy (Adversarial): ', privacy_acc_adv.item(), '\tPrivacy Accuracy (Coop): ', privacy_acc_coop.item())
                print('Utility Accuracy (Adversarial): ', utility_acc_adv.item(), '\tUtility Accuracy (Coop): ', utility_acc_coop.item())
            

        if self.use_adv and discrim_adv:
            # discrimnator (adversarial)
            discrim_out_fake = self.discriminator(torch.cat((x1_hat, x2_hat, y1_hat, y2_hat, x1_hat_, x2_hat_, y1_hat_, y2_hat_)))
            discriminator_loss = self.bce_loss(discrim_out_fake, torch.ones_like(discrim_out_fake))
            discriminator_acc = torch.sum(torch.round(discrim_out_fake) == 0).float() / (8 * batch_size)
            if verbose: print('Discriminator Loss: ', discriminator_loss.item(), '\tDiscriminator Accuracy: ', discriminator_acc.item())

        losses = {
            'rec_loss': rec_loss.item(),
            'cross_loss': cross_loss.item(),
            'end_effector_loss': end_effector_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'smoothing_loss': smoothing_loss.item(),
            'latent_consistency_loss': latent_consistency_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'privacy_loss_adv': privacy_loss_adv.item(),
            'privacy_loss_coop': privacy_loss_coop.item(),
            'privacy_acc_adv': privacy_acc_adv.item(),
            'privacy_acc_coop': privacy_acc_coop.item(),
            'utility_loss': utility_loss.item(),
            'utility_loss_adv': utility_loss_adv.item(),
            'utility_loss_coop': utility_loss_coop.item(),
            'utility_acc_adv': utility_acc_adv.item(),
            'utility_acc_coop': utility_acc_coop.item(),
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_acc': discriminator_acc.item()
        }

        return rec_loss * self.lambda_rec \
                + cross_loss * self.lambda_cross \
                + end_effector_loss * self.lambda_ee \
                + triplet_loss * self.lambda_trip \
                + latent_consistency_loss * self.lambda_latent \
                + privacy_loss \
                + utility_loss \
                + discriminator_loss * self.lambda_adv_disc \
                + smoothing_loss * self.lambda_smoothing, \
                x1_hat, x2_hat, y1_hat, y2_hat, losses

    def loss_unpaired(self, x_pos, x_rot, actors, actions, reconstruction = True, emb_adv = False, discrim_adv = False, ee = False, triplet = False, verbose = False):
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_pos)
        x_hat = self.decoder(torch.cat((d, s), dim=1))

        if not one_dimension_conv:
            x = x_pos.reshape(x_pos.size(0), T, -1)

        # initialize all losses to 0 tensor
        rec_loss = torch.zeros(1).to(device)
        end_effector_loss = torch.zeros(1).to(device)
        triplet_loss = torch.zeros(1).to(device)
        smoothing_loss = torch.zeros(1).to(device)
        privacy_loss = torch.zeros(1).to(device)
        privacy_loss_adv = torch.zeros(1).to(device)
        privacy_loss_coop = torch.zeros(1).to(device)
        utility_loss = torch.zeros(1).to(device)
        utility_loss_adv = torch.zeros(1).to(device)
        utility_loss_coop = torch.zeros(1).to(device)
        privacy_acc_adv = torch.zeros(1).to(device)
        privacy_acc_coop = torch.zeros(1).to(device)
        utility_acc_adv = torch.zeros(1).to(device)
        utility_acc_coop = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        # Reconstruction Loss
        if self.use_rec_loss and reconstruction:
            rec_loss = self.reconstruction_loss(x, x_hat)
            if verbose: print('Reconstruction Loss: ', rec_loss.item())

        # End Effector Loss
        if self.use_ee_loss and ee:
            end_effector_loss = self.end_effector_loss(x_hat, x)
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # Triplet Loss
        if self.use_trip_loss_unpaired and triplet: # anchor, positive, negative
            triplet_loss = (self.triplet_loss(d, d, s) + self.triplet_loss(s, s, d)) / 2
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        # Smoothing Loss
        if self.use_smoothing_loss:
            smoothing_loss = self.smoothing_loss(x, x_hat)
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())

        # Adversarial Loss
        if self.use_adv and emb_adv:
            actor_y = actors - 1
            actor_y = torch.eye(privacy_classes)[actor_y.long()].to(device)
            action_y = actions - 1
            action_y = torch.eye(utility_classes)[action_y.long()].to(device)

            # latent privacy loss (adv)
            privacy_loss_adv = -self.adv_loss(self.priv_adv, d, actor_y)
            privacy_acc_adv = self.adv_accuracy(self.priv_adv, d, actor_y)

            # latent privacy loss (coop)
            privacy_loss_coop = self.adv_loss(self.priv_coop, s, actor_y)
            privacy_acc_coop = self.adv_accuracy(self.priv_coop, s, actor_y)

            # latent utility loss (adv)
            utility_loss_adv = -self.adv_loss(self.util_adv, s, action_y)
            utility_acc_adv = self.adv_accuracy(self.util_adv, s, action_y)

            # latent utility loss (coop)
            utility_loss_coop = self.adv_loss(self.util_coop, d, action_y)
            utility_acc_coop = self.adv_accuracy(self.util_coop, d, action_y)

            privacy_loss = privacy_loss_adv * self.lambda_adv_priv_adv + privacy_loss_coop * self.lambda_adv_priv_coop
            utility_loss = utility_loss_adv * self.lambda_adv_util_adv + utility_loss_coop * self.lambda_adv_util_coop

            if verbose: 
                print('Privacy Loss Adv: ', privacy_loss_adv.item(), '\tPrivacy Loss Coop: ', privacy_loss_coop.item(), '\tPrivacy Loss: ', privacy_loss.item())
                print('Utility Loss Adv: ', utility_loss_adv.item(), '\tUtility Loss Coop: ', utility_loss_coop.item(), '\tUtility Loss: ', utility_loss.item())
                print('Privacy Accuracy Adv: ', privacy_acc_adv.item(), '\tPrivacy Accuracy Coop: ', privacy_acc_coop.item())
                print('Utility Accuracy Adv: ', utility_acc_adv.item(), '\tUtility Accuracy Coop: ', utility_acc_coop.item())


        if self.use_adv and discrim_adv:
            # discrimnator (adversarial)
            discrim_out_fake = self.discriminator(x_hat)
            discriminator_loss = self.bce_loss(discrim_out_fake, torch.ones_like(discrim_out_fake))
            discriminator_acc = torch.sum(torch.round(discrim_out_fake) == 0).float() / (batch_size)
            if verbose: print('Discriminator Loss: ', discriminator_loss.item(), '\tDiscriminator Accuracy: ', discriminator_acc.item())

        losses = {
            'rec_loss': rec_loss.item(),
            'end_effector_loss': end_effector_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'smoothing_loss': smoothing_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'privacy_loss_adv': privacy_loss_adv.item(),
            'privacy_loss_coop': privacy_loss_coop.item(),
            'privacy_acc_adv': privacy_acc_adv.item(),
            'privacy_acc_coop': privacy_acc_coop.item(),
            'utility_loss': utility_loss.item(),
            'utility_loss_adv': utility_loss_adv.item(),
            'utility_loss_coop': utility_loss_coop.item(),
            'utility_acc_adv': utility_acc_adv.item(),
            'utility_acc_coop': utility_acc_coop.item(),
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_acc': discriminator_acc.item()
        }

        return rec_loss * self.lambda_rec \
                + end_effector_loss * self.lambda_ee \
                + triplet_loss * self.lambda_trip \
                + privacy_loss \
                + utility_loss \
                + discriminator_loss * self.lambda_adv_disc \
                + smoothing_loss * self.lambda_smoothing, \
                x_hat, losses

    def reconstruction_loss(self, x, y):
        # return F.mse_loss(x, y)
        return torch.square(torch.norm(x - y, dim=1)).mean()
    
    def latent_consistency_loss(self, x, y):
        return F.mse_loss(x, y)
    
    def end_effector_loss(self, x, y):
        # slice to get the end effector joints
        x_ee = x[:, :, self.end_effectors.unsqueeze(-1) + torch.arange(3).to(device)] 
        y_ee = y[:, :, self.end_effectors.unsqueeze(-1) + torch.arange(3).to(device)]

        # calculate velocities
        x_vel = torch.norm(x_ee[:, 1:] - x_ee[:, :-1], dim=-1) / self.chain_lengths.unsqueeze(0)
        y_vel = torch.norm(y_ee[:, 1:] - y_ee[:, :-1], dim=-1) / self.chain_lengths.unsqueeze(0)
        
        # compute mse loss for each joint
        losses = F.mse_loss(x_vel, y_vel, reduction='none')

        # take sum over end effectors
        loss = losses.sum(dim=1)

        # take mean over batch
        loss = loss.mean()
        
        return loss
    
    def smoothing_loss(self, y, y_pred):
        # (batch, T, 75)
        # Calculate the squared sum of differences for y and y_pred
        diff_y = torch.sum(y[:, :-1] - y[:, 1:], dim=2) ** 2
        diff_y_pred = torch.sum(y_pred[:, :-1] - y_pred[:, 1:], dim=2) ** 2

        # Calculate the absolute difference
        abs_diff = torch.abs(diff_y - diff_y_pred)

        # Sum over all batches and sequence elements
        loss = torch.sum(abs_diff)

        # Normalize by the total number of elements (batch_size * sequence_length)
        total_loss = torch.sqrt(loss) / (y.size(0) * y.size(1))

        return total_loss

    def adv_loss(self, model, x, y):
        return self.cross_entropy(model(x), y)#.long().to(device))
    
    def adv_accuracy(self, model, x, y):
        return (model(x).argmax(dim=1) == y.argmax(dim=1).to(device)).float().mean()

    def train_adv_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, train_emb = True, train_discrim = True):
        if not self.use_adv: return 0,0
        # freeze encoders/decoder
        self.dynamic_encoder.eval()
        self.static_encoder.eval()
        self.decoder.eval()

        # unfreeze adversaries
        self.priv_adv.train()
        self.util_adv.train()
        self.discriminator.train()

        # zero out gradients
        self.priv_optim.zero_grad()
        self.util_optim.zero_grad()
        self.discriminator_optim.zero_grad()

        # encode
        d1 = self.dynamic_encoder(x1_rot) # A1
        d2 = self.dynamic_encoder(x2_rot) # A2
        d3 = self.dynamic_encoder(y1_rot) # A2
        d4 = self.dynamic_encoder(y2_rot) # A1
        s1 = self.static_encoder(x1) # P1
        s2 = self.static_encoder(x2) # P2
        s3 = self.static_encoder(y1) # P1
        s4 = self.static_encoder(y2) # P2

        # decode
        x1_hat = self.decoder(torch.cat((d1, s1), dim=1)) # P1, A1
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1)) # P2, A2
        y1_hat = self.decoder(torch.cat((d3, s3), dim=1)) # P1, A2
        y2_hat = self.decoder(torch.cat((d4, s4), dim=1)) # P2, A1

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            # train privacy adversary
            p1, p2 = actors[0] - 1, actors[1] - 1
            p1, p2 = torch.eye(privacy_classes)[p1.long()].to(device), torch.eye(privacy_classes)[p2.long()].to(device)
            priv_loss = (self.cross_entropy(self.priv_adv(d1), p1) + \
                        self.cross_entropy(self.priv_adv(d2), p2) + \
                        self.cross_entropy(self.priv_adv(d3), p1) + \
                        self.cross_entropy(self.priv_adv(d4), p2)) / 4
            priv_acc = (self.adv_accuracy(self.priv_adv, d1, p1) + self.adv_accuracy(self.priv_adv, d2, p2) + self.adv_accuracy(self.priv_adv, d3, p1) + self.adv_accuracy(self.priv_adv, d4, p2)) / 4
            priv_loss.backward(retain_graph=True)
            self.priv_optim.step()

            # train privacy cooperative
            priv_coop_loss = (self.cross_entropy(self.priv_coop(s1), p1) + \
                            self.cross_entropy(self.priv_coop(s2), p2) + \
                            self.cross_entropy(self.priv_coop(s3), p1) + \
                            self.cross_entropy(self.priv_coop(s4), p2)) / 4
            priv_coop_acc = (self.adv_accuracy(self.priv_coop, s1, p1) + self.adv_accuracy(self.priv_coop, s2, p2) + self.adv_accuracy(self.priv_coop, s3, p1) + self.adv_accuracy(self.priv_coop, s4, p2)) / 4
            priv_coop_loss.backward(retain_graph=True)
            self.priv_coop_optim.step()
                        
            # train utility adversary
            a1, a2 = actions[0] - 1, actions[1] - 1
            a1, a2 = torch.eye(utility_classes)[a1.long()].to(device), torch.eye(utility_classes)[a2.long()].to(device)
            util_loss = (self.cross_entropy(self.util_adv(s1), a1) + \
                        self.cross_entropy(self.util_adv(s2), a2) + \
                        self.cross_entropy(self.util_adv(s3), a2) + \
                        self.cross_entropy(self.util_adv(s4), a1)) / 4
            util_acc = (self.adv_accuracy(self.util_adv, s1, a1) + self.adv_accuracy(self.util_adv, s2, a2) + self.adv_accuracy(self.util_adv, s3, a2) + self.adv_accuracy(self.util_adv, s4, a1)) / 4
            util_loss.backward(retain_graph=True)
            self.util_optim.step()

            # train utility cooperative
            util_coop_loss = (self.cross_entropy(self.util_coop(d1), a1) + \
                            self.cross_entropy(self.util_coop(d2), a2) + \
                            self.cross_entropy(self.util_coop(d3), a2) + \
                            self.cross_entropy(self.util_coop(d4), a1)) / 4
            util_coop_acc = (self.adv_accuracy(self.util_coop, d1, a1) + self.adv_accuracy(self.util_coop, d2, a2) + self.adv_accuracy(self.util_coop, d3, a2) + self.adv_accuracy(self.util_coop, d4, a1)) / 4
            util_coop_loss.backward(retain_graph=True)
            self.util_coop_optim.step()


        if train_discrim:
            # train discriminator
            output_real = self.discriminator(torch.cat((x1.view(x1.size(0), T, -1), x2.view(x2.size(0), T, -1), y1.view(y1.size(0), T, -1), y2.view(y1.size(0), T, -1))))
            output_fake = self.discriminator(torch.cat((x1_hat, x2_hat, y1_hat, y2_hat)))
            discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + self.bce_loss(output_fake, torch.zeros_like(output_fake))
            discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / (4 * batch_size)) + (torch.sum(torch.round(output_real) == 1).float() / (4 * batch_size))) / 2
            discriminator_loss.backward()
            self.discriminator_optim.step()

        # unfreeze encoders/decoder
        self.dynamic_encoder.train()
        self.static_encoder.train()
        self.decoder.train()

        # freeze adversaries
        self.priv_adv.eval()
        self.priv_coop.eval()
        self.util_adv.eval()
        self.util_coop.eval()
        self.discriminator.eval()

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def train_adv_unpaired(self, x_pos, x_rot, actor, action, train_emb = True, train_discrim = True):
        # ensure one training method is enabled
        assert train_emb or train_discrim, 'At least one training method must be enabled'

        # freeze encoders/decoder
        self.dynamic_encoder.eval()
        self.static_encoder.eval()
        self.decoder.eval()

        # unfreeze adversaries
        self.priv_adv.train()
        self.priv_coop.train()
        self.util_adv.train()
        self.util_coop.train()
        self.discriminator.train()

        # zero out gradients
        self.priv_optim.zero_grad()
        self.priv_coop_optim.zero_grad()
        self.util_optim.zero_grad()
        self.util_coop_optim.zero_grad()
        self.discriminator_optim.zero_grad()

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            p = actor - 1
            p = torch.eye(privacy_classes)[p.long()].to(device)
            a = action - 1
            a = torch.eye(utility_classes)[a.long()].to(device)

            # train privacy adversary
            priv_loss = self.adv_loss(self.priv_adv, self.dynamic_encoder(x_rot), p)
            priv_acc = self.adv_accuracy(self.priv_adv, self.dynamic_encoder(x_rot), p)
            priv_loss.backward()
            self.priv_optim.step()

            # tain privacy cooperative
            priv_coop_loss = self.adv_loss(self.priv_coop, self.static_encoder(x_pos), p)
            priv_coop_acc = self.adv_accuracy(self.priv_coop, self.static_encoder(x_pos), p)
            priv_coop_loss.backward()
            self.priv_coop_optim.step()
            
            # train utility adversary
            util_loss = self.adv_loss(self.util_adv, self.static_encoder(x_pos), a)
            util_acc = self.adv_accuracy(self.util_adv, self.static_encoder(x_pos), a)
            util_loss.backward()
            self.util_optim.step()

            # train utility cooperative
            util_coop_loss = self.adv_loss(self.util_coop, self.dynamic_encoder(x_rot), a)
            util_coop_acc = self.adv_accuracy(self.util_coop, self.dynamic_encoder(x_rot), a)
            util_coop_loss.backward()
            self.util_coop_optim.step()

        if train_discrim:
            # encode
            d = self.dynamic_encoder(x_rot)
            s = self.static_encoder(x_pos)

            # decode
            x_hat = self.decoder(torch.cat((d, s), dim=1))

            # train discriminator
            output_real = self.discriminator(x_pos.reshape(x_pos.size(0), T, -1))
            output_fake = self.discriminator(x_hat)
            discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + self.bce_loss(output_fake, torch.zeros_like(output_fake))
            discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / batch_size) + (torch.sum(torch.round(output_real) == 1).float() / batch_size)) / 2
            discriminator_loss.backward()
            self.discriminator_optim.step()

        # unfreeze encoders/decoder
        self.dynamic_encoder.train()
        self.static_encoder.train()
        self.decoder.train()

        # freeze adversaries
        self.priv_adv.eval()
        self.priv_coop.eval()
        self.util_adv.eval()
        self.util_coop.eval()
        self.discriminator.eval()

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def val_adv_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, train_emb = True, train_discrim = True):
        if not self.use_adv: return 0,0

        # freeze encoders/decoder
        self.set_eval()

        # Encode
        d1, d2, d3, d4 = [self.dynamic_encoder(x) for x in [x1_rot, x2_rot, y1_rot, y2_rot]]
        s1, s2, s3, s4 = [self.static_encoder(x) for x in [x1, x2, y1, y2]]

        # Decode
        x1_hat, x2_hat, y1_hat, y2_hat = [self.decoder(torch.cat((d, s), dim=1)) for d, s in zip([d1, d2, d3, d4], [s1, s2, s3, s4])]

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            # privacy adversary
            p1, p2 = actors[0] - 1, actors[1] - 1
            p1, p2 = torch.eye(privacy_classes)[p1.long()].to(device), torch.eye(privacy_classes)[p2.long()].to(device)
            priv_loss = (self.cross_entropy(self.priv_adv(d1), p1) + \
                        self.cross_entropy(self.priv_adv(d2), p2) + \
                        self.cross_entropy(self.priv_adv(d3), p1) + \
                        self.cross_entropy(self.priv_adv(d4), p2)) / 4
            priv_acc = (self.adv_accuracy(self.priv_adv, d1, p1) + self.adv_accuracy(self.priv_adv, d2, p2) + self.adv_accuracy(self.priv_adv, d3, p1) + self.adv_accuracy(self.priv_adv, d4, p2)) / 4

            # privacy cooperative
            priv_coop_loss = (self.cross_entropy(self.priv_coop(s1), p1) + \
                            self.cross_entropy(self.priv_coop(s2), p2) + \
                            self.cross_entropy(self.priv_coop(s3), p1) + \
                            self.cross_entropy(self.priv_coop(s4), p2)) / 4
            priv_coop_acc = (self.adv_accuracy(self.priv_coop, s1, p1) + self.adv_accuracy(self.priv_coop, s2, p2) + self.adv_accuracy(self.priv_coop, s3, p1) + self.adv_accuracy(self.priv_coop, s4, p2)) / 4
                        
            # utility adversary
            a1, a2 = actions[0] - 1, actions[1] - 1
            a1, a2 = torch.eye(utility_classes)[a1.long()].to(device), torch.eye(utility_classes)[a2.long()].to(device)
            util_loss = (self.cross_entropy(self.util_adv(s1), a1) + \
                        self.cross_entropy(self.util_adv(s2), a2) + \
                        self.cross_entropy(self.util_adv(s3), a2) + \
                        self.cross_entropy(self.util_adv(s4), a1)) / 4
            util_acc = (self.adv_accuracy(self.util_adv, s1, a1) + self.adv_accuracy(self.util_adv, s2, a2) + self.adv_accuracy(self.util_adv, s3, a2) + self.adv_accuracy(self.util_adv, s4, a1)) / 4

            # utility cooperative
            util_coop_loss = (self.cross_entropy(self.util_coop(d1), a1) + \
                            self.cross_entropy(self.util_coop(d2), a2) + \
                            self.cross_entropy(self.util_coop(d3), a2) + \
                            self.cross_entropy(self.util_coop(d4), a1)) / 4
            util_coop_acc = (self.adv_accuracy(self.util_coop, d1, a1) + self.adv_accuracy(self.util_coop, d2, a2) + self.adv_accuracy(self.util_coop, d3, a2) + self.adv_accuracy(self.util_coop, d4, a1)) / 4


        if train_discrim:
            # discriminator
            output_real = self.discriminator(torch.cat((x1.view(x1.size(0), T, -1), x2.view(x2.size(0), T, -1), y1.view(y1.size(0), T, -1), y2.view(y1.size(0), T, -1))))
            output_fake = self.discriminator(torch.cat((x1_hat, x2_hat, y1_hat, y2_hat)))
            discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + self.bce_loss(output_fake, torch.zeros_like(output_fake))
            discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / (4 * batch_size)) + (torch.sum(torch.round(output_real) == 1).float() / (4 * batch_size))) / 2

        # unfreeze encoders/decoder
        self.set_eval(False)

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def val_adv_unpaired(self, x_pos, x_rot, actor, action, train_emb = True, train_discrim = True):
        # ensure one training method is enabled
        assert train_emb or train_discrim, 'At least one training method must be enabled'

        # freeze encoders/decoder
        self.set_eval()

        # Encode
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_pos)

        # Decode
        x_hat = self.decoder(torch.cat((d, s), dim=1))

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            p = actor - 1
            p = torch.eye(privacy_classes)[p.long()].to(device)
            a = action - 1
            a = torch.eye(utility_classes)[a.long()].to(device)

            # privacy adversary
            priv_loss = self.adv_loss(self.priv_adv, self.dynamic_encoder(x_rot), p)
            priv_acc = self.adv_accuracy(self.priv_adv, self.dynamic_encoder(x_rot), p)

            # privacy cooperative
            priv_coop_loss = self.adv_loss(self.priv_coop, self.static_encoder(x_pos), p)
            priv_coop_acc = self.adv_accuracy(self.priv_coop, self.static_encoder(x_pos), p)
            
            # utility adversary
            util_loss = self.adv_loss(self.util_adv, self.static_encoder(x_pos), a)
            util_acc = self.adv_accuracy(self.util_adv, self.static_encoder(x_pos), a)

            # utility cooperative
            util_coop_loss = self.adv_loss(self.util_coop, self.dynamic_encoder(x_rot), a)
            util_coop_acc = self.adv_accuracy(self.util_coop, self.dynamic_encoder(x_rot), a)

        if train_discrim:
            # encode
            d = self.dynamic_encoder(x_rot)
            s = self.static_encoder(x_pos)

            # decode
            x_hat = self.decoder(torch.cat((d, s), dim=1))

            # train discriminator
            output_real = self.discriminator(x_pos.reshape(x_pos.size(0), T, -1))
            output_fake = self.discriminator(x_hat)
            discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + self.bce_loss(output_fake, torch.zeros_like(output_fake))
            discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / batch_size) + (torch.sum(torch.round(output_real) == 1).float() / batch_size)) / 2

        # unfreeze encoders/decoder
        self.set_eval(False)

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def forward(self, x, x_rot):
        dyn = self.dynamic_encoder(x_rot)
        sta = self.static_encoder(x)
        x = self.decoder(torch.cat((dyn, sta), dim=1))
        return x
    
    def set_eval(self, eval=True):
        if eval:
            self.static_encoder.eval()
            self.dynamic_encoder.eval()
            self.decoder.eval()
            self.priv_adv.eval()
            self.priv_coop.eval()
            self.util_adv.eval()
            self.util_coop.eval()
            self.discriminator.eval()
        else:
            self.static_encoder.train()
            self.dynamic_encoder.train()
            self.decoder.train()

# %% [markdown]
# ## Utility/Privacy Evaluation

# %%
def test(test_loader, model, k=3):
    acces = AverageMeter()
    topk_acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.eval()

    label_output = list()
    pred_output = list()

    for i, t in enumerate(test_loader):
        inputs = t[0]
        target = t[1]
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view(
                (-1, inputs.size(0)//target.size(0), output.size(1)))
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
    target = torch.argmax(target, dim=1)  # Add this line to convert one-hot targets to class indices
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
    # Data loading
    ntu_loaders = NTUDataLoaders(dataset, case, seg=20, train_X=train_x, train_Y=train_y, test_X=test_x, test_Y=test_y, val_X=val_x, val_Y=val_y, aug=0)
    test_loader = ntu_loaders.get_test_loader(batch_size, 16)

    # Test
    return test(test_loader, model, k=k)

def run_sgn_gender_eval(train_x, train_y, test_x, test_y, val_x, val_y, model, k=1):
    # Data loading
    ntu_loaders = NTUDataLoaders(dataset, 0, seg=20, train_X=train_x, train_Y=train_y, test_X=test_x, test_Y=test_y, val_X=val_x, val_Y=val_y, aug=0)
    test_loader = ntu_loaders.get_test_loader(batch_size, 16)

    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.eval()

    label_output = list()
    pred_output = list()

    for i, t in enumerate(test_loader):
        inputs = t[0]
        target = t[1]
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view(
                (-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy(output.data, target.cuda())
        acces.update(acc[0], inputs.size(0))

    label_output = np.concatenate(label_output, axis=0)
    pred_output = np.concatenate(pred_output, axis=0)

    label_index = np.argmax(label_output, axis=1)
    pred_index = np.argmax(pred_output, axis=1)

    f1 = f1_score(label_index, pred_index, average='macro', zero_division=0)

    return acces.avg, f1

# %% [markdown]
# ## Instantiate Models

# %%
model = AutoEncoder(adv_lr=adv_lr).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# %%
load_model = True
if load_model:
    model.load_state_dict(torch.load('pretrained/PMR_NTU120.pt'))
    # model.load_state_dict(torch.load('pretrained/MR.pt'))

# %%
load_util = True
sgn_ar = SGN(utility_classes, None, seg, batch_size, 0).to(device)
sgn_priv = SGN(120, None, seg, batch_size, 0).to(device)

if ntu_120:
    if only_use_pos: # Assumes SGN preprocessing
        sgn_priv.load_state_dict(torch.load("C:\\Users\\tcarr23\\Local Code\\SGN\\results\\NTU120ri\\SGN\\1_best.pth")['state_dict'])
        sgn_ar.load_state_dict(torch.load("C:\\Users\\tcarr23\\Local Code\\SGN\\results\\NTU120ar\\SGN\\1_best.pth")['state_dict'])
    else:
        sgn_priv.load_state_dict(torch.load('SGN/pretrained/privacy.pt')['state_dict'])
        sgn_ar.load_state_dict(torch.load('SGN/pretrained/action.pt')['state_dict'])
else:
    if only_use_pos:
        if remove_two_actor_actions: sgn_ar.load_state_dict(torch.load('SGN/pretrained/action_60_sgnpt_no_two_actor.pt')['state_dict'])
        else: sgn_ar.load_state_dict(torch.load('SGN/pretrained/action_60_sgnpt.pt')['state_dict'])
        sgn_priv.load_state_dict(torch.load('SGN/pretrained/privacy_60_sgnpt.pt')['state_dict'])
    else: 
        sgn_priv.load_state_dict(torch.load('SGN/pretrained/privacy_60.pt')['state_dict'])
        sgn_ar.load_state_dict(torch.load('SGN/pretrained/action_60.pt')['state_dict'])

# %% [markdown]
# # Training

# %% [markdown]
# ## Train Motion Retargeting

# %%
sgn_train_x, sgn_train_y, sgn_val_x, sgn_val_y = np.zeros((batch_size, 300, 150)), np.zeros((batch_size, 1)), np.zeros((batch_size, 300, 150)), np.zeros((batch_size, 1))

best_metric = float('inf')
total_epochs = -1
cur_tot_epoch = 0

def train_paired(train_ae = True, train_cross = True, train_discrim = True, train_emb_adv = True, run_eval = True, use_emb_adv = True, use_discrim_adv = True, run_sgn_eval = False, save = True, k=3):
    global best_metric
    global total_epochs
    global cur_tot_epoch
    # Assertions
    # assert train_ae or train_cross or train_discrim or train_emb_adv, "At least one of the training objectives must be True"
    assert not (run_sgn_eval and not run_eval), "If run_sgn_eval is True, then run_eval must be True"
    
    # Store eval values for validation
    eval_X_known, eval_Y_known_action, eval_Y_known_actor, eval_X_rec, eval_Y_rec_action, eval_Y_rec_actor, eval_X, eval_Y_action, eval_Y_actor, eval_Y_initial_actor = [], [], [], [], [], [], [], [], [], []

    # Losses for printing
    losses = []
    rec_loss, cross_loss, end_effector_loss, smoothing_loss, triplet_loss, latent_consistency_loss, privacy_loss, privacy_loss_adv, privacy_loss_coop, privacy_acc_adv, privacy_acc_coop, priv_training_loss, utility_loss, utility_loss_adv, utility_loss_coop, utility_acc_adv, utility_acc_coop, util_training_loss, discriminator_loss, discriminator_train_losses, discriminator_training_acc, priv_coop_training_loss, priv_training_acc, priv_coop_training_acc, util_coop_training_loss, util_training_acc, util_coop_training_acc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    # Determine if adversaries need to be trained
    train_emb_this_epoch = True
    if emb_clf_update_per_epoch_paired < 1:
        if cur_tot_epoch % round(1 / emb_clf_update_per_epoch_paired) != 0:
            train_emb_this_epoch = False

    for (x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions) in train_dl:
        # Move tensors to the configured device
        x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot = x1_pos.float().to(device), x1_rot.float().to(device), x2_pos.float().to(device), x2_rot.float().to(device), y1_pos.float().to(device), y1_rot.float().to(device), y2_pos.float().to(device), y2_rot.float().to(device)
        
        # Remove rotation data if only using position data
        if only_use_pos:
            x1_rot, x2_rot, y1_rot, y2_rot = x1_pos, x2_pos, y1_pos, y2_pos

        # For 1D convolutions, flatten the data
        if one_dimension_conv:
            x1_pos = x1_pos.view(x1_pos.size(0), T, -1)
            x1_rot = x1_rot.view(x1_rot.size(0), T, -1)
            x2_pos = x2_pos.view(x2_pos.size(0), T, -1)
            x2_rot = x2_rot.view(x2_rot.size(0), T, -1)
            y1_pos = y1_pos.view(y1_pos.size(0), T, -1)
            y1_rot = y1_rot.view(y1_rot.size(0), T, -1)
            y2_pos = y2_pos.view(y2_pos.size(0), T, -1)
            y2_rot = y2_rot.view(y2_rot.size(0), T, -1)

        
        if train_discrim or train_emb_adv:
            # Train the discriminator
            if train_emb_this_epoch:
                it = 1
                if emb_clf_update_per_epoch_paired > 1: it = emb_clf_update_per_epoch_paired
                for _ in range(int(it)):
                    t_priv_loss, t_priv_coop_loss, t_util_loss, t_util_coop_loss, t_discriminator_loss, t_priv_acc, t_util_acc, t_priv_coop_acc, t_util_coop_acc, t_discriminator_acc  = model.train_adv_paired(x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions, train_emb=train_emb_adv, train_discrim=train_discrim)
                
                # Track the loss
                priv_training_loss.append(t_priv_loss)
                priv_coop_training_loss.append(t_priv_coop_loss)
                priv_training_acc.append(t_priv_acc)
                priv_coop_training_acc.append(t_priv_coop_acc)
                util_training_loss.append(t_util_loss)
                util_coop_training_loss.append(t_util_coop_loss)
                util_training_acc.append(t_util_acc)
                util_coop_training_acc.append(t_util_coop_acc)
                discriminator_train_losses.append(t_discriminator_loss)
                discriminator_training_acc.append(t_discriminator_acc)

        # Zero the gradients
        optimizer.zero_grad()

        # Train the autoencoder/cross reconstruction
        if train_ae or train_cross:
            # Forward pass
            loss, _, _, _, _, losses_ = model.loss_paired(x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions, cross=train_cross, reconstruction=train_ae, emb_adv=use_emb_adv, discrim_adv=use_discrim_adv)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track the loss
            losses.append(loss.item())
            rec_loss.append(losses_['rec_loss'])
            cross_loss.append(losses_['cross_loss'])
            end_effector_loss.append(losses_['end_effector_loss'])
            smoothing_loss.append(losses_['smoothing_loss'])
            latent_consistency_loss.append(losses_['latent_consistency_loss'])
            triplet_loss.append(losses_['triplet_loss'])
            privacy_loss.append(losses_['privacy_loss'])
            privacy_loss_adv.append(losses_['privacy_loss_adv'])
            privacy_loss_coop.append(losses_['privacy_loss_coop'])
            privacy_acc_adv.append(losses_['privacy_acc_adv'])
            privacy_acc_coop.append(losses_['privacy_acc_coop'])
            utility_loss.append(losses_['utility_loss'])
            utility_loss_adv.append(losses_['utility_loss_adv'])
            utility_loss_coop.append(losses_['utility_loss_coop'])
            utility_acc_adv.append(losses_['utility_acc_adv'])
            utility_acc_coop.append(losses_['utility_acc_coop'])
            discriminator_loss.append(losses_['discriminator_loss'])
            discriminator_training_acc.append(losses_['discriminator_acc'])
        
    # Decay learning rate (disabled for training stages)
    # scheduler.step() 

    # Validation
    if run_eval:
        with torch.no_grad():
            val_losses = []
            val_rec_loss, val_cross_loss, val_end_effector_loss, val_smoothing_loss, val_triplet_loss, val_latent_consistency_loss, val_privacy_loss, val_privacy_loss_adv, val_privacy_loss_coop, val_privacy_acc_adv, val_privacy_acc_coop, val_utility_loss, val_utility_loss_adv, val_utility_loss_coop, val_utility_acc_adv, val_utility_acc_coop, val_discriminator_loss, val_discriminator_acc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            
            for (x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions) in val_dl:
                x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot = x1_pos.float().to(device), x1_rot.float().to(device), x2_pos.float().to(device), x2_rot.float().to(device), y1_pos.float().to(device), y1_rot.float().to(device), y2_pos.float().to(device), y2_rot.float().to(device)

                # Remove rotation data if only using position data
                if only_use_pos:
                    x1_rot, x2_rot, y1_rot, y2_rot = x1_pos, x2_pos, y1_pos, y2_pos

                # For 1D convolutions, flatten the data
                if one_dimension_conv:
                    x1_pos = x1_pos.view(x1_pos.size(0), T, -1)
                    x1_rot = x1_rot.view(x1_rot.size(0), T, -1)
                    x2_pos = x2_pos.view(x2_pos.size(0), T, -1)
                    x2_rot = x2_rot.view(x2_rot.size(0), T, -1)
                    y1_pos = y1_pos.view(y1_pos.size(0), T, -1)
                    y1_rot = y1_rot.view(y1_rot.size(0), T, -1)
                    y2_pos = y2_pos.view(y2_pos.size(0), T, -1)
                    y2_rot = y2_rot.view(y2_rot.size(0), T, -1)
                
                loss, x1_hat, x2_hat, y1_hat, y2_hat, losses_ = model.loss_paired(x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions, cross=train_cross, reconstruction=train_ae, emb_adv=use_emb_adv, discrim_adv=use_discrim_adv)
                val_losses.append(loss.item())
                val_rec_loss.append(losses_['rec_loss'])
                val_cross_loss.append(losses_['cross_loss'])
                val_end_effector_loss.append(losses_['end_effector_loss'])
                val_smoothing_loss.append(losses_['smoothing_loss'])
                val_triplet_loss.append(losses_['triplet_loss'])
                val_latent_consistency_loss.append(losses_['latent_consistency_loss'])
                val_privacy_loss.append(losses_['privacy_loss'])
                val_privacy_loss_adv.append(losses_['privacy_loss_adv'])
                val_privacy_loss_coop.append(losses_['privacy_loss_coop'])
                val_privacy_acc_adv.append(losses_['privacy_acc_adv'])
                val_privacy_acc_coop.append(losses_['privacy_acc_coop'])
                val_utility_loss.append(losses_['utility_loss'])
                val_utility_loss_adv.append(losses_['utility_loss_adv'])
                val_utility_loss_coop.append(losses_['utility_loss_coop'])
                val_utility_acc_adv.append(losses_['utility_acc_adv'])
                val_utility_acc_coop.append(losses_['utility_acc_coop'])
                val_discriminator_loss.append(losses_['discriminator_loss'])
                val_discriminator_acc.append(losses_['discriminator_acc'])

                if run_sgn_eval:
                    if not one_dimension_conv:
                        x1_pos = x1_pos.view(x1_pos.size(0), T, -1)
                        x2_pos = x2_pos.view(x2_pos.size(0), T, -1)
                        y1_pos = y1_pos.view(y1_pos.size(0), T, -1)
                        y2_pos = y2_pos.view(y2_pos.size(0), T, -1)

                    # x1 = P1, A1
                    # x2 = P2, A2
                    # y1 = P1, A2
                    # y2 = P2, A1
                    # actors = A1, A2
                    # actions = P1, P2
                    # x1hat = P1, A1
                    # x2hat = P2, A2
                    # y1hat = P2, A1
                    # y2hat = P1, A2
                        
                    # Raw Data X
                    eval_X_known.append(x1_pos.cpu().numpy())
                    eval_X_known.append(x2_pos.cpu().numpy())
                    eval_X_known.append(y1_pos.cpu().numpy())
                    eval_X_known.append(y2_pos.cpu().numpy())

                    # Raw Data Utility
                    eval_Y_known_action.append(actions[0].cpu().numpy())
                    eval_Y_known_action.append(actions[1].cpu().numpy())
                    eval_Y_known_action.append(actions[1].cpu().numpy())
                    eval_Y_known_action.append(actions[0].cpu().numpy())

                    # Raw Data Privacy
                    eval_Y_known_actor.append(actors[0].cpu().numpy())
                    eval_Y_known_actor.append(actors[1].cpu().numpy())
                    eval_Y_known_actor.append(actors[0].cpu().numpy())
                    eval_Y_known_actor.append(actors[1].cpu().numpy())

                    # Reconstruction X
                    eval_X_rec.append(x1_hat.cpu().numpy())
                    eval_X_rec.append(x2_hat.cpu().numpy())
                    # Cross X
                    eval_X.append(y1_hat.cpu().numpy()) # P2, A1
                    eval_X.append(y2_hat.cpu().numpy()) # P1, A2

                    # Reconstruction Utility
                    eval_Y_rec_action.append(actions[0].cpu().numpy())
                    eval_Y_rec_action.append(actions[1].cpu().numpy())
                    # Cross Utility
                    eval_Y_action.append(actions[0].cpu().numpy())
                    eval_Y_action.append(actions[1].cpu().numpy())

                    # Reconstruction Privacy
                    eval_Y_rec_actor.append(actors[0].cpu().numpy())
                    eval_Y_rec_actor.append(actors[1].cpu().numpy())
                    # Cross Privacy
                    eval_Y_actor.append(actors[1].cpu().numpy())
                    eval_Y_actor.append(actors[0].cpu().numpy())
                    # Initial Privacy
                    eval_Y_initial_actor.append(actors[0].cpu().numpy())
                    eval_Y_initial_actor.append(actors[1].cpu().numpy())

    # Print loss/accuracy
    print(f'--------------------\nEpoch {cur_tot_epoch+1}/{total_epochs}\n--------------------')
    cur_tot_epoch += 1
    
    if train_ae or train_cross:
        print(f'Training Loss:\t\t\t{np.mean(losses)}')
        if run_eval: print(f'Validation Loss:\t\t{np.mean(val_losses)}')
        print('\nTraining Losses:')
        print(f'Reconstruction Loss:\t\t{np.mean(rec_loss)}\nCross Reconstruction Loss:\t{np.mean(cross_loss)}\nEnd Effector Loss:\t\t{np.mean(end_effector_loss)}\nSmoothing Loss:\t\t\t{np.mean(smoothing_loss)}\nTriplet Loss:\t\t\t{np.mean(triplet_loss)}\nLatent Consistency Loss:\t{np.mean(latent_consistency_loss)}')
        if use_emb_adv:
            print(f'Privacy Loss:\t\t\t{np.mean(privacy_loss)}\nPrivacy Loss Dyn:\t\t{np.mean(privacy_loss_adv)}\nPrivacy Loss Stat:\t\t{np.mean(privacy_loss_coop)}')
            print(f'Utility Loss:\t\t\t{np.mean(utility_loss)}\nUtility Loss Dyn:\t\t{np.mean(utility_loss_adv)}\nUtility Loss Stat:\t\t{np.mean(utility_loss_coop)}')
        if use_discrim_adv: print(f'Discriminator Loss:\t\t{np.mean(discriminator_loss)}')

    if run_eval:
        print('\nValidation Losses:')
        print(f'Val Reconstruction Loss:\t{np.mean(val_rec_loss)}\nVal Cross Reconstruction Loss:\t{np.mean(val_cross_loss)}\nVal End Effector Loss:\t\t{np.mean(val_end_effector_loss)}\nVal Smoothing Loss:\t\t{np.mean(val_smoothing_loss)}\nVal Triplet Loss:\t\t{np.mean(val_triplet_loss)}\nVal Latent Consistency Loss:\t{np.mean(val_latent_consistency_loss)}')
        if use_emb_adv:
            print(f'Val Privacy Loss:\t\t{np.mean(val_privacy_loss)}\nVal Privacy Loss Dyn:\t\t{np.mean(val_privacy_loss_adv)}\nVal Privacy Loss Stat:\t\t{np.mean(val_privacy_loss_coop)}')
            print(f'Val Utility Loss:\t\t{np.mean(val_utility_loss)}\nVal Utility Loss Dyn:\t\t{np.mean(val_utility_loss_adv)}\nVal Utility Loss Stat:\t\t{np.mean(val_utility_loss_coop)}')
        if use_discrim_adv: print(f'Val Discriminator Loss:\t\t{np.mean(val_discriminator_loss)}')
    
    if train_emb_adv or train_discrim:
        print('\nEmbedding Classifers')
        if train_emb_adv and train_emb_this_epoch:
            print(f'Adv Privacy Training Loss:\t\t{np.mean(priv_training_loss)}\nAdv Utility Training Loss:\t\t{np.mean(util_training_loss)}\nCoop Privacy Training Loss:\t{np.mean(priv_coop_training_loss)}\nCoop Utility Training Loss:\t{np.mean(util_coop_training_loss)}\nDiscriminator Training Loss:\t{np.mean(discriminator_train_losses)}')
            print(f'Adv Privacy Training Acc:\t\t{np.mean(priv_training_acc)}\nAdv Utility Training Acc:\t\t{np.mean(util_training_acc)}\nCoop Privacy Training Acc:\t\t{np.mean(priv_coop_training_acc)}\nCoop Utility Training Acc:\t\t{np.mean(util_coop_training_acc)}\nDiscriminator Training Acc:\t{np.mean(discriminator_training_acc)}')
            if train_ae or train_cross: print(f'Privacy Acc Adv:\t\t{np.mean(privacy_acc_adv)}\nPrivacy Acc Coop:\t\t{np.mean(privacy_acc_coop)}\nUtility Acc Adv:\t\t{np.mean(utility_acc_adv)}\nUtility Acc Coop:\t\t{np.mean(utility_acc_coop)}')
            if run_eval: print(f'Val Privacy Acc Adv:\t\t{np.mean(val_privacy_acc_adv)}\nVal Privacy Acc Coop:\t\t{np.mean(val_privacy_acc_coop)}\nVal Utility Acc Adv:\t\t{np.mean(val_utility_acc_adv)}\nVal Utility Acc Coop:\t\t{np.mean(val_utility_acc_coop)}')
    
    if train_ae or train_cross: print(f'Discriminator Acc:\t\t{np.mean(discriminator_training_acc)}')
    if run_eval: print(f'Val Discriminator Acc:\t\t{np.mean(val_discriminator_acc)}')

    # Test Accuracy
    if run_sgn_eval and run_eval:
        print('\n')
        sgn_acc_known_acc, sgn_acc_known_f1, sgn_acc_known_prec, sgn_acc_known_recall, sgn_acc_known_topk = sgn_eval(eval_X_known, eval_Y_known_action, 'Known Action', is_action=True, k=k)
        sgn_acc_rec_acc, sgn_acc_rec_f1, sgn_acc_rec_prec, sgn_acc_rec_recall, sgn_acc_rec_topk = sgn_eval(eval_X_rec, eval_Y_rec_action, 'Reconstructed Action', is_action=True, k=k)
        sgn_acc_cross_acc, sgn_acc_cross_f1, sgn_acc_cross_prec, sgn_acc_cross_recall, sgn_acc_cross_topk = sgn_eval(eval_X, eval_Y_action, 'Generated Action', is_action=True, k=k)
        print('\n')
        sgn_priv_known_acc, sgn_priv_known_f1, sgn_priv_known_prec, sgn_priv_known_recall, sgn_priv_known_topk = sgn_eval(eval_X_known, eval_Y_known_actor, 'Known Actor', is_actor=True, k=k)
        sgn_priv_rec_acc, sgn_priv_rec_f1, sgn_priv_rec_prec, sgn_priv_rec_recall, sgn_priv_rec_topk = sgn_eval(eval_X_rec, eval_Y_rec_actor, 'Reconstructed Actor', is_actor=True, k=k)
        sgn_priv_cross_acc, sgn_priv_cross_f1, sgn_priv_cross_prec, sgn_priv_cross_recall, sgn_priv_cross_topk = sgn_eval(eval_X, eval_Y_actor, 'Generated Actor', is_actor=True, k=k)
        sgn_priv_initial_acc, sgn_priv_initial_f1, sgn_priv_initial_prec, sgn_priv_initial_recall, sgn_priv_initial_topk = sgn_eval(eval_X, eval_Y_initial_actor, 'Initial Actor', is_actor=True, k=k)
    else: print('\n')

    # Return dict with all losses and accuracies for plotting
    losses_dict = {}
    if train_ae or train_cross:
        losses_dict['loss'] = np.mean(losses)
        if run_eval: losses_dict['val_loss'] = np.mean(val_losses)
        losses_dict['rec_loss'] = np.mean(rec_loss)
        losses_dict['cross_loss'] = np.mean(cross_loss)
        losses_dict['end_effector_loss'] = np.mean(end_effector_loss)
        losses_dict['smoothing_loss'] = np.mean(smoothing_loss)
        losses_dict['triplet_loss'] = np.mean(triplet_loss)
        losses_dict['latent_consistency_loss'] = np.mean(latent_consistency_loss)
        losses_dict['privacy_loss'] = np.mean(privacy_loss)
        losses_dict['privacy_loss_adv'] = np.mean(privacy_loss_adv)
        losses_dict['privacy_loss_coop'] = np.mean(privacy_loss_coop)
        losses_dict['utility_loss'] = np.mean(utility_loss)
        losses_dict['utility_loss_adv'] = np.mean(utility_loss_adv)
        losses_dict['utility_loss_coop'] = np.mean(utility_loss_coop)
        losses_dict['discriminator_loss'] = np.mean(discriminator_loss)
    if run_eval:
        losses_dict['val_rec_loss'] = np.mean(val_rec_loss)
        losses_dict['val_cross_loss'] = np.mean(val_cross_loss)
        losses_dict['val_end_effector_loss'] = np.mean(val_end_effector_loss)
        losses_dict['val_smoothing_loss'] = np.mean(val_smoothing_loss)
        losses_dict['val_triplet_loss'] = np.mean(val_triplet_loss)
        losses_dict['val_latent_consistency_loss'] = np.mean(val_latent_consistency_loss)
        losses_dict['val_privacy_loss'] = np.mean(val_privacy_loss)
        losses_dict['val_privacy_loss_adv'] = np.mean(val_privacy_loss_adv)
        losses_dict['val_privacy_loss_coop'] = np.mean(val_privacy_loss_coop)
        losses_dict['val_utility_loss'] = np.mean(val_utility_loss)
        losses_dict['val_utility_loss_adv'] = np.mean(val_utility_loss_adv)
        losses_dict['val_utility_loss_coop'] = np.mean(val_utility_loss_coop)
        losses_dict['val_discriminator_loss'] = np.mean(val_discriminator_loss)
    if (train_emb_adv or train_discrim) and train_emb_this_epoch:
        losses_dict['priv_training_loss'] = np.mean(priv_training_loss)
        losses_dict['util_training_loss'] = np.mean(util_training_loss)
        losses_dict['discriminator_train_loss'] = np.mean(discriminator_train_losses)
        losses_dict['priv_training_acc'] = np.mean(priv_training_acc)
        losses_dict['util_training_acc'] = np.mean(util_training_acc)
        losses_dict['priv_coop_training_loss'] = np.mean(priv_coop_training_loss)
        losses_dict['priv_coop_training_acc'] = np.mean(priv_coop_training_acc)
        losses_dict['util_coop_training_loss'] = np.mean(util_coop_training_loss)
        losses_dict['util_coop_training_acc'] = np.mean(util_coop_training_acc)
        losses_dict['discriminator_training_acc'] = np.mean(discriminator_training_acc)
        if train_ae or train_cross:
            losses_dict['privacy_acc_adv'] = np.mean(privacy_acc_adv)
            losses_dict['privacy_acc_coop'] = np.mean(privacy_acc_coop)
            losses_dict['utility_acc_adv'] = np.mean(utility_acc_adv)
            losses_dict['utility_acc_coop'] = np.mean(utility_acc_coop)
        if run_eval:
            losses_dict['val_privacy_acc_adv'] = np.mean(val_privacy_acc_adv)
            losses_dict['val_privacy_acc_coop'] = np.mean(val_privacy_acc_coop)
            losses_dict['val_utility_acc_adv'] = np.mean(val_utility_acc_adv)
            losses_dict['val_utility_acc_coop'] = np.mean(val_utility_acc_coop)
    if train_ae or train_cross:
        losses_dict['discriminator_acc'] = np.mean(discriminator_training_acc)
    if run_eval:
        losses_dict['val_discriminator_acc'] = np.mean(val_discriminator_acc)
    if run_sgn_eval and run_eval:
        losses_dict['sgn_acc_known_acc'] = sgn_acc_known_acc
        losses_dict['sgn_acc_known_f1'] = sgn_acc_known_f1
        losses_dict['sgn_acc_known_prec'] = sgn_acc_known_prec
        losses_dict['sgn_acc_known_recall'] = sgn_acc_known_recall
        losses_dict['sgn_acc_rec_acc'] = sgn_acc_rec_acc
        losses_dict['sgn_acc_rec_f1'] = sgn_acc_rec_f1
        losses_dict['sgn_acc_rec_prec'] = sgn_acc_rec_prec
        losses_dict['sgn_acc_rec_recall'] = sgn_acc_rec_recall
        losses_dict['sgn_acc_cross_acc'] = sgn_acc_cross_acc
        losses_dict['sgn_acc_cross_f1'] = sgn_acc_cross_f1
        losses_dict['sgn_acc_cross_prec'] = sgn_acc_cross_prec
        losses_dict['sgn_acc_cross_recall'] = sgn_acc_cross_recall
        losses_dict['sgn_priv_known_acc'] = sgn_priv_known_acc
        losses_dict['sgn_priv_known_f1'] = sgn_priv_known_f1
        losses_dict['sgn_priv_known_prec'] = sgn_priv_known_prec
        losses_dict['sgn_priv_known_recall'] = sgn_priv_known_recall
        losses_dict['sgn_priv_rec_acc'] = sgn_priv_rec_acc
        losses_dict['sgn_priv_rec_f1'] = sgn_priv_rec_f1
        losses_dict['sgn_priv_rec_prec'] = sgn_priv_rec_prec
        losses_dict['sgn_priv_rec_recall'] = sgn_priv_rec_recall
        losses_dict['sgn_priv_cross_acc'] = sgn_priv_cross_acc
        losses_dict['sgn_priv_cross_f1'] = sgn_priv_cross_f1
        losses_dict['sgn_priv_cross_prec'] = sgn_priv_cross_prec
        losses_dict['sgn_priv_cross_recall'] = sgn_priv_cross_recall
        losses_dict['sgn_acc_known_topk'] = sgn_acc_known_topk
        losses_dict['sgn_acc_rec_topk'] = sgn_acc_rec_topk
        losses_dict['sgn_acc_cross_topk'] = sgn_acc_cross_topk
        losses_dict['sgn_priv_known_topk'] = sgn_priv_known_topk
        losses_dict['sgn_priv_rec_topk'] = sgn_priv_rec_topk
        losses_dict['sgn_priv_cross_topk'] = sgn_priv_cross_topk
        losses_dict['sgn_priv_initial_acc'] = sgn_priv_initial_acc
        losses_dict['sgn_priv_initial_f1'] = sgn_priv_initial_f1
        losses_dict['sgn_priv_initial_prec'] = sgn_priv_initial_prec
        losses_dict['sgn_priv_initial_recall'] = sgn_priv_initial_recall
        losses_dict['sgn_priv_initial_topk'] = sgn_priv_initial_topk
    
    # Save model
    if save and metric in losses_dict and losses_dict[metric] > 0:
        if matric_minimize:
            if np.mean(val_losses) < best_metric:
                best_metric = np.mean(val_losses)
                torch.save(model.state_dict(), 'pretrained/DMR_NTU120.pt')
        elif np.mean(val_losses) > best_metric:
            best_metric = np.mean(val_losses)
            torch.save(model.state_dict(), 'pretrained/DMR_NTU120.pt')

    return losses_dict


def sgn_eval(X, Y, label='Undefined', is_actor=False, is_action=False, k=3):
    assert is_actor != is_action, "is_actor and is_action cannot both be True"
    assert is_actor or is_action, "Either is_actor or is_action must be True"

    if is_actor:
        classes = privacy_classes
        sgn = sgn_priv
    elif is_action:
        classes = utility_classes
        sgn = sgn_ar

    X = np.concatenate(X)
    X = np.pad(X, ((0,0), (0,225), (0,75)), 'constant')

    Y = np.concatenate(Y) - 1
    Y = np.eye(classes)[Y.astype(int)]

    acc, f1, prec, recall, topk = run_sgn_eval(sgn_train_x, sgn_train_y, X, Y, sgn_val_x, sgn_val_y, 1, sgn, k=k)
    print(f'\n{label} Accuracy:\t\t{acc}\n{label} F1:\t\t\t{f1*100}\n{label} Precision:\t\t{prec*100}\n{label} Recall:\t\t{recall*100}\n{label} Top-{k} Accuracy:\t{topk}\n')
    return acc, f1, prec, recall, topk

# Simplified training loop for only AE
def train_unpaired(run_eval=True, run_sgn_eval=True, save=True, ae=True, ee=False, triplet=False, use_emb_adv=False, use_discrim_adv=False, emb_adv=False, discrim_adv=False, k=3, smoothing=True):
    global best_metric
    global total_epochs
    global cur_tot_epoch

    # Store eval values for validation
    eval_X_known, eval_Y_known_action, eval_Y_known_actor, eval_X_rec, eval_Y_rec_action, eval_Y_rec_actor = [], [], [], [], [], []
    
    # Losses for printing
    rec_loss, end_effector_loss, smoothing_loss, triplet_loss, privacy_loss, privacy_loss_adv, privacy_loss_coop, privacy_acc_adv, privacy_acc_coop, priv_training_loss, utility_loss, utility_loss_adv, utility_loss_coop, utility_acc_adv, utility_acc_coop, util_training_loss, discriminator_loss, discriminator_train_losses, discriminator_acc, discriminator_train_accs, priv_coop_training_loss, priv_training_acc, priv_coop_training_acc, util_coop_training_loss, util_training_acc, util_coop_training_acc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    val_rec_loss, val_end_effector_loss, val_smoothing_loss, val_triplet_loss, val_privacy_loss, val_privacy_loss_adv, val_privacy_loss_coop, val_privacy_acc_adv, val_privacy_acc_coop, val_utility_loss, val_utility_loss_adv, val_utility_loss_coop, val_utility_acc_adv, val_utility_acc_coop, val_discriminator_loss, val_discriminator_acc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    losses, val_losses = [], []

    # Determine if adversaries need to be trained
    train_emb_this_epoch = True
    if emb_clf_update_per_epoch_unpaired < 1 and ae:
        if cur_tot_epoch % round(1 / emb_clf_update_per_epoch_unpaired) != 0:
            train_emb_this_epoch = False

    for (x, actors, actions) in rec_train_dl:
        # Move tensors to the configured device
        x = x.float().to(device)

        # Split into position and rotation
        if only_use_pos:
            x_pos = x
            x_rot = x
        else:
            x_pos = x[:, :, :, :3]
            x_rot = x[:, :, :, 3:]

        # Train adversaries
        if emb_adv or discrim_adv:
            # Train the discriminator
            if train_emb_this_epoch:
                it = 1
                if emb_clf_update_per_epoch_unpaired > 1: it = emb_clf_update_per_epoch_unpaired
                for _ in range(int(it)):
                    priv_train_loss, priv_train_coop_loss, util_train_loss, util_train_coop_loss, discriminator_train_loss, priv_acc, util_acc, priv_coop_acc, util_coop_acc, discriminator_train_acc = model.train_adv_unpaired(x_pos, x_rot, actors, actions, train_emb=emb_adv, train_discrim=discrim_adv)
            
                # Track the loss
                priv_training_loss.append(priv_train_loss)
                priv_coop_training_loss.append(priv_train_coop_loss)
                priv_training_acc.append(priv_acc)
                priv_coop_training_acc.append(priv_coop_acc)
                util_training_loss.append(util_train_loss)
                util_coop_training_loss.append(util_train_coop_loss)
                util_training_acc.append(util_acc)
                util_coop_training_acc.append(util_coop_acc)
                discriminator_train_losses.append(discriminator_train_loss)
                discriminator_train_accs.append(discriminator_train_acc)
        
        if not ae: continue

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss, _, losses_ = model.loss_unpaired(x_pos, x_rot, actors, actions, reconstruction=ae, emb_adv=use_emb_adv, discrim_adv=use_discrim_adv, ee=ee, triplet=triplet)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Track the loss
        losses.append(loss.item())

        rec_loss.append(losses_['rec_loss'])
        end_effector_loss.append(losses_['end_effector_loss'])
        smoothing_loss.append(losses_['smoothing_loss'])
        triplet_loss.append(losses_['triplet_loss'])
        privacy_loss.append(losses_['privacy_loss'])
        privacy_loss_adv.append(losses_['privacy_loss_adv'])
        privacy_loss_coop.append(losses_['privacy_loss_coop'])
        privacy_acc_adv.append(losses_['privacy_acc_adv'])
        privacy_acc_coop.append(losses_['privacy_acc_coop'])
        utility_loss.append(losses_['utility_loss'])
        utility_loss_adv.append(losses_['utility_loss_adv'])
        utility_loss_coop.append(losses_['utility_loss_coop'])
        utility_acc_adv.append(losses_['utility_acc_adv'])
        utility_acc_coop.append(losses_['utility_acc_coop'])
        discriminator_loss.append(losses_['discriminator_loss'])
        discriminator_acc.append(losses_['discriminator_acc'])


    # Decay learning rate
    # scheduler.step()

    # Validation
    if run_eval:
        with torch.no_grad():
            for (x, actors, actions) in rec_val_dl:
                x = x.float().to(device)

                # Split into position and rotation
                if only_use_pos:
                    x_pos = x
                    x_rot = x
                else:
                    x_pos = x[:, :, :, :3]
                    x_rot = x[:, :, :, 3:]
                
                loss, _, losses_ = model.loss_unpaired(x_pos, x_rot, actors, actions, reconstruction=ae, emb_adv=use_emb_adv, discrim_adv=use_discrim_adv, ee=ee, triplet=triplet)
                val_losses.append(loss.item())

                if run_sgn_eval:
                    eval_X_known.append(x_pos.contiguous().view(x_pos.size(0), T, -1).cpu().numpy())
                    eval_Y_known_action.append(np.array(actions))
                    eval_Y_known_actor.append(np.array(actors))

                    eval_X_rec.append(model(x_pos, x_rot).cpu().numpy())
                    eval_Y_rec_action.append(np.array(actions))
                    eval_Y_rec_actor.append(np.array(actors))
                
                val_rec_loss.append(losses_['rec_loss'])
                val_end_effector_loss.append(losses_['end_effector_loss'])
                val_smoothing_loss.append(losses_['smoothing_loss'])
                val_triplet_loss.append(losses_['triplet_loss'])
                val_privacy_loss.append(losses_['privacy_loss'])
                val_privacy_loss_adv.append(losses_['privacy_loss_adv'])
                val_privacy_loss_coop.append(losses_['privacy_loss_coop'])
                val_privacy_acc_adv.append(losses_['privacy_acc_adv'])
                val_privacy_acc_coop.append(losses_['privacy_acc_coop'])
                val_utility_loss.append(losses_['utility_loss'])
                val_utility_loss_adv.append(losses_['utility_loss_adv'])
                val_utility_loss_coop.append(losses_['utility_loss_coop'])
                val_utility_acc_adv.append(losses_['utility_acc_adv'])
                val_utility_acc_coop.append(losses_['utility_acc_coop'])
                val_discriminator_loss.append(losses_['discriminator_loss'])
                val_discriminator_acc.append(losses_['discriminator_acc'])

    # Print loss/accuracy
    print(f'--------------------\nEpoch {cur_tot_epoch+1}/{total_epochs}\n--------------------')
    cur_tot_epoch += 1
    if ae:
        print(f'Training Loss:\t\t\t{np.mean(losses)}\nValidation Loss:\t\t{np.mean(val_losses)}\n')
        print('Training Losses:')
        print(f'Reconstruction Loss:\t\t{np.mean(rec_loss)}\nEnd Effector Loss:\t\t{np.mean(end_effector_loss)}\nSmoothing Loss:\t\t\t{np.mean(smoothing_loss)}\nTriplet Loss:\t\t\t{np.mean(triplet_loss)}')
        if use_emb_adv:
            print(f'Privacy Loss:\t\t\t{np.mean(privacy_loss)}\nPrivacy Loss Dyn:\t\t{np.mean(privacy_loss_adv)}\nPrivacy Loss Stat:\t\t{np.mean(privacy_loss_coop)}')
            print(f'Utility Loss:\t\t\t{np.mean(utility_loss)}\nUtility Loss Dyn:\t\t{np.mean(utility_loss_adv)}\nUtility Loss Stat:\t\t{np.mean(utility_loss_coop)}')
        if use_discrim_adv: print(f'Discriminator Loss:\t\t{np.mean(discriminator_loss)}')
    if run_eval:
        print('\nValidation Losses:')
        print(f'Val Reconstruction Loss:\t{np.mean(val_rec_loss)}\nVal End Effector Loss:\t\t{np.mean(val_end_effector_loss)}\nVal Smoothing Loss:\t\t{np.mean(val_smoothing_loss)}\nVal Triplet Loss:\t\t{np.mean(val_triplet_loss)}')
        if use_emb_adv:
            print(f'Val Privacy Loss:\t\t{np.mean(val_privacy_loss)}\nVal Privacy Loss Dyn:\t\t{np.mean(val_privacy_loss_adv)}\nVal Privacy Loss Stat:\t\t{np.mean(val_privacy_loss_coop)}')
            print(f'Val Utility Loss:\t\t{np.mean(val_utility_loss)}\nVal Utility Loss Dyn:\t\t{np.mean(val_utility_loss_adv)}\nVal Utility Loss Stat:\t\t{np.mean(val_utility_loss_coop)}')
        if use_discrim_adv: print(f'Val Discriminator Loss:\t\t{np.mean(val_discriminator_loss)}')
    if (emb_adv or discrim_adv) and train_emb_this_epoch:
        print('\nAdversary Losses')
        print(f'Privacy Training Loss:\t\t{np.mean(priv_training_loss)}\nUtility Training Loss:\t\t{np.mean(util_training_loss)}\nDiscriminator Training Loss:\t{np.mean(discriminator_train_losses)}')
        print(f'Privacy Training Acc:\t\t{np.mean(priv_training_acc)}\nUtility Training Acc:\t\t{np.mean(util_training_acc)}\nDiscriminator Training Acc:\t{np.mean(discriminator_train_accs)}')
        print(f'Privacy Training Coop Loss:\t{np.mean(priv_coop_training_loss)}\nUtility Training Coop Loss:\t{np.mean(util_coop_training_loss)}')
        print(f'Privacy Training Coop Acc:\t{np.mean(priv_coop_training_acc)}\nUtility Training Coop Acc:\t{np.mean(util_coop_training_acc)}')
        if emb_adv and ae:
            print(f'Privacy Acc Adv:\t\t{np.mean(privacy_acc_adv)}\nPrivacy Acc Coop:\t\t{np.mean(privacy_acc_coop)}\nUtility Acc Adv:\t\t{np.mean(utility_acc_adv)}\nUtility Acc Coop:\t\t{np.mean(utility_acc_coop)}')
            if run_eval: print(f'Val Privacy Acc Adv:\t\t{np.mean(val_privacy_acc_adv)}\nVal Privacy Acc Coop:\t\t{np.mean(val_privacy_acc_coop)}\nVal Utility Acc Adv:\t\t{np.mean(val_utility_acc_adv)}\nVal Utility Acc Coop:\t\t{np.mean(val_utility_acc_coop)}')
        if discrim_adv and ae:
            print(f'Discriminator Acc:\t\t{np.mean(discriminator_acc)}')
            if run_eval: print(f'Val Discriminator Acc:\t\t{np.mean(val_discriminator_acc)}')


    # Test Accuracy
    if run_sgn_eval and run_eval:
        print('\n')
        sgn_acc_known_acc, sgn_acc_known_f1, sgn_acc_known_prec, sgn_acc_known_recall, sgn_acc_known_topk = sgn_eval(eval_X_known, eval_Y_known_action, 'Known Action', is_action=True, k=k)
        sgn_acc_rec_acc, sgn_acc_rec_f1, sgn_acc_rec_prec, sgn_acc_rec_recall, sgn_acc_rec_topk = sgn_eval(eval_X_rec, eval_Y_rec_action, 'Reconstructed Action', is_action=True, k=k)
        print('\n')
        sgn_priv_known_acc, sgn_priv_known_f1, sgn_priv_known_prec, sgn_priv_known_recall, sgn_priv_known_topk = sgn_eval(eval_X_known, eval_Y_known_actor, 'Known Actor', is_actor=True, k=k)
        sgn_priv_rec_acc, sgn_priv_rec_f1, sgn_priv_rec_prec, sgn_priv_rec_recall, sgn_priv_rec_topk = sgn_eval(eval_X_rec, eval_Y_rec_actor, 'Reconstructed Actor', is_actor=True, k=k)
        print('\n')
    else: print('\n')

    losses_dict = {}
    losses_dict['loss'] = np.mean(losses)

    if ae: losses_dict['rec_loss'] = np.mean(rec_loss)
    if ee: losses_dict['end_effector_loss'] = np.mean(end_effector_loss)
    if smoothing: losses_dict['smoothing_loss'] = np.mean(smoothing_loss)
    if triplet: losses_dict['triplet_loss'] = np.mean(triplet_loss)
    if run_eval:
        losses_dict['val_loss'] = np.mean(val_losses)
        if ae: losses_dict['val_rec_loss'] = np.mean(val_rec_loss)
        if ee: losses_dict['val_end_effector_loss'] = np.mean(val_end_effector_loss)
        if smoothing: losses_dict['val_smoothing_loss'] = np.mean(val_smoothing_loss)
        if triplet: losses_dict['val_triplet_loss'] = np.mean(val_triplet_loss)
        if run_sgn_eval:
            losses_dict['sgn_acc_known_acc'] = sgn_acc_known_acc
            losses_dict['sgn_acc_known_f1'] = sgn_acc_known_f1
            losses_dict['sgn_acc_known_prec'] = sgn_acc_known_prec
            losses_dict['sgn_acc_known_recall'] = sgn_acc_known_recall
            losses_dict['sgn_acc_rec_acc'] = sgn_acc_rec_acc
            losses_dict['sgn_acc_rec_f1'] = sgn_acc_rec_f1
            losses_dict['sgn_acc_rec_prec'] = sgn_acc_rec_prec
            losses_dict['sgn_acc_rec_recall'] = sgn_acc_rec_recall
            losses_dict['sgn_acc_known_topk'] = sgn_acc_known_topk
            losses_dict['sgn_acc_rec_topk'] = sgn_acc_rec_topk
            losses_dict['sgn_priv_known_acc'] = sgn_priv_known_acc
            losses_dict['sgn_priv_known_f1'] = sgn_priv_known_f1
            losses_dict['sgn_priv_known_prec'] = sgn_priv_known_prec
            losses_dict['sgn_priv_known_recall'] = sgn_priv_known_recall
            losses_dict['sgn_priv_rec_acc'] = sgn_priv_rec_acc
            losses_dict['sgn_priv_rec_f1'] = sgn_priv_rec_f1
            losses_dict['sgn_priv_rec_prec'] = sgn_priv_rec_prec
            losses_dict['sgn_priv_rec_recall'] = sgn_priv_rec_recall
            losses_dict['sgn_priv_known_topk'] = sgn_priv_known_topk
            losses_dict['sgn_priv_rec_topk'] = sgn_priv_rec_topk
    if use_emb_adv:
        losses_dict['privacy_loss'] = np.mean(privacy_loss)
        losses_dict['privacy_loss_adv'] = np.mean(privacy_loss_adv)
        losses_dict['privacy_loss_coop'] = np.mean(privacy_loss_coop)
        losses_dict['utility_loss'] = np.mean(utility_loss)
        losses_dict['utility_loss_adv'] = np.mean(utility_loss_adv)
        losses_dict['utility_loss_coop'] = np.mean(utility_loss_coop)
        losses_dict['privacy_acc_adv'] = np.mean(privacy_acc_adv)
        losses_dict['privacy_acc_coop'] = np.mean(privacy_acc_coop)
        losses_dict['utility_acc_adv'] = np.mean(utility_acc_adv)
        losses_dict['utility_acc_coop'] = np.mean(utility_acc_coop)
        if run_eval:    
            losses_dict['val_privacy_acc_adv'] = np.mean(val_privacy_acc_adv)
            losses_dict['val_privacy_acc_coop'] = np.mean(val_privacy_acc_coop)
            losses_dict['val_utility_acc_adv'] = np.mean(val_utility_acc_adv)
            losses_dict['val_utility_acc_coop'] = np.mean(val_utility_acc_coop)
            losses_dict['val_privacy_loss'] = np.mean(val_privacy_loss)
            losses_dict['val_privacy_loss_adv'] = np.mean(val_privacy_loss_adv)
            losses_dict['val_privacy_loss_coop'] = np.mean(val_privacy_loss_coop)
            losses_dict['val_utility_loss'] = np.mean(val_utility_loss)
            losses_dict['val_utility_loss_adv'] = np.mean(val_utility_loss_adv)
            losses_dict['val_utility_loss_coop'] = np.mean(val_utility_loss_coop)
    if emb_adv and train_emb_this_epoch:
        losses_dict['priv_training_loss'] = np.mean(priv_training_loss)
        losses_dict['priv_training_acc'] = np.mean(priv_training_acc)
        losses_dict['priv_coop_training_loss'] = np.mean(priv_coop_training_loss)
        losses_dict['priv_coop_training_acc'] = np.mean(priv_coop_training_acc)
        losses_dict['util_training_loss'] = np.mean(util_training_loss)
        losses_dict['util_training_acc'] = np.mean(util_training_acc)
        losses_dict['util_coop_training_loss'] = np.mean(util_coop_training_loss)
        losses_dict['util_coop_training_acc'] = np.mean(util_coop_training_acc)
    if use_discrim_adv:
        losses_dict['discriminator_loss'] = np.mean(discriminator_loss)
        losses_dict['discriminator_acc'] = np.mean(discriminator_acc)
        if run_eval:
            losses_dict['val_discriminator_loss'] = np.mean(val_discriminator_loss)
            losses_dict['val_discriminator_acc'] = np.mean(val_discriminator_acc)
    if discrim_adv and train_emb_this_epoch:
        losses_dict['discriminator_train_loss'] = np.mean(discriminator_train_losses)
        losses_dict['discriminator_train_acc'] = np.mean(discriminator_train_accs)
        
    # Save model
    if save and metric in losses_dict and losses_dict[metric] > 0:
        if matric_minimize:
            if np.mean(val_losses) < best_metric:
                best_metric = np.mean(val_losses)
                torch.save(model.state_dict(), 'pretrained/DMR_NTU120.pt')
        elif np.mean(val_losses) > best_metric:
            best_metric = np.mean(val_losses)
            torch.save(model.state_dict(), 'pretrained/DMR_NTU120.pt')

    return losses_dict

# %%
training_stages = [
    # # Pre-Train Cross to separate embeddings
    # {'epochs': 5, 'paired': True, 'ae': True, 'ee': True, 'cross': True, 'triplet': True, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'sgn_eval': False, 'save': False},
    
    # # Pre-Train AE
    # {'epochs': 20, 'paired': False, 'ae': True, 'ee': True, 'cross': False, 'triplet': True, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'sgn_eval': False, 'save': False},
    
    # # Pre-Train Adversaries (Paired)
    # {'epochs': 20, 'paired': True, 'ae': False, 'ee': False, 'cross': False, 'triplet': False, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'sgn_eval': False, 'save': False},
    
    # # Pre-Train Adversaries
    # {'epochs': 50, 'paired': False, 'ae': False, 'ee': False, 'cross': False, 'triplet': False, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'sgn_eval': False, 'save': False},
    
    # # Train AE and adversaries with adversary loss
    {'epochs': 20, 'paired': False, 'ae': True, 'ee': True, 'cross': False, 'triplet': True, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': True, 'sgn_eval': True, 'save': True},
    # {'epochs': 10, 'paired': False, 'ae': True, 'ee': True, 'cross': False, 'triplet': True, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': True, 'discrim_adv': True, 'eval': True, 'sgn_eval': True, 'save': True},
    # Paired Training (Crossing)
    {'epochs': 40, 'paired': True, 'ae': True, 'ee': True, 'cross': True, 'triplet': True, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': True, 'sgn_eval': False, 'save': True},
]
sgn_stage = {'epochs': 1, 'paired': True, 'ae': False, 'ee': False, 'cross': False, 'triplet': False, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': True, 'sgn_eval': True, 'save': False}
total_epochs = sum([stage['epochs'] for stage in training_stages])
cur_tot_epoch = 0
if sgn_eval_after_each_stage: total_epochs += len(training_stages)

# Load AE pretrained model
# uncomment training stages to use full training
# pre_trained = 'pretrained/20240510-110342/stage_4.pt'
# pre_trained = 'pretrained/20240405-130711/stage_5.pt'
# model.load_state_dict(torch.load(pre_trained))

# mlflow logging
try: mlflow.end_run()
except: pass
mlflow.start_run()
mlflow.log_param('total_epochs', total_epochs)
mlflow.log_param('batch_size', batch_size)
mlflow.log_param('learning_rate', lr)
mlflow.log_param('one_dimension_conv', one_dimension_conv)
mlflow.log_param('ntu120', ntu_120)
mlflow.log_param('train_equal_test', str(not seperate_train_test))
mlflow.log_param('only_use_pos', str(only_use_pos))
mlflow.log_param('encoded_channels', str(encoded_channels))
mlflow.log_param('cross_samples_train', cross_samples_train)
mlflow.log_param('cross_samples_test', cross_samples_test)
mlflow.log_param('T', T)
mlflow.log_params(model.get_loss_params())

# os.mkdir('training_stages_log')
training_stage_name = f'training_stages_log/stages{datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
with open(training_stage_name, 'w') as f:
    json.dump(training_stages, f)
mlflow.log_artifact(training_stage_name)

stages_save_path = f'pretrained/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
os.mkdir(stages_save_path)

for i, stage in enumerate(training_stages):
    print('\nMoving to new stage')
    print(stage, '\n')
    if stage['save']: assert stage['eval'], 'Cannot save model without evaluating'
    for epoch in range(stage['epochs']):
        if stage['sgn_eval']:
            if validation_acc_freq > 0 and epoch % validation_acc_freq == 0: use_sgn = True
            else: use_sgn = False
        else: use_sgn = False
        if not stage['paired']:
            log_dict = train_unpaired(run_eval=stage['eval'], run_sgn_eval= use_sgn, save=stage['save'], ae=stage['ae'], ee=stage['ee'], triplet=stage['triplet'], use_emb_adv=stage['emb_adv'], use_discrim_adv=stage['discrim_adv'], emb_adv=stage['train_emb_adv'], discrim_adv=stage['train_discrim_adv'], k=k)
        else: 
            log_dict = train_paired(train_ae=stage['ae'], train_cross=stage['cross'], train_discrim=stage['train_discrim_adv'], train_emb_adv=stage['train_emb_adv'], run_eval=stage['eval'], use_emb_adv=stage['emb_adv'], use_discrim_adv=stage['discrim_adv'], run_sgn_eval= use_sgn, save=stage['save'], k=k)
        
        for key, value in log_dict.items():
            mlflow.log_metric(key, value, step=cur_tot_epoch-1)

    # save model
    torch.save(model.state_dict(), f'{stages_save_path}/stage_{i}.pt')

    if sgn_eval_after_each_stage:
        print('\nEvaluating Stage\n')
        stage = sgn_stage
        log_dict = train_paired(train_ae=stage['ae'], train_cross=stage['cross'], train_discrim=stage['train_discrim_adv'], train_emb_adv=stage['train_emb_adv'], run_eval=stage['eval'], use_emb_adv=stage['emb_adv'], use_discrim_adv=stage['discrim_adv'], run_sgn_eval= use_sgn, save=stage['save'], k=k)
        cur_tot_epoch += 1
        for key, value in log_dict.items():
            mlflow.log_metric(key, value, step=cur_tot_epoch-1)

mlflow.pytorch.log_state_dict(model.state_dict(), 'final_model')
mlflow.end_run()

torch.save(model.state_dict(), 'pretrained/DMR_NTU120.pt')

# %%
torch.save(model.state_dict(), 'NTU60_1.3_priv.5_util10_5.18.24.pt')

# %%
# # run sgn evals for the stages
# run = 'pretrained/20240226-132853/stage_'
# model.set_eval()
# for i in range(6):
#     model.load_state_dict(torch.load(run + str(i) + '.pt'))
#     stage = sgn_stage
#     log_dict = train_paired(train_ae=stage['ae'], train_cross=stage['cross'], train_discrim=stage['train_discrim_adv'], train_emb_adv=stage['train_emb_adv'], run_eval=stage['eval'], use_emb_adv=stage['emb_adv'], use_discrim_adv=stage['discrim_adv'], run_sgn_eval= use_sgn, save=stage['save'], k=k)
#     print(log_dict)
# model.set_eval(False)

# %% [markdown]
# ## Visualize Results

# %% [markdown]
# ### Raw Data

# %%
# render_video(val_data[0][0][2][:, :, :3], gif='test', show_render=False)

# %% [markdown]
# ### Reconstruction Data

# %%
# sk = val_data[0][0][2][:, :, :3].unsqueeze(0).to(device)
# render_video(model(sk, sk).cpu().detach().numpy()[0])

# %% [markdown]
# ### Retarget Data

# %%
# x1 = val_data[0][0][2][:, :, :3].unsqueeze(0).to(device)
# x2 = val_data[0][1][2][:, :, :3].unsqueeze(0).to(device)
# y1 = val_data[0][2][2][:, :, :3].unsqueeze(0).to(device)
# y2 = val_data[0][3][2][:, :, :3].unsqueeze(0).to(device)
# print(f'Actors: {val_data[0][0][0]}, {val_data[0][2][0]}')
# print(f'Actions: {val_data[0][0][1]}, {val_data[0][1][1]}')

# %%
# render_video(model(x1, x2).cpu().detach().numpy()[0])
# render_video(model(x2, x1).cpu().detach().numpy()[0])
# render_video(model(y1, y2).cpu().detach().numpy()[0])
# render_video(model(y2, y1).cpu().detach().numpy()[0])

# %% [markdown]
# # Retargeting

# %%
dmr = DMR().to(device)
dmr.load_state_dict(torch.load('pretrained/DMR_NTU120.pt'))

# %%
val_model = AutoEncoder(use_adv=False).to(device)
weights = torch.load('pretrained/DMR_NTU120.pt')
keys_to_rem = ["priv_adv.conv1.weight", "priv_adv.conv1.bias", "priv_adv.conv2.weight", "priv_adv.conv2.bias", "priv_adv.conv3.weight", "priv_adv.conv3.bias", "priv_adv.bn1.weight", "priv_adv.bn1.bias", "priv_adv.bn1.running_mean", "priv_adv.bn1.running_var", "priv_adv.bn1.num_batches_tracked", "priv_adv.bn2.weight", "priv_adv.bn2.bias", "priv_adv.bn2.running_mean", "priv_adv.bn2.running_var", "priv_adv.bn2.num_batches_tracked", "priv_adv.bn3.weight", "priv_adv.bn3.bias", "priv_adv.bn3.running_mean", "priv_adv.bn3.running_var", "priv_adv.bn3.num_batches_tracked", "priv_adv.fc1.weight", "priv_adv.fc1.bias", "priv_adv.fc2.weight", "priv_adv.fc2.bias", "priv_adv.fc3.weight", "priv_adv.fc3.bias", "priv_coop.conv1.weight", "priv_coop.conv1.bias", "priv_coop.conv2.weight", "priv_coop.conv2.bias", "priv_coop.conv3.weight", "priv_coop.conv3.bias", "priv_coop.bn1.weight", "priv_coop.bn1.bias", "priv_coop.bn1.running_mean", "priv_coop.bn1.running_var", "priv_coop.bn1.num_batches_tracked", "priv_coop.bn2.weight", "priv_coop.bn2.bias", "priv_coop.bn2.running_mean", "priv_coop.bn2.running_var", "priv_coop.bn2.num_batches_tracked", "priv_coop.bn3.weight", "priv_coop.bn3.bias", "priv_coop.bn3.running_mean", "priv_coop.bn3.running_var", "priv_coop.bn3.num_batches_tracked", "priv_coop.fc1.weight", "priv_coop.fc1.bias", "priv_coop.fc2.weight", "priv_coop.fc2.bias", "priv_coop.fc3.weight", "priv_coop.fc3.bias", "util_adv.conv1.weight", "util_adv.conv1.bias", "util_adv.conv2.weight", "util_adv.conv2.bias", "util_adv.conv3.weight", "util_adv.conv3.bias", "util_adv.bn1.weight", "util_adv.bn1.bias", "util_adv.bn1.running_mean", "util_adv.bn1.running_var", "util_adv.bn1.num_batches_tracked", "util_adv.bn2.weight", "util_adv.bn2.bias", "util_adv.bn2.running_mean", "util_adv.bn2.running_var", "util_adv.bn2.num_batches_tracked", "util_adv.bn3.weight", "util_adv.bn3.bias", "util_adv.bn3.running_mean", "util_adv.bn3.running_var", "util_adv.bn3.num_batches_tracked", "util_adv.fc1.weight", "util_adv.fc1.bias", "util_adv.fc2.weight", "util_adv.fc2.bias", "util_adv.fc3.weight", "util_adv.fc3.bias", "util_coop.conv1.weight", "util_coop.conv1.bias", "util_coop.conv2.weight", "util_coop.conv2.bias", "util_coop.conv3.weight", "util_coop.conv3.bias", "util_coop.bn1.weight", "util_coop.bn1.bias", "util_coop.bn1.running_mean", "util_coop.bn1.running_var", "util_coop.bn1.num_batches_tracked", "util_coop.bn2.weight", "util_coop.bn2.bias", "util_coop.bn2.running_mean", "util_coop.bn2.running_var", "util_coop.bn2.num_batches_tracked", "util_coop.bn3.weight", "util_coop.bn3.bias", "util_coop.bn3.running_mean", "util_coop.bn3.running_var", "util_coop.bn3.num_batches_tracked", "util_coop.fc1.weight", "util_coop.fc1.bias", "util_coop.fc2.weight", "util_coop.fc2.bias", "util_coop.fc3.weight", "util_coop.fc3.bias", "discriminator.enc1.weight", "discriminator.enc1.bias", "discriminator.enc2.weight", "discriminator.enc2.bias", "discriminator.enc3.weight", "discriminator.enc3.bias", "discriminator.enc4.weight", "discriminator.enc4.bias", "discriminator.fc1.weight", "discriminator.fc1.bias", "discriminator.fc2.weight", "discriminator.fc2.bias"]
for key in keys_to_rem:
    del weights[key]
val_model.load_state_dict(weights)
model = val_model

# %%
val_model = model

# %%
def retarget_random_action():
    X_hat_random = {}
    X_hat_constant = {}

    # const = random.sample(list(X.keys()), 1)[0]
    const = 'S007C001P025R001A045'.encode('utf-8')
    x2_const = X[const].float().cuda().unsqueeze(0)
    print(const)
    times = []
    with torch.no_grad():
        for file in tqdm(X):
            x1 = X[file].unsqueeze(0)
            while True:
                sample = random.sample(list(X.keys()), 1)[0]
                # ensure different actor
                if sample.decode('utf-8')[9:12] != file.decode('utf-8')[9:12]:
                    break
            x2_random = X[sample].unsqueeze(0)
            start = time.time()
            X_hat_random[file] = val_model.eval(x1.float().cuda(), x2_random.float().cuda()).cpu().numpy().squeeze()
            times.append(time.time() - start)
            start = time.time()
            X_hat_constant[file] = val_model.eval(x1.float().cuda(), x2_const).cpu().numpy().squeeze()
            times.append(time.time() - start)
            # render_video(X_hat_random[file])
            # render_video(X_hat_constant[file])

    print(f'Average time: {np.mean(times)}')
    
    # Save results
    with open('results/NTU120_DMR_X_hat_random_RA.pkl', 'wb') as f:
        pickle.dump(X_hat_random, f)
    with open(f'results/NTU120_DMR_X_hat_constant_RA.pkl', 'wb') as f:
        pickle.dump(X_hat_constant, f)

def retarget_constant_action():
    X_hat_random = {}
    X_hat_constant = {}

    const = 8
    # x2_const = X[const].float().cuda().unsqueeze(0)
    times = []

    const_dict = {}
    # Get a sample of each action from the constant actor
    for file in X:
        info = parse_file_name(file)
        if info['P'] == const and info['A'] not in const_dict:
            const_dict[info['A']] = X[file].float().cuda().unsqueeze(0)

    with torch.no_grad():
        for file in tqdm(X):
            x1 = X[file].unsqueeze(0)
            info = parse_file_name(file)
            while True:
                sample = random.sample(list(X.keys()), 1)[0]
                # ensure different actor and same action
                info_ = parse_file_name(sample)
                if info_['P'] != info['P'] and info_['A'] == info['A']:
                    break
            x2_random = X[sample].unsqueeze(0)
            start = time.time()
            X_hat_random[file] = val_model.eval(x1.float().cuda(), x2_random.float().cuda()).cpu().numpy().squeeze()
            times.append(time.time() - start)
            
            start = time.time()
            X_hat_constant[file] = val_model.eval(x1.float().cuda(), const_dict[info['A']]).cpu().numpy().squeeze()
            times.append(time.time() - start)
            
            # render_video(X_hat_random[file])
            # render_video(X_hat_constant[file])

    print(f'Average time: {np.mean(times)}')
    
    # Save results
    with open('results/NTU_120_DMR_X_hat_random_CA.pkl', 'wb') as f:
        pickle.dump(X_hat_random, f)
    with open(f'results/NTU_120_DMR_X_hat_constant_CA.pkl', 'wb') as f:
        pickle.dump(X_hat_constant, f)

def retarget_specific(dummy, reference = None, just_render = False, use_dmr = False):
    X_hat = {}
    X_hat_dmr = {}

    x2_const = X[dummy].float().cuda().unsqueeze(0)
    with torch.no_grad():
        if reference is None:
            for file in X:
                x1 = X[file].unsqueeze(0)
                X_hat[file] = val_model.eval(x1.float().cuda(), x2_const).cpu().numpy().squeeze()
                if use_dmr: X_hat_dmr[file] = dmr.eval(x1.float().cuda(), x2_const).cpu().numpy().squeeze()
                if just_render: 
                    render_video(X_hat[file])
                    if use_dmr: render_video(X_hat_dmr[file])
                break
        else:
            X_hat = model.eval(X[reference].unsqueeze(0).float().cuda(), X[dummy].unsqueeze(0).float().cuda()).cpu().numpy().squeeze()
            if use_dmr: X_hat_dmr = dmr.eval(X[reference].unsqueeze(0).float().cuda(), X[dummy].unsqueeze(0).float().cuda()).cpu().numpy().squeeze()
            if just_render: 
                render_video(X_hat)
                if use_dmr: render_video(X_hat_dmr)
            else:
                render_video(X_hat, gif=f'pmr_{dummy}')
                if use_dmr: render_video(X_hat_dmr, gif=f'dmr_{dummy}')

    # Save results
    # with open(f'results/X_hat_{dummy}.pkl', 'wb') as f:
    #     pickle.dump(X_hat, f)

    # if use_dmr:
    #     with open(f'results/X_hat_dmr_{dummy}.pkl', 'wb') as f:
    #         pickle.dump(X_hat_dmr, f)

    return X_hat

# retarget_random_action()
# retarget_constant_action()

# %% [markdown]
# ## Testing Retargeting

# %%
# Finds samples and genders of specific actions
genders = pd.read_csv('NTU\SGN\statistics\Genders.csv')
use_dmr = True

action_to_use = 1
for i in range(500):
    dummy = random.sample(list(X.keys()), 1)[0]
    actor = int(dummy[9:12])
    action = int(dummy[17:20])
    if action != action_to_use: continue
    gender = genders.loc[actor-1, 'Gender']
    print(dummy, gender)

# %%
def make_visuals(action1, action2):
    files = []
    split_genders = True
    first_gender = None
    attempts = 0
    while True:
        attempts += 1
        dummy = random.sample(list(X.keys()), 1)[0]
        actor = int(dummy[9:12])
        action = int(dummy[17:20])
        if actor < 40: continue
        gender = genders.loc[actor-1, 'Gender']
        if action == action1: 
            if split_genders:
                if gender == first_gender: continue
                first_gender = gender
            files.append(dummy)
            action1 = -1
        if action == action2:
            if split_genders:
                if gender == first_gender: continue
                first_gender = gender
            files.append(dummy)
            action2 = -1
        if action1 == -1 and action2 == -1: break
    print(files)
    retarget_specific(files[0], reference=files[1], just_render=False, use_dmr=use_dmr)
    retarget_specific(files[1], reference=files[0], just_render=False, use_dmr=use_dmr)
    render_video(X[files[0]], gif=files[0].decode('utf-8'))
    render_video(X[files[1]], gif=files[1].decode('utf-8'))

for i in range(60, 120):
    print(f'Action {i}')
    make_visuals(i, i)

# %%
# Reading
# sitting S: 007, 015, 011
# standing S: 016, 003, 006, 008
ref = 'S006C003P017R001A011'.encode('utf-8') # female
dummy = 'S015C003P019R001A011'.encode('utf-8') # male
retarget_specific(dummy, reference=ref, just_render=False, use_dmr=use_dmr)
render_video(X[ref])#, gif=ref)
render_video(X[dummy])#, gif=dummy)

# %%
# Drink water
ref = 'S001C001P001R001A001'.encode('utf-8') # female
dummy = 'S010C002P021R001A001'.encode('utf-8') # male 
retarget_specific(dummy, reference=ref, just_render=False, use_dmr=use_dmr)
render_video(X[ref], gif=ref)
render_video(X[dummy], gif=dummy)

# %%
# Cross hands
ref = 'S003C001P002R001A040'.encode('utf-8') # female
dummy = 'S009C003P008R002A040'.encode('utf-8') # male 
retarget_specific(dummy, reference=ref, just_render=True, use_dmr=use_dmr)
render_video(X[ref])#, gif=ref)
render_video(X[dummy])#, gif=dummy)

# %%
# Cross hands
ref = 'S003C001P002R001A040'.encode('utf-8') # female
dummy = 'S009C003P008R002A040'.encode('utf-8') # male 
retarget_specific(dummy, reference=ref, just_render=True, use_dmr=use_dmr)
render_video(X[ref])#, gif=ref)
render_video(X[dummy])#, gif=dummy)

# %% [markdown]
# # Evaluate utility of other baselines

# %%
# Load SGN gender classification
sgn_gender = SGN(2, None, seg, batch_size, 0).to(device)
sgn_gender.load_state_dict(torch.load('SGN/pretrained/gender.pt')['state_dict'])

# %%
sgn_train_x, sgn_train_y, sgn_val_x, sgn_val_y = np.zeros((batch_size, 300, 150)), np.zeros((batch_size, 1)), np.zeros((batch_size, 300, 150)), np.zeros((batch_size, 1))

eval_renders_str_skeleton = ['S006C003P017R001A011.skeleton',
                             'S015C003P019R001A011.skeleton']
eval_renders_str = [x[:-9] for x in eval_renders_str_skeleton]
eval_render_byte = [x.encode('utf-8') for x in eval_renders_str]
genders = pd.read_csv('NTU\SGN\statistics\Genders.csv').replace('M', 1).replace('F', 0)

def anonymizer_to_sgn(t, max_frames=300):
    xyz, frames, joints, actors = t.shape
    
    # Pre-allocate memory for the output array
    X = np.zeros((max_frames, xyz * joints * actors), dtype=np.float32)
    
    # Reshape the input array for easier manipulation
    t_reshaped = t.reshape((frames, -1))
    
    # Copy over the reshaped data to the pre-allocated output
    X[:frames, :t_reshaped.shape[1]] = t_reshaped
    
    return X

def eval(X_dict, gif_name=None, cameras=None, just_render=False):
    # Remove NTU120 if needed
    if not ntu_120:
        X_dict = {k: v for k, v in X_dict.items() if int(k[17:20]) <= 60}

    if only_ntu_120:
        X_dict = {k: v for k, v in X_dict.items() if int(k[17:20]) > 60}

    # Remove cameras if needed
    if cameras is not None:
        X_dict = {k: v for k, v in X_dict.items() if int(k[7]) in cameras}

    print(f'Number of files: {len(X_dict)}')

    if not just_render:
        # Calculate MSE
        anon = torch.zeros((len(X_dict), 300, 75))
        raw = torch.zeros((len(X_dict), 75, 25, 3))
        rem=0
        for i, file in enumerate(X_dict):
            if only_use_pos:
                if type(file) != np.bytes_: file_byte = file.split('.')[0].encode('utf-8')
                else: file_byte = file
            else: file_byte = file
            
            # Ensure file exists in both dicts
            if file_byte not in X or file not in X_dict:
                rem+=1
                continue
            anon[i] = torch.tensor(X_dict[file])
            if only_use_pos:
                raw[i] = X[file_byte]
            else:
                raw[i] = X[file_byte][:, :, :3]

        # Remove non existent files
        anon = anon[:len(X_dict)-rem]
        raw = raw[:len(X_dict)-rem]

        # Remove zeros from the end of the sequence
        for i in range(anon.shape[1]):
            if not torch.all(anon[:, i] == 0):
                anon = anon[:, :i+1]
                raw = raw[:, :i+1]
                break

        # Reshape anon to be 75, 25, 3
        if anon.shape[1] > 75: anon = anon[:, :75, :]
        anon = anon[:, :, :75]
        anon = anon.reshape((anon.shape[0], anon.shape[1], 25, 3))

        # Calculate MSE
        mse = torch.mean((anon - raw)**2, dim=3)
        l2 = torch.mean(torch.sqrt(torch.sum((anon-raw)**2, dim=3)))
        print(f'MSE:\t\t\t\t{torch.mean(mse)}\nL2:\t\t\t\t{l2}\n')
        # return
        # Pre-allocate memory for the output array
        x = np.zeros((len(X_dict), 300, 150), dtype=np.float32)
        y_util = np.zeros(len(X_dict))
        y_priv = np.zeros(len(X_dict))
        y_gender = np.zeros(len(X_dict))

        for i, file in enumerate(X_dict):
            if X_dict[file].shape[1] == 75:
                X_dict[file] = np.pad(X_dict[file], ((0, 0), (0, 75)), 'constant')

            x[i] = np.array(X_dict[file], dtype=np.float32)
            y_util[i] = int(file[17:20])
            y_priv[i] = int(file[9:12])
            y_gender[i] = genders.loc[y_priv[i]-1, 'Gender']

        y_util = y_util - 1
        y_priv = y_priv - 1
        y_util = np.eye(utility_classes)[y_util.astype(int)]
        y_priv = np.eye(privacy_classes)[y_priv.astype(int)]

        print(x.shape)

        acc, f1, prec, recall, topk = run_sgn_eval(sgn_train_x, sgn_train_y, x, y_util, sgn_val_x, sgn_val_y, 1, sgn_ar, k=k)
        print(f'Utility Accuracy:\t\t{acc}\nUtility F1:\t\t\t{f1*100}\nUtility Precision:\t\t{prec*100}\nUtility Recall:\t\t\t{recall*100}\nTop-{k} Accuracy:\t\t\t{topk}\n')

        acc, f1, prec, recall, topk = run_sgn_eval(sgn_train_x, sgn_train_y, x, y_priv, sgn_val_x, sgn_val_y, 1, sgn_priv, k=k)
        print(f'Privacy Accuracy:\t\t{acc}\nPrivacy F1:\t\t\t{f1*100}\nPrivacy Precision:\t\t{prec*100}\nPrivacy Recall:\t\t\t{recall*100}\nTop-{k} Accuracy:\t\t\t{topk}\n')

        # Gender classification
        acc, f1 = run_sgn_gender_eval(sgn_train_x, sgn_train_y, x, y_priv, sgn_val_x, sgn_val_y, sgn_priv)
        print(f'Gender Classificiation Accuracy:\t{acc}\nF1:\t\t\t\t{f1*100}\n')
    return
    if gif_name is None: return
    print(gif_name, ' preparing to render')
    for key in X_dict:
        print(key)
        break
    print(eval_renders_str_skeleton)
    print(eval_renders_str)
    print(eval_render_byte)
    
    # Trim the data to only the first 75 frames, remove second actor, and remove zeroed out entries
    for file in X_dict:
        if X_dict[file].shape[0] != 75:
            if X_dict[file].shape[1] != 75: X_dict[file] = X_dict[file][:75, :75]
            else: X_dict[file] = X_dict[file][:75, :]
        elif X_dict[file].shape[1] != 75: X_dict[file] = X_dict[file][:, :75]

        for i in range(75):
            if np.all(X_dict[file][i] == 0):
                X_dict[file] = X_dict[file][:i]
                break
    
    # Render the videos
    for file in eval_renders_str_skeleton: 
        if file in X_dict:
            print('rendering')
            render_video(X_dict[file], gif=f'{gif_name}_{file[16:20]}, {file[8:12]}', show_render=False)
            print(file)
    for file in eval_renders_str: 
        if file in X_dict:
            print('rendering')
            render_video(X_dict[file], gif=f'{gif_name}_{file[16:20]}, {file[8:12]}', show_render=False)
            print(file)
    for file in eval_render_byte:
        if file in X_dict:
            print('rendering')
            a = file[16:20].decode('utf-8')
            p = file[8:12].decode('utf-8')
            render_video(X_dict[file], gif=f'{gif_name}_{a}, {p}', show_render=False)
            print(file)

# %%
def process_data(x_pkl, from_moon=False, pad_data=True, gif_name='', cameras=None, just_render=False):
    with open(x_pkl, 'rb') as f:
        test_x = pickle.load(f)

    if from_moon:
        test_x = {k: v[0] for k, v in test_x.items()}
        for file in test_x:
            # Assuming anonymizer_to_sgn is a predefined function you have
            test_x[file] = anonymizer_to_sgn(test_x[file])[:, :75]

    if pad_data:
        for file in test_x:
            if test_x[file].shape[0] == 1:
                test_x[file] = test_x[file][0]
            test_x[file] = np.pad(test_x[file], ((0, 300-test_x[file].shape[0]), (0, 0)), 'constant')

    # If keys are bytes, convert to string
    if type(list(test_x.keys())[0]) == np.bytes_: test_x = {k.decode('utf-8'): v for k, v in test_x.items()}

    eval(test_x, gif_name=gif_name, cameras=cameras, just_render=just_render)

datasets = {
    # 'dmr_random_CA': ('results/DMR_X_hat_constant.pkl', False, True),
    # 'dmr_constant_CA': ('results/DMR_X_hat_random.pkl', False, True),
    # 'pmr_random_RA': ('results/X_hat_random_RA.pkl', False, True),
    # 'pmr_constant_RA': ('results/X_hat_constant_RA.pkl', False, True),
    # 'pmr_random_CA': ('results/X_hat_random_CA.pkl', False, True),
    # 'pmr_constant_CA': ('results/X_hat_constant_CA.pkl', False, True),
    # 'dmr_random_RA': ('results/DMR_X_hat_random_RA.pkl', False, True),
    # 'dmr_constant_RA': ('results/DMR_X_hat_constant_RA.pkl', False, True),
    # 'dmr_random_CA': ('results/DMR_X_hat_random_CA.pkl', False, True),
    # 'dmr_constant_CA': ('results/DMR_X_hat_constant_CA.pkl', False, True),
    # 'moon_unet': ('C:\\Users\\Carrt\\OneDrive\\Code\\Linkage Attack\\External Repositories\\Skeleton-anonymization\\X_unet_file.pkl', True, False),
    # 'moon_resnet': ('C:\\Users\\Carrt\\OneDrive\\Code\\Linkage Attack\\External Repositories\\Skeleton-anonymization\\X_resnet_file.pkl', True, False),
    # 'cmr': ('C:\\Users\\Carrt\\OneDrive\\Code\\Motion Privacy\\Defense Models\\Mean Skeleton\\X_FileNameKey_SingleActor_filtered.pkl', False, False)
    # 'pmr_random_CA': ('results/NTU_120_PMR_X_hat_random_CA.pkl', False, True),
    # 'pmr_constant_CA': ('results/NTU_120_PMR_X_hat_constant_CA.pkl', False, True),
    # 'pmr_random_RA': ('results/NTU120_PMR_X_hat_random_RA.pkl', False, True),
    # 'pmr_constant_RA': ('results/NTU120_PMR_X_hat_constant_RA.pkl', False, True),
    'dmr_random_CA': ('results/NTU_120_DMR_X_hat_random_CA.pkl', False, True),
    'dmr_constant_CA': ('results/NTU_120_DMR_X_hat_constant_CA.pkl', False, True),
    'dmr_random_RA': ('results/NTU120_DMR_X_hat_random_RA.pkl', False, True),
    'dmr_constant_RA': ('results/NTU120_DMR_X_hat_constant_RA.pkl', False, True),
}

# %%
# Process all datasets
for gif_name, (x_pkl, from_moon, pad_data) in datasets.items():
    print(f'Processing {gif_name}')
    process_data(x_pkl, from_moon=from_moon, pad_data=pad_data, gif_name=gif_name, just_render=False)

# %%
# Process specific dataset
ds = 'pmr_random_CA'
process_data(datasets[ds][0], from_moon=datasets[ds][1], pad_data=datasets[ds][2], gif_name=ds, cameras=test_cameras)

# %%
# Raw
if only_use_pos:
    with open('ntu/SGN/X_full.pkl', 'rb') as f:
        raw = pickle.load(f)
else:
    with open('ntu/X.pkl', 'rb') as f:
        raw = pickle.load(f)

for file in raw:
    if only_use_pos:
        # chop from (300, 150) to (300, 75)
        raw[file] = raw[file][:, :75]
    else:
        # reshape from (frame, 25, 3) to (frame, 75)
        raw[file] = raw[file][:, :, :3].reshape((raw[file].shape[0], 75))
        # pad data to 300 frames
        if raw[file].shape[0] == 300: continue
        raw[file] = np.pad(raw[file], ((0, 300-raw[file].shape[0]), (0, 0)), 'constant')
        if raw[file].shape != (300, 75): 
            print(file, raw[file].shape)
            del raw[file]

eval(raw, gif_name=None)

# %%
# for file in eval_render_byte:
#     a = file[16:20].decode('utf-8')
#     p = file[8:12].decode('utf-8')
#     render_video(X[file], gif=f'raw_{a}, {p}', show_render=False)
#     print(file)

# %% [markdown]
# # Clustering

# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
model = val_model

# %%
# Action clustering
d = []
y = []
for key in tqdm(X.keys()):
        embedding = model.dynamic_encoder(X[key].unsqueeze(0).float().cuda()).cpu().detach().numpy().flatten()
        d.append(embedding)
        y.append(int(key[17:20]))

d = np.array(d)
y = np.array(y)

# Perform TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(d)

# Create a DataFrame for Plotly
df = pd.DataFrame(tsne_results, columns=['TSNE-1', 'TSNE-2'])
df['label'] = y.astype(str)  # Convert labels to string for better categorical handling

# Create an interactive scatter plot
fig = px.scatter(df, x='TSNE-1', y='TSNE-2', color='label',
                 color_discrete_sequence=px.colors.qualitative.G10,
                 labels={'label': 'Label'},
                 title='t-SNE results with Interactive Labels')

# Add interactive functionality for toggling visibility
label_buttons = [dict(label='All',
                      method='update',
                      args=[{'visible': [True] * len(df['label'].unique())},
                            {'title': 't-SNE results with Interactive Labels'}])]

for label in df['label'].unique():
    label_buttons.append(dict(label=f'Label {label}',
                              method='update',
                              args=[{'visible': [lbl == label for lbl in df['label']]},
                                    {'title': f't-SNE: Label {label}'}]))

fig.update_layout(showlegend=False,
                  updatemenus=[dict(active=0,
                                    buttons=label_buttons,
                                    x=0.0,
                                    xanchor='left',
                                    y=1.2,
                                    yanchor='top')])

# Show plot
fig.show()

# %%
actions = set([6, 9, 22, 25, 29, 38, 39])

# Actor clustering
d = []
y = []
for key in tqdm(X.keys()):
    if int(key[17:20]) in actions:
        embedding = model.dynamic_encoder(X[key].unsqueeze(0).float().cuda()).cpu().detach().numpy().flatten()
        d.append(embedding)
        y.append(int(key[17:20]))

d = np.array(d)
y = np.array(y)

# Map labels
label_mapping = {
    '6': 'pick up',
    '7': 'throw',
    '8': 'sit down',
    '9': 'stand up',
    '16': 'put on a shoe',
    '17': 'take off shoe',
    '22': 'put on glasses',
    '23': 'hand waving',
    '25': 'reach into pocket',
    '29': 'play with phone/tablet',
    '38': 'salute',
    '39': 'put palms together',
    '40': 'cross hands in front',
    '42': 'staggering',
    '43': 'falling down'
}
y_mapped = [label_mapping[str(label)] for label in y]

# Perform TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=2000)
tsne_results = tsne.fit_transform(d)

# Plot
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=y_mapped,
    palette=sns.color_palette("hls", len(actions)),
    legend="full",
    alpha=0.3
)

# %%
# Actor Clustering
d_act = []
y_act = []
for key in tqdm(X.keys()):
#     if int(key[9:12]) in actors:
        embedding = model.static_encoder(X[key].unsqueeze(0).float().cuda()).cpu().detach().numpy().flatten()
        d_act.append(embedding)
        y_act.append(int(key[9:12]))

d_act = np.array(d_act)
y_act = np.array(y_act)

# Perform TSNE
tsne_act = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=2000, learning_rate='auto')
tsne_results_act = tsne_act.fit_transform(d_act)

df = pd.DataFrame(tsne_results_act, columns=['TSNE-1', 'TSNE-2'])
df['label'] = y_act.astype(str)  # Convert labels to string for better categorical handling

# Create an interactive scatter plot
fig = px.scatter(df, x='TSNE-1', y='TSNE-2', color='label',
                 color_discrete_sequence=px.colors.qualitative.G10,
                 labels={'label': 'Label'},
                 title='t-SNE results with Interactive Labels')

def create_visibility_list(selected_label, all_labels):
    return [label == selected_label for label in all_labels]

# Initialize visibility: Initially show all
initial_visibility = [True] * len(df['label'].unique())

# Create buttons
label_buttons = [dict(label='All',
                      method='update',
                      args=[{'visible': initial_visibility},
                            {'title': 't-SNE results with Interactive Labels'}])]

for label in df['label'].unique():
    visibility_list = create_visibility_list(label, df['label'].unique())
    label_buttons.append(dict(label=f'Label {label}',
                              method='update',
                              args=[{'visible': visibility_list},
                                    {'title': f't-SNE: Label {label}'}]))

fig.update_layout(showlegend=False,
                  updatemenus=[dict(active=0,
                                    buttons=label_buttons,
                                    x=0.0,
                                    xanchor='left',
                                    y=1.2,
                                    yanchor='top')])

# Show plot
fig.show()

# %%
actors = set([27, 28, 35, 37])

# Actor clustering
d_act = []
y_act = []
for key in tqdm(X.keys()):
    if int(key[9:12]) in actors:
        embedding = model.static_encoder(X[key].unsqueeze(0).float().cuda()).cpu().detach().numpy().flatten()
        d_act.append(embedding)
        y_act.append(int(key[9:12]))

d_act = np.array(d_act)
y_act = np.array(y_act)

# # Perform TSNE
tsne_act = TSNE(n_components=2, verbose=1)#, perplexity=100, n_iter=2000, learning_rate='auto')
tsne_results_act = tsne_act.fit_transform(d_act)

y_act_str = [f'Actor: {x}' for x in y_act]

# Plot
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results_act[:,0], y=tsne_results_act[:,1],
    hue=y_act_str,
    palette=sns.color_palette("hls", len(np.unique(y_act))),
    legend="full",
    alpha=0.3
)

# %%
# Render action 1 from each actor above
action = 1
for actor in actors:
    for key in X:
        if int(key[9:12]) == actor and int(key[17:20]) == action:
            render_video(X[key])
            break

# %% [markdown]
# # Sampling

# %%
# Test DMR MSE
in_skeleton = []
for key in X:
    in_skeleton.append(X[key].unsqueeze(0).float())

in_skeleton = torch.cat(in_skeleton, dim=0).to(device)
# split into batches
mse = []
for i in tqdm(range(0, len(in_skeleton), batch_size)):
    x = in_skeleton[i:i+batch_size]
    x_hat = dmr(x, x)
    x = x.view(x.shape[0], x.shape[1], -1)
    mse.append(F.mse_loss(x_hat, x).item())

print(f'MSE: {np.mean(mse)}')

# %%
def reshape_skeleton(skeleton):
    """
    Reshape the skeleton data from shape [1, 75, 25, 3] to [300, 150].
    """
    skeleton = skeleton.squeeze(0)  # Remove the batch dimension, shape becomes [75, 25, 3]
    skeleton = skeleton.reshape(75, -1)  # Reshape to [75, 75]
    
    # Pre-allocate the output with zeros to match the expected [300, 150] shape
    padded_skeleton = torch.zeros((20, 75), dtype=torch.float32).cpu()
    
    # Fill in the existing data
    padded_skeleton = skeleton[:20]
    
    return padded_skeleton

sgn_train_x, sgn_train_y, sgn_val_x, sgn_val_y = np.zeros((batch_size, 300, 150)), np.zeros((batch_size, 1)), np.zeros((batch_size, 300, 150)), np.zeros((batch_size, 1))


def predict_sgn(model, skeleton):
    skeleton = torch.tensor(skeleton).cuda()
    model.cuda()
    out = model.eval_single(skeleton)
    out = out.view((-1, skeleton.size(0)//skeleton.size(0), out.size(1)))
    out = out.mean(1)
    out = out.cpu().detach().numpy()
    out = np.argmax(out, axis=1)
    return out[0]


def sample_motion_retargeting(dataset, motion_retargeting_model, action_recognition_model, re_identification_model, target_sample_count=50, verbose=False):
    successful_samples = {}
    successful_sample_map = {}

    # Continue sampling until we have enough successful samples
    while len(successful_samples) < target_sample_count:
        # Randomly select a reference skeleton
        reference_file = random.sample(list(dataset.keys()), 1)[0]
        reference_skeleton = dataset[reference_file].unsqueeze(0)
        reference_action, reference_actor = parse_file_name(reference_file)['A'], parse_file_name(reference_file)['P']

        # Randomly select a dummy skeleton that is not the same as the reference skeleton
        while True:
            dummy_file = random.sample(list(dataset.keys()), 1)[0]
            dummy_identity = parse_file_name(dummy_file)['P']
            if dummy_identity != reference_actor:
                dummy_skeleton = dataset[dummy_file].unsqueeze(0)
                break

        if verbose: print(f"Attemtping to anonymize {reference_file} using {dummy_file}")

        # Perform motion retargeting
        anonymized_skeleton = motion_retargeting_model(reference_skeleton.cuda(), dummy_skeleton.cuda()).cpu().squeeze()

        # Reshape the anonymized skeleton
        anonymized_skeleton = reshape_skeleton(anonymized_skeleton)
        
        # Evaluate action recognition (utility)
        anonymized_action = predict_sgn(action_recognition_model, anonymized_skeleton.unsqueeze(0).detach().numpy()) + 1 # +1 to match the original indexing

        # Evaluate re-identification (privacy)
        anonymized_identity = predict_sgn(re_identification_model, anonymized_skeleton.unsqueeze(0).detach().numpy()) + 1 # +1 to match the original indexing

        # Keep the sample if it meets both conditions (action correct, identity anonymized)
        if anonymized_action == reference_action and anonymized_identity == dummy_identity:
            successful_samples[reference_file] = anonymized_skeleton
            successful_sample_map[reference_file] = dummy_file
            print(f"Sample {len(successful_samples)} added: {reference_file}")
        else:
            if verbose: print(f"Sample rejected: {reference_file} using {dummy_file}. Predicted action: {anonymized_action.item()}, Actual action: {reference_action}. Predicted identity: {anonymized_identity.item()}, Actual identity: {dummy_identity}")

    return successful_samples, successful_sample_map


# %%
samples, key_to_dummy = sample_motion_retargeting(X, model, sgn_ar, sgn_priv)
samples

# %%
# Save the samples
with open('results/samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
with open('results/samples_map.pkl', 'wb') as f:
    pickle.dump(key_to_dummy, f)

# %%
key_to_dummy


