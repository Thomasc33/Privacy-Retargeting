import pickle
from SGN.model import SGN
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from SGN.data import AverageMeter
with open('NTU/SGN/X_full.pkl', 'rb') as f:
    X = pickle.load(f)
T=75

import argparse

parser = argparse.ArgumentParser(description='SGN')
parser.add_argument('--action', type=int, default=42, help='Action class to evaluate')
action = parser.parse_args().action
action = action - 1

for file, value in X.items():
    # If SGN preprocessing, remove zero padding
    if True:
        first_zero_index = value.shape[0]
        for i in range(value.shape[0]):
            if np.all(value[i] == 0):
                first_zero_index = i
                break
        value = value[:first_zero_index]

    num_frames = value.shape[0]

    # Pad or trim
    if num_frames < T:
        if True: padding = np.repeat(value[-1][np.newaxis, :], T - num_frames, axis=0)
        else: padding = np.repeat(value[-1][np.newaxis, :, :], T - num_frames, axis=0)
        value = np.concatenate((value, padding), axis=0)
    elif num_frames > T:
        # Randomly sample T frames
        start = random.randint(0, num_frames - T)
        value = value[start:start+T]
    
    # Convert to tensor and store back
    X[file] = torch.from_numpy(value).float()
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


x = []
y = []

for file, value in X.items():
    x.append(value)
    y.append(parse_file_name(file)['A']-1)

# pad all sequences to the same 300
for i in range(len(x)):
    x[i] = torch.cat((x[i], torch.zeros(300 - x[i].shape[0], 150)), 0)

x = np.array(x)
y = np.array(y)
sgn_train_x, sgn_train_y, sgn_val_x, sgn_val_y = np.zeros((32, 300, 150)), np.zeros((32, 1)), np.zeros((32, 300, 150)), np.zeros((32, 1))
model = SGN(60, None, 20, 32, 0).to('cuda')
model.load_state_dict(torch.load('SGN/pretrained/action_60_sgnpt.pt')['state_dict'], strict=False)
from SGN.data import NTUDataLoaders
data = NTUDataLoaders(case=0, seg=20, train_X=sgn_train_x, train_Y=sgn_train_y, val_X=sgn_val_x, val_Y=sgn_val_y, test_X=x, test_Y=y)
dl = data.get_test_loader(32, None)
from sklearn.metrics import f1_score, precision_score, recall_score

def test(test_loader, model, k=3):
    acces = AverageMeter()
    topk_acces = AverageMeter()
    model.eval()

    label_output = []
    pred_output = []

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

    # Since targets might not be one-hot encoded, ensure they are class indices
    label_index = label_output.astype(int).flatten()
    pred_index = np.argmax(pred_output, axis=1)

    f1 = f1_score(label_index, pred_index, average='macro', zero_division=0)
    precision = precision_score(label_index, pred_index, average='macro', zero_division=0)
    recall = recall_score(label_index, pred_index, average='macro', zero_division=0)

    f1_class = f1_score(label_index, pred_index, labels=[action], average=None, zero_division=0)[0]

    print(f"F1 score for class {action+1}: {f1_class}")

    return acces.avg, f1, precision, recall, topk_acces.avg, f1_class

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    return correct.mul_(100.0 / batch_size)

def top_k_accuracy(output, target, k=3):
    batch_size = target.size(0)
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

acc_avg, f1_macro, precision_macro, recall_macro, topk_acc_avg, f1_class = test(dl, model, k=3)
print("Raw data")
print(f"Overall Accuracy: {acc_avg}")
print(f"Macro F1 Score: {f1_macro}")
print(f"F1 Score for Class {action+1}: {f1_class}")
# pmr data
def process_data(x_pkl, from_moon=False, pad_data=True, gif_name='', cameras=None, just_render=False):
    with open(x_pkl, 'rb') as f:
        test_x = pickle.load(f)

    if pad_data:
        for file in test_x:
            if test_x[file].shape[0] == 1:
                test_x[file] = test_x[file][0]
            test_x[file] = np.pad(test_x[file], ((0, 300-test_x[file].shape[0]), (0, 0)), 'constant')

    # If keys are bytes, convert to string
    if type(list(test_x.keys())[0]) == np.bytes_: test_x = {k.decode('utf-8'): v for k, v in test_x.items()}

    return test_x
# 'pmr_constant_CA': ('results/X_hat_constant_CA.pkl', False, True),
X = process_data('results/X_hat_constant_CA.pkl', pad_data=True)
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


x = []
y = []

for file, value in X.items():
    x.append(value)
    y.append(parse_file_name(file)['A']-1)


x = np.array(x)
y = np.array(y)
x = np.array([np.pad(frame, ((0, 0), (0, 75)), mode='constant') for frame in x])

data = NTUDataLoaders(case=0, seg=20, train_X=sgn_train_x, train_Y=sgn_train_y, val_X=sgn_val_x, val_Y=sgn_val_y, test_X=x, test_Y=y)
dl = data.get_test_loader(32, None)

acc_avg, f1_macro, precision_macro, recall_macro, topk_acc_avg, f1_clas = test(dl, model, k=3)
print('\nPMR Constant CA data')
print(f"Overall Accuracy: {acc_avg}")
print(f"Macro F1 Score: {f1_macro}")
print(f"F1 Score for Class {action+1}: {f1_class}")
# 'dmr_constant_CA': ('results/DMR_X_hat_constant_CA.pkl', False, True),
X = process_data('results/DMR_X_hat_constant_CA.pkl', pad_data=True)
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


x = []
y = []

for file, value in X.items():
    x.append(value)
    y.append(parse_file_name(file)['A']-1)


x = np.array(x)
y = np.array(y)
x = np.array([np.pad(frame, ((0, 0), (0, 75)), mode='constant') for frame in x])

data = NTUDataLoaders(case=0, seg=20, train_X=sgn_train_x, train_Y=sgn_train_y, val_X=sgn_val_x, val_Y=sgn_val_y, test_X=x, test_Y=y)
dl = data.get_test_loader(32, None)

acc_avg, f1_macro, precision_macro, recall_macro, topk_acc_avg, f1_class = test(dl, model, k=3)
print('\nDMR Constant CA data')
print(f"Overall Accuracy: {acc_avg}")
print(f"Macro F1 Score: {f1_macro}")
print(f"F1 Score for Class {action+1}: {f1_class}")