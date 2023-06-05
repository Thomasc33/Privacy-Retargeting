import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='D:\\Datasets\\Motion Privacy\\NTU RGB+D 120\\Skeleton Data')

args = parser.parse_args(args=[])

# Read the files
files = [f for f in listdir(args.data_path) if isfile(join(args.data_path, f))]

X = {}

# Get stats for each file based on name
files_ = []
for file in files:
    data = {'file': file,
            's': file[0:4],
            'c': file[4:8],
            'p': file[8:12],
            'r': file[12:16],
            'a': file[16:20]
            }
    files_.append(data)

# Generate X and Y
for file_ in tqdm(files_, desc='Files Parsed', position=0):
    try:
        file = join(args.data_path, file_['file'])
        data = open(file, 'r')
        lines = data.readlines()
        frames_count = int(lines.pop(0).replace('\n', ''))
        file_['frames'] = frames_count
    except UnicodeDecodeError: # .DS_Store file
        print('UnicodeDecodeError: ', file)
        continue

    # Get P and add to X if not already there
    p = file_['file']
    if p not in X:
        X[p] = []

    # Skip file if 2 actors
    if lines[0].replace('\n', '') != '1': continue

    for f in tqdm(range(frames_count), desc='Frames Parsed', position=1, leave=False):
        try:
            # Get actor count
            actors = int(lines.pop(0).replace('\n', ''))
            
            # Get actor info
            t = lines.pop(0)

            # Get joint count
            joint_count = int(lines.pop(0).replace('\n', ''))

            # Get joint info
            d = []
            for j in range(joint_count):
                joint = lines.pop(0).replace('\n', '').split(' ')
                d.extend(joint[0:3])

            # Skip if not 25 joints
            if len(d) != 75: continue

            # Convert to numpy array
            d = np.array(d)

            # Append to X and Y
            X[p].append(d)
        except:
            break
        
    # Convert to numpy array
    X[p] = np.array(X[p], dtype=np.float16)

print('X Generated, saving to pickle...')

# Save the data
with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)

print('X Saved to pickle')