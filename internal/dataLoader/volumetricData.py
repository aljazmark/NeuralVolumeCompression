import torch.utils.data as data
import numpy as np
import torch
import json
import zipfile
import glob
import random
import gzip
import io

class VolumetricData(data.Dataset):
    def __init__(self, data_path, split=0.8,block_size=32):
        self.data_path = data_path
        self.split = split
        blocks,manifest = readBvpFiles(self.data_path)
        num_blocks = len(blocks)
        self.inputs = []
        mini_size = block_size
        # Load data here, e.g.
        for b in range(0, num_blocks):
            input_data = blocks[b][0]
            dims = blocks[b][1]
            input_data = toVolume(input_data,dims)
            input_data = normalizeVolume(input_data)
            input_data = padVolume(input_data,mini_size)
            input_data = toMiniVolumes(input_data,mini_size)
            self.inputs.extend(input_data)

        random.shuffle(self.inputs)
        self.labels = self.inputs
        self.train_inputs = self.inputs
        self.train_labels = self.labels
        # Split data into train and test sets
        #self.train_inputs, self.test_inputs, self.train_labels, self.test_labels = train_test_split(
        #    self.inputs, self.labels, train_size=self.split, random_state=32)

    def __len__(self):
        return len(self.train_inputs)

    def __getitem__(self, idx):
        input_data = self.train_inputs[idx]
        label_data = self.train_labels[idx]
        input_tensor = torch.from_numpy(input_data).unsqueeze(0)
        label_tensor = torch.from_numpy(label_data).unsqueeze(0)
        sample = {'input': input_tensor, 'label': label_tensor}
        return sample

    def get_test_set(self):
        return self.test_inputs, self.test_labels

class TestingData(data.Dataset):
    def __init__(self, train_data):
        self.test_inputs = train_data[0]
        self.test_labels = train_data[1]

    def __len__(self):
        return len(self.test_inputs)

    def __getitem__(self, idx):
        input_data = self.test_inputs[idx]
        label_data = self.test_labels[idx]
        input_tensor = torch.from_numpy(input_data).unsqueeze(0)
        label_tensor = torch.from_numpy(label_data).unsqueeze(0)
        sample = {'input': input_tensor, 'label': label_tensor}
        return sample


def padVolume(volume,mini_size=32):
    dims = volume.shape
    # Pad the input data to a size that is a multiple of mini_size in all dimensions
    padding = [(0, (mini_size - dims[i] % mini_size) % mini_size) for i in range(3)]
    input_data = np.pad(volume, padding, mode='symmetric')
    return input_data

def unPadVolume(volume, original_dims):
    return volume[:original_dims[0], :original_dims[1], :original_dims[2]]

def toVolume(volume,dims):
    return np.reshape(volume, dims)

def toVector(volume):
    return volume.reshape(-1)


def toMiniVolumes(volume,mini_size):
    org_size = volume.shape
    if org_size[0] == mini_size and org_size[1] == mini_size and org_size[2] == mini_size:
        return np.expand_dims(volume,0)
    else:
        return volume.reshape((-1, mini_size, mini_size, mini_size))

def normalizeVolume(volume):
    return volume.astype(np.float32) / 255

def deNormalizeVolume(volume):
    return np.abs(volume * 255).round()

def readBvpFiles(folderPath):
    #bvp_files = glob.glob(folderPath+'/*.bvp')
    bvp_file = glob.glob(folderPath)
    blocks = []
    for file_path in bvp_file:
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            with zip_file.open("manifest.json", 'r') as file:
                data = json.load(file)
                for sublist in data['blocks']:
                    with zip_file.open(sublist['url'], 'r') as bfile:
                        rawblock = np.frombuffer(bfile.read(), dtype=np.uint8)
                        dims = [sublist['dimensions']['width'], sublist['dimensions']['height'],
                                sublist['dimensions']['depth']]
                        blockname = sublist['url'].split('/')[2]
                        blocks.append([rawblock, dims, blockname])
    return blocks,data



def readGZIPFiles(folderPath):
    gzip_files = glob.glob(folderPath+'/*.gz')
    blocks = []
    for file_path in gzip_files:
        with gzip.open(file_path, 'rb') as f:
            buffer = f.read()
        blocks.append(torch.load(io.BytesIO(buffer)))

    return blocks

def getCompressionBlocks(filePath):
    blocks = readBvpFiles(filePath)
    output = []
    for i in range(0, len(blocks)):
        input_data = blocks[i][0]
        dims = blocks[i][1]
        # Reshape and normalize input data
        input_data = np.reshape(input_data, (dims[0], dims[1], dims[2]))
        input_data = input_data.astype(np.float32) / 255.0
        # Pad the input data to a size of 128x128x128
        #input_shape = input_data.shape
        #padding = [(0, max(0, 128 - input_shape[i])) for i in range(3)]
        #input_data = np.pad(input_data, padding, mode='constant')
        #input_tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
        input_data = input_data.reshape((-1, 32, 32, 32))
        input_data = torch.from_numpy(input_data).unsqueeze(1)
        output.append([input_data,blocks[i][2]])

    return output
