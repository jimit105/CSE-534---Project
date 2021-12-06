import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import gc

class AWF_Data(Dataset):
    def __init__(self, path_data, path_labels):
        with open(path_data, 'rb') as f:
            self.data_npy = np.load(f, allow_pickle=True)
        self.raw_data = self.data_npy.astype(dtype='float_')
        self.data = np.transpose(self.raw_data.reshape(self.raw_data.shape[0], self.raw_data.shape[1], 1), (0, 2, 1))

        with open(path_labels, 'rb') as f:
            self.labels_npy = np.load(path_labels, allow_pickle=True)

        dict_labels = None
        possible_labels = list(set(self.labels_npy))

        if not dict_labels:
            dict_labels = {}
            n = 0
            for label in possible_labels:
                dict_labels[label] = n
                n = n + 1

        new_labels = []
        for label in self.labels_npy:
            new_labels.append(dict_labels[label])

        self.label = new_labels

        self.data_len = len(self.data)
        
    def __getitem__(self, index):
        single_label = self.label[index]
        return (self.data[index], single_label)
 
    def __len__(self):
        return self.data_len


class AWF_Network(nn.Module):
    def __init__(self, nb_classes):
        super(AWF_Network, self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv1d(1, 32, 8, 1, 0),
            nn.BatchNorm1d(32), 
            nn.ELU(),                     
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.2), 
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(39904, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, nb_classes)
        )   

    def forward(self, x):
        x = self.conv1(x)
        output = self.out(x)
        return output, x  


NB_CLASSES = 101
BATCH_SIZE = 128
LR = 0.001
print("loading dataset")
print(gc.collect())

full_data = AWF_Data("./data/data.npy", "./data/labels.npy")
print(gc.collect())
print("loading complete")

train_split= 0.9
validate_split = 0.05
test_split = 0.05
shuffle_dataset = True
random_seed = 16

dataset_size = len(full_data)
indices = list(range(dataset_size))
train_size = int(train_split * dataset_size)
validation_size = int(validate_split * dataset_size)
test_size = int(dataset_size - train_size - validation_size)

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices= indices[:train_size], indices[train_size:train_size+validation_size], indices[train_size+validation_size:]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(full_data, batch_size=BATCH_SIZE,sampler=train_sampler)
validation_loader = DataLoader(full_data, batch_size=BATCH_SIZE,sampler=valid_sampler)
test_loader = DataLoader(full_data, batch_size=BATCH_SIZE,sampler=test_sampler)

cuda_gpu = torch.cuda.is_available()
cnn = AWF_Network(NB_CLASSES).float()
if(cuda_gpu):
    cnn = torch.nn.DataParallel(cnn, device_ids=[0]).cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


print("starting training")
writer = SummaryWriter()
temp_counter = 0
for epoch in range(1):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = cnn(b_x.float())[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            corrects = 0
            avg_loss = 0
            for (b_x, b_y) in validation_loader:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                logit = cnn(b_x.float())[0]
                loss = loss_func(logit, b_y)
                avg_loss += loss.item()
                corrects += (torch.max(logit, 1)
                            [1].view(b_y.size()).data == b_y.data).sum()
            
            size = validation_size
            avg_loss /= size
            accuracy = 100.0 * corrects / size
            print('Epoch: {:2d}({:6d}/{}) Evaluation - loss: {:.6f}  acc: {:3.4f}%({}/{})'.format(
                                                                            epoch,
                                                                            step * 128,
                                                                            train_size,
                                                                            avg_loss, 
                                                                            accuracy, 
                                                                            corrects, 
                                                                            size))
        writer.add_scalar("loss/training", avg_loss, temp_counter)
        writer.add_scalar("accuracy/training", accuracy, temp_counter)
        temp_counter+=1

        writer.flush()
writer.close()


print("Saving Model")
torch.save(cnn, './model/awf.pkl')
print("Model Saved")


corrects = 0
avg_loss = 0
for (b_x, b_y) in test_loader:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                logit = cnn(b_x.float())[0]
                loss = loss_func(logit, b_y)
                avg_loss += loss.item()
                corrects += (torch.max(logit, 1)
                            [1].view(b_y.size()).data == b_y.data).sum()
            
size = test_size
accuracy = 100.0 * corrects / size
print("accuracy: {:3.4f}%".format(accuracy))
