import torch, os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import numpy as np
from vgg import VGG

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")


# 取出后 10000 个作为测试
class Cifar_10_Dataset(Dataset):
    def __init__(self, data_path, label_path):
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3)
        var = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3)
        self.x = np.load(data_path)[50000:] / 255
        self.x = (self.x - mean) / var
        self.x = self.x.transpose(0, 3, 1, 2)
        self.label = np.load(label_path)[50000:]
        self.label = np.reshape(self.label, (self.x.shape[0], ))

        self.x, self.label = torch.from_numpy(
            self.x).float(), torch.from_numpy(self.label)

    def __getitem__(self, index):
        return self.x[index], self.label[index]

    def __len__(self):
        return self.x.size(0)


class Cifar_10_ADVDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.x = np.load(data_path)[50000:]
        self.label = np.load(label_path)[50000:]
        self.label = np.reshape(self.label, (self.x.shape[0], ))

        self.x, self.label = torch.from_numpy(
            self.x).float(), torch.from_numpy(self.label)

    def __getitem__(self, index):
        return self.x[index], self.label[index]

    def __len__(self):
        return self.x.size(0)


def acc(model, device, dataloader):
    total = 0
    correct = 0
    print('natural acc: ', end='')
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # inputs = purfier(inputs)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


print('load model')
net = VGG('VGG16')
pthfile = "adversial/make_attack/VGG16_ckpt.pth"
d = torch.load(pthfile)['net']
d = OrderedDict([(k[7:], v) for (k, v) in d.items()])
net.load_state_dict(d)
net.to(device)
net.eval()
# 加载 purifier

print('load data')
clean_path = "adversial/clean_data/cifar10/cifar10_data.npy"
label_path = "adversial/clean_data/cifar10/cifar10_label.npy"

dataset = Cifar_10_Dataset(clean_path, label_path)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

print(acc(net, device=device, dataloader=dataloader))

print('begin testing on adv')
data_root_path = 'adversial/adv_data/cifar10/'
adv_file = ['PGD/']
perturbation = [
    '0.05_cifar10.npy', '0.1_cifar10.npy', '0.15_cifar10.npy',
    '0.3_cifar10.npy'
]

# 验证在对抗样本上的准确率
for adv in adv_file:
    for path_ in perturbation:
        adv_data_path = data_root_path + adv + path_
        # print(adv, '  ', path_[0:-8], end=', ')

        advdata = Cifar_10_ADVDataset(adv_data_path, label_path)
        adv_loader = torch.utils.data.DataLoader(advdata, batch_size=128)
        print(
            'to pridict: ', path_,
            ', acc is {}'.format(acc(net, device=device,
                                     dataloader=adv_loader)))
