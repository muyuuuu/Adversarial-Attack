import torch, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from resnet import ResNet50
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")


class Cifar_10_Dataset(Dataset):
    def __init__(self, data_path, label_path):
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3)
        var = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3)
        self.x = np.load(data_path) / 255
        self.x = (self.x - mean) / var
        self.x = self.x.transpose(0, 3, 1, 2)
        self.label = np.load(label_path)
        self.label = np.reshape(self.label, (self.x.shape[0], ))

        # https://stackoverflow.com/questions/44717100/pytorch-convert-floattensor-into-doubletensor
        self.x, self.label = torch.from_numpy(
            self.x).float(), torch.from_numpy(self.label)

    def __getitem__(self, index):
        return self.x[index], self.label[index]

    def __len__(self):
        return self.x.size(0)


def pgd_attack(model, images, labels, eps, alpha=2 / 255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        # 通过 eps 限制扰动球面的范围
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    return images


print('load model')
model = ResNet50()
pth_file = 'adversial/make_attack/resnet50_ckpt.pth'

d = torch.load(pth_file)['net']
d = OrderedDict([(k[7:], v) for (k, v) in d.items()])
model.load_state_dict(d)
model.to(device)
model.eval()

data_path = "adversial/clean_data/cifar10/cifar10_data.npy"
label_path = "adversial/clean_data/cifar10/cifar10_label.npy"

print('load data')
data_set = Cifar_10_Dataset(data_path=data_path, label_path=label_path)
data_loader = DataLoader(dataset=data_set, batch_size=128, shuffle=False)

for epsilon in [0.05, 0.1, 0.15, 0.3]:
    save_data = None
    print("{} attack...".format(epsilon))
    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)

        perturbed_data = pgd_attack(model=model,
                                    images=data,
                                    labels=target,
                                    eps=epsilon)

        if save_data is None:
            save_data = perturbed_data.cpu().numpy()
        else:
            save_data = np.concatenate(
                (save_data, perturbed_data.cpu().numpy()), axis=0)
    np.save('adversial/adv_data/cifar10/PGD/{}_cifar10.npy'.format(epsilon),
            save_data)
    print('{}_cifar10 has been saved'.format(epsilon))
