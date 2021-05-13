import torch, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
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


def cw_l2_attack(model,
                 images,
                 labels,
                 targeted=False,
                 c=0.1,
                 kappa=0,
                 max_iter=1000,
                 learning_rate=0.01):

    # Define f-function
    def f(x):
        # 论文中的 Z(X) 输出 batchsize, num_classes
        outputs = model(x)
        # batchszie，根据labels 的取值，确定每一行哪一个为 1
        # >>> a = torch.eye(10)[[2, 3]]
        # >>> a
        # tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        # 水平方向最大的取值，忽略索引。意思是，除去真实标签，看看每个 batchsize 中哪个标签的概率最大，取出概率
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        # 选择真实标签的概率
        j = torch.masked_select(outputs, one_hot_labels.bool())

        # 如果有攻击目标，虚假概率减去真实概率，
        if targeted:
            return torch.clamp(i - j, min=-kappa)
        # 没有攻击目标，就让真实的概率小于虚假的概率，逐步降低，也就是最小化这个损失
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    #
    prev = 1e10

    for step in range(max_iter):
        a = 1 / 2 * (nn.Tanh()(w) + 1)
        # 第一个目标，对抗样本与原始样本足够接近
        loss1 = nn.MSELoss(reduction='sum')(a, images)
        # 第二个目标，误导模型输出
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost

    attack_images = 1 / 2 * (nn.Tanh()(w) + 1)

    return attack_images


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
data_loader = DataLoader(dataset=data_set, batch_size=200, shuffle=False)

save_data = None
print("cw attack...")
for data, target in tqdm(data_loader):
    data, target = data.to(device), target.to(device)

    perturbed_data = cw_l2_attack(model=model, images=data, labels=target)

    if save_data is None:
        save_data = perturbed_data.detach_().cpu().numpy()
    else:
        save_data = np.concatenate(
            (save_data, perturbed_data.detach_().cpu().numpy()), axis=0)
np.save('adversial/adv_data/cifar10/CW2/cw2_cifar10.npy', save_data)
print('cw2_cifar10 has been saved')
