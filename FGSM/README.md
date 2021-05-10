- `fgsm_attack.py`，使用 `fgsm` 算法制作对抗样本
- `resnet`，使用 `ResNet50` 制作对抗样本
- `vgg`，黑盒攻击，使用 `VGG16` 作为目标模型
- `eval.py`，验证目标模型在对抗样本攻击下的准确率

| 攻击模型 | 目标模型 | 原始准确率 | 扰动0.05 | 扰动0.1 | 扰动0.15 | 扰动0.3 |
| -------- | -------- | ---------- | -------- | ------- | -------- | ------- |
| ResNet50 | VGG16    | 94.27      | 52.52    | 47.37   | 42.86    | 32.24   |

预训练模型[下载](https://github.com/laisimiao/classification-cifar10-pytorch)。

# 运行

- `python fgsm_attack.py`，制作对抗样本
- `python eval.py`，验证目标模型在对抗样本攻击下的准确率