# torch库
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 调整学习率 当参考的评价指标停止改进时,降低学习率,factor为每次下降的比例,训练过程中,当指标连续patience次数还没有改进时,降低学习率;

import numpy as np
from tqdm import tqdm

# 已经定义的model模块
from model.modules import MLP, CGBlock, MCGBlock, HistoryEncoder
from model.multipathpp import MultiPathPP
# dataloader部分
from model.data import get_dataloader, dict_to_cuda, normalize
# loss部分
from model.losses import pytorch_neg_multi_log_likelihood_batch, nll_with_covariances
# 工具类
from prerender.utils.utils import data_to_numpy, get_config

import subprocess
from matplotlib import pyplot as plt
import os
import glob
import sys
import random

# 设置随机种子
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 获取文件夹下最近生成的文件
def get_last_file(path):
    # 查找path路径下符合特定规则的文件路径
    list_of_files = glob.glob(f'{path}/*')
    if len(list_of_files) == 0:
        return None
    # os.path.getctime返回文件创建的时间
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# 
def get_git_revision_short_hash():
    # git rev-parse --short HEAD 获取最新的commit id 
    # decode的作用是将ascii编码的字符串转换成unicode编码 
    # strip()去掉前后空格
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

# 指定config 用来debug
# 获取命令行传入的配置
# config = get_config(sys.argv[1])
config = get_config("configs/final_RoP_Cov_Single.yaml")
# alias = sys.argv[1].split("/")[-1].split(".")[0]
alias = "configs/final_RoP_Cov_Single.yaml".split("/")[-1].split(".")[0]
# try:
#     models_path = os.path.join("../models", f"{alias}__{get_git_revision_short_hash()}")
#     os.mkdir(tb_path)
#     os.mkdir(models_path)
# except:
#     pass
models_path = "../models"

# 获取最新的权重
last_checkpoint = get_last_file(models_path)

# 获取dataloader
dataloader = get_dataloader(config["train"]["data_config"])
val_dataloader = get_dataloader(config["val"]["data_config"])

# 模型的配置
model = MultiPathPP(config["model"])
model.cuda()

# 优化器的初始学习率的设置
optimizer = Adam(model.parameters(), **config["train"]["optimizer"])

# 是否使用 ReduceLROnPlateau根据测试的指标调整学习率
if config["train"]["scheduler"]:
    # patience多少次指标没有变化 factor每次下将的比例
    scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)
num_steps = 0


# 如果last_checkpoint不空 加载模型
if last_checkpoint is not None:
    # 加载checkpoint的四个信息 model_state_dict optimizer_state_dict num_steps scheduler_state_dict
    model.load_state_dict(torch.load(last_checkpoint)["model_state_dict"])
    optimizer.load_state_dict(torch.load(last_checkpoint)["optimizer_state_dict"])
    num_steps = torch.load(last_checkpoint)["num_steps"]
    if config["train"]["scheduler"]:
        scheduler.load_state_dict(torch.load(last_checkpoint)["scheduler_state_dict"])
    print("LOADED ", last_checkpoint)


# 一个数据算一步
this_num_steps = 0
# 过滤出模型中可以反传梯度的参数
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# 计算模型的参数数量
params = sum([np.prod(p.size()) for p in model_parameters])
print("N PARAMS=", params)
# loss值
train_losses = []

# epoch大循环
for epoch in tqdm(range(config["train"]["n_epochs"])):
    # 每个批次小循环
    pbar = tqdm(dataloader)
    for data in pbar:
        # 进入训练模式 model.train()启用 batch normalization 和 dropout 
        model.train()
        # 初始化梯度为0
        optimizer.zero_grad()
        # 将数据归一化
        if config["train"]["normalize"]:
            data = normalize(data, config)
        # 将数据放入cuda
        dict_to_cuda(data)
        # 模型的输出结果
        probas, coordinates, covariance_matrices, loss_coeff = model(data, num_steps)
        # 判断其是不是有限值
        assert torch.isfinite(coordinates).all()
        assert torch.isfinite(probas).all()
        assert torch.isfinite(covariance_matrices).all()
        xy_future_gt = data["target/future/xy"]
        # 输出正则化
        if config["train"]["normalize_output"]:
            # assert not (config["train"]["normalize_output"] and config["train"]["trainable_cov"])
            # gt减去均值 做归一化
            xy_future_gt = (data["target/future/xy"] - torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()) / 10.
            # 判断其是不是有限值
            assert torch.isfinite(xy_future_gt).all()
        # 计算loss
        loss = nll_with_covariances(
            xy_future_gt, coordinates, probas, data["target/future/valid"].squeeze(-1),
            covariance_matrices) * loss_coeff
        # 加入loss数组中
        train_losses.append(loss.item())
        # 回传loss
        loss.backward()
        # 是否对梯度截取
        if "clip_grad_norm" in config["train"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad_norm"])
        # 更新梯度
        optimizer.step()
        # 是否做归一化
        if config["train"]["normalize_output"]:
            # detach() 从计算图上分离requires_grad为false 梯度不会反传
            _coordinates = coordinates.detach() * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
        else:
            _coordinates = coordinates.detach()
        # 每10个step打印loss信息
        if num_steps % 10 == 0:
            # 查看小数点后两位
            pbar.set_description(f"loss = {round(loss.item(), 2)}")
        # 每1000step保存模型
        if num_steps % 1000 == 0 and this_num_steps > 0:
            saving_data = {
                "num_steps": num_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if config["train"]["scheduler"]:
                saving_data["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(saving_data, os.path.join(models_path, f"last.pth"))
        # 训练到一个批次的一半时 eval一次
        if num_steps % (len(dataloader) // 2) == 0 and this_num_steps > 0:
            # 解除data变量对数据的引用
            del data
            # 释放unactivate的memory
            torch.cuda.empty_cache()
            # 评估模式 
            model.eval()
            # 计算指标
            with torch.no_grad():
                losses = []
                min_ades = []
                first_batch = True
                # 跑验证集的数据
                for data in tqdm(val_dataloader):
                    # 数据归一化
                    if config["train"]["normalize"]:
                        data = normalize(data, config)
                    # 放到cuda中
                    dict_to_cuda(data)
                    # 获得模型的输出
                    probas, coordinates, _, _ = model(data, num_steps)
                    # 映射到原来的值
                    if config["train"]["normalize_output"]:
                        coordinates = coordinates * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
                train_losses = []
            # 训练到中途的时候也保存模型
            saving_data = {
                "num_steps": num_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if config["train"]["scheduler"]:
                saving_data["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(saving_data, os.path.join(models_path, f"{num_steps}.pth"))
        # 一个data step + 1
        num_steps += 1
        # 本次训练的次数 + 1
        this_num_steps += 1
        # 如果大于最大的训练step次数 退出训练
        if "max_iterations" in config["train"] and num_steps > config["train"]["max_iterations"]:
            break

