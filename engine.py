# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for adjusting keep rate and visualization -- Youwei Liang
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from helpers import adjust_keep_rate
from visualize_mask import get_real_idx, mask, save_img_batch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    writer=None,
                    set_training_mode=True,
                    args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    log_interval = 100
    it = epoch * len(data_loader)
    ITERS_PER_EPOCH = len(data_loader)

    base_rate = args.base_keep_rate

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        keep_rate = adjust_keep_rate(it, epoch, warmup_epochs=args.shrink_start_epoch,
                                         total_epochs=args.shrink_start_epoch + args.shrink_epochs,
                                         ITERS_PER_EPOCH=ITERS_PER_EPOCH, base_keep_rate=base_rate)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, keep_rate)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        # 这段代码检查了optimizer对象是否具有属性is_second_order，并且该属性的值为True。
        # is_second_order通常用于指示优化器是否支持二阶梯度计算，也就是计算Hessian矩阵或者类似的二阶导数信息。
        # 在深度学习中，常见的优化算法（如SGD、Adam等）是一阶优化算法，它们只计算一阶梯度（即一阶导数），用于更新模型的参数。
        # 但是，有些优化算法需要计算二阶梯度（如牛顿法），以便更准确地更新参数。
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if torch.distributed.get_rank() == 0 and it % log_interval == 0:
            writer.add_scalar('loss', loss_value, it)
            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], it)
            writer.add_scalar('keep_rate', keep_rate, it)
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, keep_rate


@torch.no_grad()
def evaluate(data_loader, model, device, keep_rate=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, keep_rate)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_acc(data_loader, model, device, keep_rate=None, tokens=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, keep_rate, tokens)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return metric_logger.acc1.global_avg

# 这段代码是一个用于可视化模型输出的函数。它接受一个数据加载器、一个模型、一个设备、一个输出目录、一个可视化数量和一个融合令牌作为输入，并输出一个字典，其中包含损失和准确率等指标的平均值。
# 在这个函数中，我们可以看到以下步骤：
# 1. 创建一个交叉熵损失函数。
# 2. 创建一个MetricLogger对象，用于记录和显示训练指标。
# 3. 将模型切换到评估模式。
# 4. 遍历数据加载器中的图像和标签。
# 5. 将图像和标签移动到指定的设备上。
# 6. 使用模型对图像进行前向传递，并计算输出和损失。
# 7. 计算输出的准确率。
# 8. 对图像进行反归一化处理。
# 9. 对输出进行可视化，并将结果保存到输出目录中。
# 10. 更新指标记录器中的指标。
# 11. 如果达到了指定的可视化数量，则停止遍历数据加载器。
# 12. 同步指标记录器中的指标。
# 13. 打印平均损失和准确率等指标的值。
# 14. 返回指标记录器中的指标的平均值。
# 总之，这个函数是一个用于可视化模型输出的工具，它使用PyTorch中的函数和类来实现图像处理、模型评估和指标记录等功能。它可以根据需要进行修改和定制，以适应不同的应用场景。
@torch.no_grad()
def visualize_mask(data_loader, model, device, output_dir, n_visualization, fuse_token, keep_rate=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)

    # switch to evaluation mode
    model.eval()

    ii = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        B = images.size(0)

        with torch.cuda.amp.autocast():
            # idx.shape = (Depth, B, N_)
            output, idx = model(images, keep_rate, get_idx=True)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # denormalize
        images = images * std + mean

        idxs = get_real_idx(idx, fuse_token)
        for jj, idx in enumerate(idxs):
            masked_img = mask(images, patch_size=16, idx=idx)
            save_img_batch(masked_img, output_dir, file_name='img_{}' + f'_l{jj}.jpg', start_idx=world_size * B * ii + rank * B)

        save_img_batch(images, output_dir, file_name='img_{}_a.jpg', start_idx=world_size * B * ii + rank * B)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.synchronize_between_processes()
        ii += 1
        if world_size * B * ii >= n_visualization:
            break

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
