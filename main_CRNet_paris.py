# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

# 导入您的模型和框架
from networks.CRNet import CRNet
from framework import MyFrame
from loss import dice_bce_loss
from data4 import ImageFolder
import Constants11
import image_utils


def CE_Net_Train():
    NAME = 'CRNet' + Constants11.ROOT.split('/')[-1]
    print(NAME)

    # 初始化训练器
    solver = MyFrame(CRNet, dice_bce_loss, 2e-4)
    # 如果需要从某个检查点继续训练，可以取消下面这行的注释
    # solver.load('./weights/MRRNet8_contextparis.th')

    batchsize = torch.cuda.device_count() * Constants11.BATCHSIZE_PER_CARD
    dataset = ImageFolder(root_path=Constants11.ROOT, datasets='paris_road')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=16)

    # 创建日志文件
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()

    no_optim = 0
    total_epoch = Constants11.TOTAL_EPOCH
    train_epoch_best_loss = Constants11.INITAL_EPOCH_LOSS
    
    # 记录当前的学习率
    current_lr = solver.old_lr 

    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index += 1

        train_epoch_loss /= len(data_loader_iter)

        # 将信息打印并写入日志
        log_info = f'********\nepoch: {epoch}    time: {int(time() - tic)}\ntrain_loss: {train_epoch_loss}\nSHAPE: {Constants11.Image_size}\n********\n'
        print(log_info, end='')
        mylog.write(log_info)
        mylog.flush()

        # 检查是否找到了更优的模型
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '.th')
            print(f"New best model saved at epoch {epoch}.", file=mylog)
            print(f"New best model saved at epoch {epoch}.")

        # ------------------------ 关键修改：学习率变化检测与保存 ------------------------
        # 在每次更新学习率之前，检查当前学习率是否发生了变化
        if no_optim > Constants11.NUM_UPDATE_LR:
            # 获取即将被更新前的学习率（即当前有效学习率）
            old_lr_before_update = solver.old_lr 
            
            # 加载上一个最佳模型，准备进行学习率更新
            solver.load('./weights/' + NAME + '.th')
            
            # **在更新学习率之前，先保存一份“旧学习率”下的最终快照**
            # 这确保了即使没有更好的loss，我们也能保留这个学习率阶段的最终状态。
            lr_snapshot_path = './weights/' + NAME + f'_lr_{old_lr_before_update:.1e}_final_epoch{epoch}.th'
            solver.save(lr_snapshot_path)
            print(f"Learning rate snapshot saved at: {lr_snapshot_path}", file=mylog)
            print(f"Learning rate snapshot saved at: {lr_snapshot_path}")

            # 更新学习率
            solver.update_lr(5.0, factor=True, mylog=mylog) 
            # 注意：update_lr 可能会改变 solver.old_lr 的值
            
            # **再次保存，以记录新的学习率**
            # 这次保存是为了明确标记新学习率的开始。
            new_lr_after_update = solver.old_lr 
            lr_changed_path = './weights/' + NAME + f'_lr_{new_lr_after_update:.1e}_updated_at_epoch{epoch}.th'
            solver.save(lr_changed_path)
            print(f"Learning rate changed and weights saved at epoch {epoch}: {lr_changed_path}", file=mylog)
            print(f"Learning rate changed and weights saved at epoch {epoch}: {lr_changed_path}")
            
            # 重置计数器
            no_optim = 0 

        # -------------------------------------------------------------------------

        # 早停机制
        if no_optim > Constants11.NUM_EARLY_STOP:
            print(f'Early stop at epoch {epoch}', file=mylog)
            print(f'Early stop at epoch {epoch}')
            break

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    CE_Net_Train()