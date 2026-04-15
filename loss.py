import numpy as np
from time import time
import Nmetrics
import matplotlib.pyplot as plt
import random
from dataset import MultiViewDataset
import yaml
from box import Box
import string
import cupy as cp
from cuml.cluster import KMeans as cuKMeans
import torch
from torch.utils.data import DataLoader
from models.MSGMVC import MSGMVC
import cupy as cp
import torch.nn as nn
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import Nmetrics
from util import enhance_distribution, student_distribution
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import math


def contrastive_loss_row(y_true: torch.Tensor,
                         y_pred: torch.Tensor,
                         tau: float = 1,
                         eps: float = 1e-12):
    """
    """
    # 防止概率为0

    P = torch.clamp(y_true, min=eps)
    Q = torch.clamp(y_pred, min=eps)
    P = P / P.sum(dim=1, keepdim=True)
    Q = Q / Q.sum(dim=1, keepdim=True)
    Q_log = torch.log(Q + eps)
    P_log = torch.log(P + eps)
    N = P.size(0)
    loss1 = -(P * Q).sum(dim=1).mean() / tau
    loss2 = -(Q * P).sum(dim=1).mean() / tau

    # symmetric
    return (loss1 + loss2) / 2


# def contrastive_loss_row(y_true: torch.Tensor,
#                            y_pred: torch.Tensor,
#                            tau: float = 1,
#                            eps: float = 1e-12):
#     """
#     """
#     # 防止概率为0

#     P = torch.clamp(y_true, min=eps)
#     Q = torch.clamp(y_pred, min=eps)
#     P = P / P.sum(dim=1, keepdim=True)
#     Q = Q / Q.sum(dim=1, keepdim=True)
#     Q_log = torch.log(Q + eps)
#     P_log = torch.log(P + eps)
#     N = P.size(0)
#     targets = torch.arange(N, device=P.device)

#     # view1 -> view2
#     logits1 = (P @ Q_log.t())/tau
#     # loss = F.nll_loss(logits, targets, reduction="mean") 
#     loss1 = F.cross_entropy(logits1, targets, reduction="mean")

#     # view2 -> view1YTF10
#     logits2 = (Q @ P_log.t()) / tau
#     loss2 = F.cross_entropy(logits2, targets, reduction="mean")

#     # symmetric
#     return (loss1 + loss2) / 2


def contrastive_loss_column(y_true: torch.Tensor,
                            y_pred: torch.Tensor,
                            alpha: float = 1,
                            eps: float = 1e-12,
                            beta: float = 0.1):
    """
    """
    # 防止概率为0
    P = y_true.t()
    Q = y_pred.t()
    P = torch.clamp(P, min=eps)
    Q = torch.clamp(Q, min=eps)
    P = P / P.sum(dim=1, keepdim=True)
    Q = Q / Q.sum(dim=1, keepdim=True)
    Q_log = torch.log(Q + eps)
    P_log = torch.log(P + eps)
    N = P.size(0)
    targets = torch.arange(N, device=P.device)

    # view1 -> view2
    logits1 = (P @ Q_log.t())
    loss1 = F.cross_entropy(logits1, targets, reduction="mean")

    # view2 -> view1
    logits2 = (Q @ P_log.t())
    loss2 = F.cross_entropy(logits2, targets, reduction="mean")

    # 计算熵正则项（基于论文中的公式8）
    # 计算每个视图的聚类分布
    q_P = torch.mean(P, dim=1)  # 视图1的聚类分布
    q_Q = torch.mean(Q, dim=1)  # 视图2的聚类分布

    # 计算熵损失 (L_a)
    entropy_loss_P = torch.sum(q_P * torch.log(q_P + eps))
    entropy_loss_Q = torch.sum(q_Q * torch.log(q_Q + eps))
    entropy_loss = entropy_loss_P + entropy_loss_Q

    # 对称对比损失 + 熵正则项
    contrastive_loss = (loss1 + loss2) / 2
    total_loss = alpha * contrastive_loss + beta * entropy_loss

    return total_loss


def mimvc_loss(
        x,
        z,
        features,
        reconstructed_x,
        reconstructed_z,
        cluster_unique_assign,
        cluster_sp_assign,
        args
):
    eps = 1e-15
    # cluster_unique_assign_log = (cluster_unique_assign + eps).log()
    mse_loss = nn.MSELoss()

    losses_sic = [mse_loss(x[v], reconstructed_x[v]) for v in range(len(x))]
    # losses_rci = [
    #     0.5 * mse_loss(features[v].detach(), reconstructed_z[v]) + 0.5 *  mse_loss(reconstructed_z[v].detach(), features[v]) for v in range(len(x))
    # ]

    alpha = args.alpha

    losses_cca_row = [contrastive_loss_row(cluster_unique_assign, cluster_sp_assign[v]) for v in range(len(x))]

    losses_cca_column = [
        contrastive_loss_column(cluster_unique_assign, cluster_sp_assign[v], alpha, 1e-12, args.beta)
        for v in range(len(x))
    ]

    # cluster_unique_assign shape: [B, K]，与 trainer 中 cluster_unique_assign[idx] 对齐
    # cluster_sp_assign[v] shape: [B, K]，来自 model(x)

    # === 组合各项损失 ===
    w_ae = args.ae_weight
    # w_dg = args.dg_weight
    w_col = args.contrastive_weight_column
    w_row = args.contrastive_weight_row

    loss = []
    for v in range(len(x)):
        loss_v = (
                w_ae * losses_sic[v] +
                # w_dg * losses_rci[v] +
                w_col * losses_cca_column[v] +
                w_row * losses_cca_row[v]
        )
        loss.append(loss_v)
    return loss
