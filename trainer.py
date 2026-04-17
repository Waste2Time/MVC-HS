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
from sklearn.cluster import KMeans as skKMeans
import torch
from torch.utils.data import DataLoader
from models.MSGMVC import MSGMVC
import cupy as cp
import torch.nn as nn
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import Nmetrics
from util import enhance_distribution, student_distribution, plot_tsne
from sklearn.preprocessing import normalize
from sklearn.metrics import calinski_harabasz_score, silhouette_score, silhouette_samples
import torch.nn.functional as F
import os


def minmax_scale_tensor(x: torch.Tensor, eps=1e-12):
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)


class Trainer():
    def __init__(
        self,
        pre_data_loader,
        data_loader,
        model,
        pre_opt,
        opt,
        scheduler,
        loss_fn,
        device,
        args,
    ):
        self.pre_data_loader = pre_data_loader
        self.data_loader = data_loader
        self.dataset = data_loader.dataset
        self.model = model
        self.pre_opt = pre_opt
        self.opt = opt
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.args = args
        self.n_clusters = self.dataset.get_num_clusters()
        self.dims = self.dataset.get_views()
        self.views = len(self.dims)
        self.seed = args.seed

    def pre_train(self):
        self.model.train()
        pre_train_epoch = self.args.pretrain_epochs
        crit = nn.MSELoss()
        for i in range(pre_train_epoch):
            t_epoch0 = time()
            loss_sum = [0.0] * self.views
            print(f'epoch: {i + 1}')
            for x, _, _ in self.pre_data_loader:
                x = [xi.to(self.device) for xi in x]
                self.pre_opt.zero_grad()
                reconstructed_x = self.model(x, is_pretrain=True)
                losses = [crit(x[v], reconstructed_x[v]) for v in range(len(x))]
                loss = sum(losses)
                loss.backward()
                self.pre_opt.step()
                for view in range(self.views):
                    loss_sum[view] += losses[view].item()

            loss_total = sum(loss_sum) / self.views
            loss_sum = [loss_total] + loss_sum
            for view in range(len(loss_sum)):
                loss_sum[view] = loss_sum[view] / len(self.pre_data_loader)
            t_epoch = time() - t_epoch0
            print(f'loss: {loss_sum} | time_epoch={t_epoch:.2f}s')
            print()
        self.model.save_pretrain_model()

    def extract_features(self):
        '''
        提取整体数据的特征, 用于聚类
        '''
        features_list = [[] for i in range(self.views)]
        data_loader = DataLoader(self.dataset, batch_size=1024, shuffle=False, num_workers=0)
        self.model.eval()
        with torch.no_grad():
            for x, _, _ in data_loader:
                for i, xi in enumerate(x):
                    z = self.model.encoders[i](xi.to(self.device))
                    features_list[i].append(z.detach())

        features = [torch.cat(f, dim=0) for f in features_list]
        self.model.train()
        return features

    def view_sp_cluster(self):
        """
        每个视图各自聚类
        """
        y_preds = []
        centers = []
        features = self.extract_features()
        if self.args.normalize == 1:
            features = [F.normalize(f, p=2, dim=1) for f in features]

        # sklearn k-means on cpu
        for view in range(self.views):
            kmeans = skKMeans(n_clusters=self.n_clusters, init='k-means++', n_init=100, random_state=self.seed)
            t = features[view].contiguous().float()
            x = t.detach().cpu().numpy()
            y_pred_np = kmeans.fit_predict(x)
            y_pred = torch.from_numpy(y_pred_np).to(self.device)
            y_preds.append(y_pred)
            cc = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)  # (K, D)
            centers.append(cc)

        return y_preds, centers, features

    def unique_cluster(self):
        """
        多视图融合聚类 形成一个公共的聚类中心
        """
        y_pred = []
        features = self.extract_features()
        # 融合
        if self.args.normalize == 1:
            features = [F.normalize(f, p=2, dim=1) for f in features]
        z = sum(features) / len(features)

        # NOTE: keep your original logic unchanged (even if it looks odd),
        # since you requested only logging changes.
        kmeans = cuKMeans(n_clusters=self.n_clusters, init='k-means++', n_init=100, random_state=self.seed)
        t = z.contiguous().float()
        t_np = t.detach().cpu().numpy()
        y_pred_np = (kmeans.fit_predict(t_np))
        y_pred = [torch.from_numpy(y_pred_np).to(self.device)]
        cc = kmeans.cluster_centers_
        center = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)

        return y_pred, center, z

    def init_sp_cluster_centers(self, centers):
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                self.model.cluster_layers[view].clusters.copy_(centers[view])
                self.model.cluster_layers[view].clusters.grad = None

    def evaluate_sp_cluster(self, y_pred_k, centers, features):
        '''
        评估各个视角的聚类结果
        '''
        y_pred_k = [y_pred_k[i].cpu().numpy() for i in range(self.views)]
        y_pred_list = [[] for i in range(self.views)]
        y_true = self.dataset.y.cpu().numpy()
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                for i in range(0, len(self.dataset), 1024):
                    features_batch = features[view][i:i + 1024]
                    q = self.model.cluster_layers[view](features_batch)
                    q = enhance_distribution(q)
                    y_pred_list[view].append(torch.argmax(q, dim=1))

        y_pred = [torch.cat(f, dim=0).cpu().numpy() for f in y_pred_list]
        for view in range(self.views):
            acc = Nmetrics.acc(y_true, y_pred[view])
            nmi = Nmetrics.nmi(y_true, y_pred[view])
            ari = Nmetrics.ari(y_true, y_pred[view])
            pur = Nmetrics.pur(y_true, y_pred[view])
            print(f'View: {view + 1}, acc: {acc:.5f}, nmi: {nmi:.5f}, ari: {ari:.5f}, pur: {pur:.5f}')
        self.model.train()

    def evaluate_unique_cluster_views(self, y_pred_k, centers, features):
        '''
        评估各个视角的聚类结果
        '''
        y_pred_list = []
        q_list = [[] for i in range(self.views)]
        y_true = self.dataset.y.cpu().numpy()
        self.model.eval()
        with torch.no_grad():
            for view in range(self.views):
                for i in range(0, len(self.dataset), 1024):
                    features_batch = features[view][i:i + 1024]
                    q = self.model.cluster_layers[view](features_batch)
                    q = enhance_distribution(q)
                    q_list[view].append(q)
        q_stacked = [torch.cat(f, dim=0) for f in q_list]
        q_stacked = torch.stack(q_stacked, dim=0)  # [v, N, K]
        avg_q = q_stacked.mean(dim=0)
        y_pred = torch.argmax(avg_q, dim=1)
        y_pred = y_pred.cpu().numpy()
        acc = Nmetrics.acc(y_true, y_pred)
        nmi = Nmetrics.nmi(y_true, y_pred)
        ari = Nmetrics.ari(y_true, y_pred)
        pur = Nmetrics.pur(y_true, y_pred)
        print(f'Unique: acc: {acc:.5f}, nmi: {nmi:.5f}, ari: {ari:.5f}, pur: {pur:.5f}')
        self.model.train()
        indices = {
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'pur': pur
        }
        return indices

    def evaluate_unique_cluster(self, y_pred_k, center, features):
        '''
        评估融合结果的聚类结果
        '''
        y_pred_k = y_pred_k[0].cpu().numpy()
        y_pred_list = []
        y_true = self.dataset.y.cpu().numpy()
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(self.dataset), 1024):
                features_batch = features[i:i + 1024]
                q = student_distribution(features_batch, self.model.unique_center)
                q = enhance_distribution(q)
                y_pred_list.append(torch.argmax(q, dim=1))
        y_pred = torch.cat(y_pred_list, dim=0).cpu().numpy()
        acc = Nmetrics.acc(y_true, y_pred)
        nmi = Nmetrics.nmi(y_true, y_pred)
        ari = Nmetrics.ari(y_true, y_pred)
        pur = Nmetrics.pur(y_true, y_pred)
        print(f'Unique: acc: {acc:.5f}, nmi: {nmi:.5f}, ari: {ari:.5f}, pur: {pur:.5f}')
        self.model.train()
        indices = {
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'pur': pur,
        }
        return indices

    # ----------------------------- Hard-sample mining (MVCHSG) -----------------------------
    @staticmethod
    def _set_requires_grad(module, flag: bool):
        """Safely toggle requires_grad for all parameters in a (sub)module."""
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    @staticmethod
    def _safe_l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

    def _get_hard_hparams(self):
        """Fetch hard-mining hyperparameters with backward-compatible defaults."""
        hp_maxiter = int(getattr(self.args, "hp_maxiter", 5) or 5)
        hn_maxiter = int(getattr(self.args, "hn_maxiter", 5) or 5)

        hp_alpha = float(getattr(self.args, "hp_alpha", 0.5) or 0.5)
        hn_beta = float(getattr(self.args, "hn_beta", 0.5) or 0.5)

        hard_tau = float(getattr(self.args, "hard_tau", 0.0) or 0.0)
        hard_weight = float(getattr(self.args, "hard_weight", 1.0) or 1.0)
        return hp_maxiter, hn_maxiter, hp_alpha, hn_beta, hard_tau, hard_weight

    def _alpha_to_shrink(self, a: float) -> float:
        """Convert a user-provided scaling factor to a *shrink* multiplier in (0,1]."""
        if a is None:
            return 0.5
        a = float(a)
        if a <= 0:
            return 0.5
        if a < 1:
            return a
        return 1.0 / a

    def _generate_hard_positive(
        self,
        zi: torch.Tensor,
        ci: torch.Tensor,
        view_idx: int,
        maxiter: int,
        hp_alpha: float,
        anchor_label: int,
    ) -> torch.Tensor:
        cluster_layer = self.model.cluster_layers[view_idx]
        ell = ci - zi
        d = 1.0
        hp = ci
        shrink = self._alpha_to_shrink(hp_alpha)

        for _ in range(int(maxiter)):
            temp = hp + d * ell
            with torch.no_grad():
                q_temp = cluster_layer(temp.unsqueeze(0))  # [1, K]
                temp_label = int(torch.argmax(q_temp, dim=1).item())
            if temp_label == int(anchor_label):
                hp = temp
            else:
                d = d * shrink
        return hp

    def _generate_hard_negative(
        self,
        zi: torch.Tensor,
        cj: torch.Tensor,
        view_idx: int,
        maxiter: int,
        hn_beta: float,
        anchor_label: int,
    ) -> torch.Tensor:
        cluster_layer = self.model.cluster_layers[view_idx]
        ell = zi - cj
        d = 1.0
        hn = cj
        shrink = self._alpha_to_shrink(hn_beta)

        for _ in range(int(maxiter)):
            temp = hn + d * ell
            with torch.no_grad():
                q_temp = cluster_layer(temp.unsqueeze(0))  # [1, K]
                temp_label = int(torch.argmax(q_temp, dim=1).item())
            if temp_label != int(anchor_label):
                hn = temp
            else:
                d = d * shrink
        return hn

    def _build_hard_pos_neg_batch(self, feats: list):
        hp_maxiter, hn_maxiter, hp_alpha, hn_beta, _, _ = self._get_hard_hparams()
        V = len(feats)
        B = feats[0].shape[0]

        clusters_by_view = [self.model.cluster_layers[v].clusters.detach() for v in range(V)]  # [K,D]

        anchor_labels = []
        for v in range(V):
            with torch.no_grad():
                q = self.model.cluster_layers[v](feats[v].detach())  # [B,K]
                anchor_labels.append(torch.argmax(q, dim=1))  # [B]

        hp_by_view = []
        hn_by_view = []
        for v in range(V):
            clusters = clusters_by_view[v]  # [K,D]
            labels_v = anchor_labels[v]  # [B]
            hp_list = []
            hn_list = []

            for i in range(B):
                zi = feats[v][i].detach()
                yi = int(labels_v[i].item())
                ci = clusters[yi]

                dist = torch.sum((clusters - zi.unsqueeze(0)) ** 2, dim=1)
                dist[yi] = float("inf")
                j = int(torch.argmin(dist).item())
                cj = clusters[j]

                hp_i = self._generate_hard_positive(
                    zi=zi, ci=ci, view_idx=v,
                    maxiter=hp_maxiter, hp_alpha=hp_alpha,
                    anchor_label=yi,
                )
                hn_i = self._generate_hard_negative(
                    zi=zi, cj=cj, view_idx=v,
                    maxiter=hn_maxiter, hn_beta=hn_beta,
                    anchor_label=yi,
                )

                hp_i = self._safe_l2_normalize(hp_i.unsqueeze(0), dim=1)[0]
                hn_i = self._safe_l2_normalize(hn_i.unsqueeze(0), dim=1)[0]

                hp_list.append(hp_i.unsqueeze(0))
                hn_list.append(hn_i.unsqueeze(0))

            hp_by_view.append(torch.cat(hp_list, dim=0))  # [B,D]
            hn_by_view.append(torch.cat(hn_list, dim=0))  # [B,D]

        hard_pos = {}
        hard_neg = {}
        for v in range(V):
            for m in range(V):
                if m == v:
                    continue
                hard_pos[(v, m)] = hp_by_view[m]
                hard_neg[(v, m)] = hn_by_view[m]
        return hard_pos, hard_neg

    def _hard_contrastive_loss(self, feats: list, hard_pos: dict, hard_neg: dict, tau: float) -> torch.Tensor:
        if tau is None or abs(float(tau)) < 1e-12:
            return torch.tensor(0.0, device=feats[0].device)
        tau = float(tau)
        V = len(feats)
        total = torch.tensor(0.0, device=feats[0].device)

        for v in range(V):
            anchor = feats[v]
            for m in range(V):
                if m == v:
                    continue
                pos = hard_pos[(v, m)]
                neg = hard_neg[(v, m)]
                s_pos = torch.sum(anchor * pos, dim=1) / tau
                s_neg = torch.sum(anchor * neg, dim=1) / tau
                total = total + torch.mean(F.softplus(s_neg - s_pos))
        return total

    def _build_margin_hard_pos_neg_batch(self, feats: list):
        """Margin-hard mining: select low-margin samples, then mine hard pos/neg across views by cosine similarity."""
        V = len(feats)
        B = feats[0].shape[0]
        hard_ratio = float(getattr(self.args, "hard_ratio", 0.3) or 0.3)
        hard_ratio = min(max(hard_ratio, 0.05), 1.0)
        k_hard = max(1, int(B * hard_ratio))

        with torch.no_grad():
            labels_by_view = []
            hard_idx_by_view = []
            for v in range(V):
                q = self.model.cluster_layers[v](feats[v].detach())
                top2 = torch.topk(q, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                labels = torch.argmax(q, dim=1)
                hard_idx = torch.argsort(margin)[:k_hard]
                labels_by_view.append(labels)
                hard_idx_by_view.append(hard_idx)

        emb = [self._safe_l2_normalize(f, dim=1).detach() for f in feats]
        hard_pos, hard_neg = {}, {}

        for v in range(V):
            anchor = emb[v]  # [B, D]
            labels_v = labels_by_view[v]
            for m in range(V):
                if m == v:
                    continue
                cand = emb[m]
                labels_m = labels_by_view[m]
                hard_m = hard_idx_by_view[m]

                pos_list, neg_list = [], []
                for i in range(B):
                    yi = labels_v[i]
                    sim = torch.matmul(cand, anchor[i])  # [B]

                    pos_mask_h = (labels_m == yi)
                    pos_mask_h = pos_mask_h & torch.zeros_like(pos_mask_h, dtype=torch.bool).scatter_(0, hard_m, True)
                    if not torch.any(pos_mask_h):
                        pos_mask_h = (labels_m == yi)
                    if not torch.any(pos_mask_h):
                        pos_idx = i
                    else:
                        pos_idx = torch.argmin(sim.masked_fill(~pos_mask_h, float('inf')))

                    neg_mask_h = (labels_m != yi)
                    neg_mask_h = neg_mask_h & torch.zeros_like(neg_mask_h, dtype=torch.bool).scatter_(0, hard_m, True)
                    if not torch.any(neg_mask_h):
                        neg_mask_h = (labels_m != yi)
                    if not torch.any(neg_mask_h):
                        neg_idx = i
                    else:
                        neg_idx = torch.argmax(sim.masked_fill(~neg_mask_h, float('-inf')))

                    pos_list.append(cand[pos_idx].unsqueeze(0))
                    neg_list.append(cand[neg_idx].unsqueeze(0))

                hard_pos[(v, m)] = torch.cat(pos_list, dim=0)
                hard_neg[(v, m)] = torch.cat(neg_list, dim=0)

        return hard_pos, hard_neg

    def _cluster_layer_update_one_epoch(self):
        eps = 1e-12
        self.model.train()

        self._set_requires_grad(self.model, False)
        self._set_requires_grad(self.model.cluster_layers, True)

        for x, _, _ in self.data_loader:
            x = [xi.to(self.device) for xi in x]
            feats = [self.model.encoders[v](x[v]).detach() for v in range(self.views)]
            if getattr(self.args, "normalize", 0) == 1:
                feats = [F.normalize(f, p=2, dim=1) for f in feats]

            q_list = [self.model.cluster_layers[v](feats[v]) for v in range(self.views)]
            q_list = [torch.clamp(q, min=eps) for q in q_list]
            p_list = [enhance_distribution(q).detach() for q in q_list]

            loss = 0.0
            for v in range(self.views):
                loss = loss + F.kl_div(torch.log(q_list[v]), p_list[v], reduction="batchmean")
            loss = loss / self.views

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        self._set_requires_grad(self.model, True)


    def _train_hard_mining(self):
        hp_maxiter, hn_maxiter, hp_alpha, hn_beta, hard_tau, hard_weight = self._get_hard_hparams()

        self.model.eval()
        with torch.no_grad():
            # 初始化
            y_pred_sp, centers_sp, features_sp = self.view_sp_cluster()
            y_pred_uq, centers_uq, features_uq = self.unique_cluster()

            self.model.unique_center = centers_uq
            cluster_unique_assign = student_distribution(features_uq, self.model.unique_center)
            cluster_unique_assign = enhance_distribution(cluster_unique_assign)

            self.init_sp_cluster_centers(centers_sp)
            self.evaluate_sp_cluster(y_pred_sp, centers_sp, features_sp)
            new_indices = self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
            is_updated = self.model.update_best_indice(new_indices)
            print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' %
                  (self.model.best_indice['acc'], self.model.best_indice['nmi'],
                   self.model.best_indice['ari'], self.model.best_indice['pur']))
            if is_updated is True and self.args.save is True:
                print('saving model to:', self.args.weights)
                self.model.save_model()
            print()

        for epoch in range(self.args.epochs):
            t_epoch0 = time()
            print(f'epoch: {epoch + 1}')
            do_hard_epoch = ((epoch + 1) % self.args.update_interval == 0)

            t_cluster0 = time()
            self._cluster_layer_update_one_epoch()
            t_cluster = time() - t_cluster0

            self.model.train()
            self._set_requires_grad(self.model, True)
            self._set_requires_grad(self.model.cluster_layers, False)
            for p in self.model.cluster_layers.parameters():
                p.requires_grad = False

            # ====== logging accumulators (ONLY logging; training unchanged) ======
            losses_sum = [0.0] * self.views
            hard_sum = 0.0
            base_sum = 0.0
            total_sum = 0.0

            t_train0 = time()
            for x, y, idx in self.data_loader:
                x = [xi.to(self.device) for xi in x]
                self.opt.zero_grad()

                features, reconstructed_x, cluster_sp_assign = self.model(x, is_pretrain=False)
                if getattr(self.args, "normalize", 0) == 1:
                    features = [F.normalize(f, p=2, dim=1) for f in features]

                z = torch.stack(features, dim=0).mean(0)
                reconstructed_z = [self.model.generator[v](z) for v in range(self.views)]

                losses = self.loss_fn(
                    x=x,
                    z=z,
                    features=features,
                    reconstructed_x=reconstructed_x,
                    reconstructed_z=reconstructed_z,
                    cluster_unique_assign=cluster_unique_assign[idx],
                    cluster_sp_assign=cluster_sp_assign,
                    args=self.args,
                )
                base_loss = sum(losses) / self.views

                hard_loss = torch.tensor(0.0, device=self.device)
                if hard_tau > 0 and do_hard_epoch:
                    hard_pos, hard_neg = self._build_margin_hard_pos_neg_batch(features)
                    emb = [self._safe_l2_normalize(f, dim=1) for f in features]
                    hard_loss = self._hard_contrastive_loss(emb, hard_pos, hard_neg, tau=hard_tau)
                    hard_sum += float(hard_loss.detach().item())

                loss = base_loss + hard_weight * hard_loss
                loss.backward()
                self.opt.step()

                # ====== logging only ======
                base_sum += float(base_loss.detach().item())
                total_sum += float(loss.detach().item())
                for v in range(self.views):
                    losses_sum[v] += float(losses[v].detach().item())

            t_train = time() - t_train0
            num_batches = max(1, len(self.data_loader))
            epoch_loss_total = total_sum / num_batches
            epoch_loss_base = base_sum / num_batches
            epoch_hard_cl = (hard_sum / num_batches) if (hard_tau > 0 and do_hard_epoch) else 0.0

            # --------- NEW structured log line (easy to parse) ---------
            # NOTE: hard_weight is intentionally NOT logged (per your request).
            t_epoch = time() - t_epoch0
            print(
                f"[TRAIN][hard] epoch={epoch + 1} "
                f"loss_total={epoch_loss_total:.6f} "
                f"loss_base={epoch_loss_base:.6f} "
                f"hard_cl={epoch_hard_cl:.6f} "
                f"time_cluster={t_cluster:.2f}s "
                f"time_train={t_train:.2f}s "
                f"time_epoch={t_epoch:.2f}s"
            )
            # ----------------------------------------------------------

            if (epoch + 1) % self.args.update_interval == 0:
                print('更新聚类质心')
                self.model.eval()
                t_update0 = time()
                with torch.no_grad():
                    t_uc0 = time()
                    y_pred_uq, centers_uq, features_uq = self.unique_cluster()
                    t_uc = time() - t_uc0
                    self.model.unique_center = centers_uq
                    cluster_unique_assign = student_distribution(features_uq, self.model.unique_center)
                    cluster_unique_assign = enhance_distribution(cluster_unique_assign)
                    t_ev0 = time()
                    new_indices = self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
                    t_ev = time() - t_ev0
                    is_updated = self.model.update_best_indice(new_indices)
                    print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' %
                          (self.model.best_indice['acc'], self.model.best_indice['nmi'],
                           self.model.best_indice['ari'], self.model.best_indice['pur']))
                    if is_updated is True and self.args.save is True:
                        print('saving model to:', self.args.weights)
                        self.model.save_model()

                t_update_total = time() - t_update0
                print(f"[TIME] epoch={epoch+1} stage=update_cluster total={t_update_total:.2f}s unique_cluster={t_uc:.2f}s eval_unique={t_ev:.2f}s")

            elif (epoch + 1) % self.args.cluster_interval == 0 and epoch != 0:
                self.model.eval()
                # timing (logging only)
                t_eval0 = time()
                with torch.no_grad():
                    t_uc0 = time()
                    y_pred_uq, centers_uq, features_uq = self.unique_cluster()
                    t_uc = time() - t_uc0
                    t_sp0 = time()
                    self.evaluate_sp_cluster(y_pred_sp, centers_sp, features_sp)
                    t_sp = time() - t_sp0
                    is_updated = self.model.update_best_indice(new_indices)
                    print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' %
                          (self.model.best_indice['acc'], self.model.best_indice['nmi'],
                           self.model.best_indice['ari'], self.model.best_indice['pur']))
                    if is_updated is True and self.args.save is True:
                        print('saving model to:', self.args.weights)
                        self.model.save_model()
                t_eval_total = time() - t_eval0
                print(f"[TIME] epoch={epoch+1} stage=cluster_eval total={t_eval_total:.2f}s unique_cluster={t_uc:.2f}s eval_sp={t_sp:.2f}s")
            print()

        self._set_requires_grad(self.model, True)
    def train(self):
        _, _, _, _, hard_tau, _ = self._get_hard_hparams()
        if hard_tau <= 0:
            return self._train_baseline()
        return self._train_hard_mining()

    def _train_baseline(self):
        self.model.eval()
        with torch.no_grad():
            y_pred_sp, centers_sp, features_sp = self.view_sp_cluster()
            y_pred_uq, centers_uq, features_uq = self.unique_cluster()
            self.model.unique_center = centers_uq
            cluster_unique_assign = student_distribution(features_uq, self.model.unique_center)
            cluster_unique_assign = enhance_distribution(cluster_unique_assign)
            self.init_sp_cluster_centers(centers_sp)
            self.evaluate_sp_cluster(y_pred_sp, centers_sp, features_sp)
            new_indices = self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
            is_updated = self.model.update_best_indice(new_indices)
            print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' %
                  (self.model.best_indice['acc'], self.model.best_indice['nmi'],
                   self.model.best_indice['ari'], self.model.best_indice['pur']))
            if is_updated is True and self.args.save is True:
                print('saving model to:', self.args.weights)
                self.model.save_model()

        for i in range(self.args.epochs):
            t_epoch0 = time()
            print(f'epoch: {i + 1}')
            self.model.train()

            losses_sum = [0.0] * self.views
            total_sum = 0.0  # logging only: true optimized objective for baseline (= loss)

            t_train0 = time()
            for x, y, idx in self.data_loader:
                x = [xi.to(self.device) for xi in x]
                self.opt.zero_grad()
                features, reconstructed_x, cluster_sp_assign = self.model(x, is_pretrain=False)
                if self.args.normalize == 1:
                    features = [F.normalize(f, p=2, dim=1) for f in features]
                z = torch.stack(features, dim=0).mean(0)
                reconstructed_z = [self.model.generator[view](z) for view in range(self.views)]

                losses = self.loss_fn(
                    x=x,
                    z=z,
                    features=features,
                    reconstructed_x=reconstructed_x,
                    reconstructed_z=reconstructed_z,
                    cluster_unique_assign=cluster_unique_assign[idx],
                    cluster_sp_assign=cluster_sp_assign,
                    args=self.args
                )
                loss = sum(losses) / self.views
                for view in range(self.views):
                    losses_sum[view] += float(losses[view].detach().item())

                loss.backward()
                self.opt.step()

                # logging only
                total_sum += float(loss.detach().item())

            t_train = time() - t_train0
            num_batches = max(1, len(self.data_loader))
            epoch_loss_total = total_sum / num_batches

            # --------- NEW structured log line (easy to parse) ---------
            t_epoch = time() - t_epoch0
            print(
                f"[TRAIN][baseline] epoch={i + 1} "
                f"loss_total={epoch_loss_total:.6f} "
                f"time_train={t_train:.2f}s "
                f"time_epoch={t_epoch:.2f}s"
            )
            # ----------------------------------------------------------

            if (i + 1) % self.args.update_interval == 0:
                print('更新聚类质心')
                self.model.eval()
                t_update0 = time()
                with torch.no_grad():
                    t_uc0 = time()
                    y_pred_uq, centers_uq, features_uq = self.unique_cluster()
                    t_uc = time() - t_uc0
                    self.model.unique_center = centers_uq
                    cluster_unique_assign = student_distribution(features_uq, self.model.unique_center)
                    cluster_unique_assign = enhance_distribution(cluster_unique_assign)
                    t_ev0 = time()
                    new_indices = self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
                    t_ev = time() - t_ev0
                    is_updated = self.model.update_best_indice(new_indices)
                    print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' %
                          (self.model.best_indice['acc'], self.model.best_indice['nmi'],
                           self.model.best_indice['ari'], self.model.best_indice['pur']))
                    if is_updated is True and self.args.save is True:
                        print('saving model to:', self.args.weights)
                        self.model.save_model()

                t_update_total = time() - t_update0
                print(f"[TIME] epoch={i+1} stage=update_cluster total={t_update_total:.2f}s unique_cluster={t_uc:.2f}s eval_unique={t_ev:.2f}s")

            elif (i + 1) % self.args.cluster_interval == 0 and i != 0:
                self.model.eval()
                # timing (logging only)
                t_eval0 = time()
                with torch.no_grad():
                    t_uc0 = time()
                    y_pred_uq, centers_uq, features_uq = self.unique_cluster()
                    t_uc = time() - t_uc0
                    t_sp0 = time()
                    self.evaluate_sp_cluster(y_pred_sp, centers_sp, features_sp)
                    t_sp = time() - t_sp0
                    is_updated = self.model.update_best_indice(new_indices)
                    print('Best Indicators: ACC=%.5f, NMI=%.5f, ARI=%.5f, PUR = %.5f' %
                          (self.model.best_indice['acc'], self.model.best_indice['nmi'],
                           self.model.best_indice['ari'], self.model.best_indice['pur']))
                    if is_updated is True and self.args.save is True:
                        print('saving model to:', self.args.weights)
                        self.model.save_model()

                t_eval_total = time() - t_eval0
                print(f"[TIME] epoch={i+1} stage=cluster_eval total={t_eval_total:.2f}s unique_cluster={t_uc:.2f}s eval_sp={t_sp:.2f}s")

            print()

    def test(self):
        self.model = self.model.load_model(self.device)
        self.model.eval()
        y_pred_uq, centers_uq, features_uq = self.unique_cluster()
        self.model.unique_center = centers_uq
        cluster_unique_assign = student_distribution(features_uq, self.model.unique_center)
        cluster_unique_assign = enhance_distribution(cluster_unique_assign)
        self.evaluate_unique_cluster(y_pred_uq, centers_uq, features_uq)
        fig_dir = os.path.join(self.args.save_dir, self.args.dataset + '.pdf')
        plot_tsne(features_uq.cpu().numpy(), y_pred_uq[0].cpu().numpy(), fig_dir, self.args.seed)
        is_pause = 1
