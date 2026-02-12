# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from utils import kmeans, dist_utils


def sinkhorn(cost, n_iters: int = 5, epsilon: float = 10., is_distributed: bool = True):
    """
    Sinkhorn algorithm.
    Args:
        cost (Tensor): shape with (B, K)
    """
    Q = torch.exp(- cost * epsilon).t()  # (K, B)
    if is_distributed:
        B = Q.size(1) * dist.get_world_size()
    else:
        B = Q.size(1)
    K = Q.size(0)

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if is_distributed:
        dist.all_reduce(sum_Q)
    Q /= (sum_Q + 1e-8)

    for _ in range(n_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if is_distributed:
            dist.all_reduce(sum_of_rows)
        Q /= (sum_of_rows + 1e-8)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()  # (B, K)


class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5,
                 distance_type='l2'):
        # 码本大小：创建 n_embed + 1 个嵌入，其中最后一个作为 padding index
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(True) for p in self.parameters()]

            # padding index is not updated by EMA
            # 通过 cluster_size_ema 跟踪每个码本的使用频率，当使用次数为零时重置
            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

        self.initialize_weights()
        self.distance_type = distance_type

        # 0606退火算法
        # self.prob_decay = 0.00001
        # self.eps = 1e-5
        # self.temperature = 1.

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight)  # , a=1.0

    @torch.no_grad()
    def compute_distances(self, inputs):
        # l2 distance
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        if self.distance_type == 'l2':
            inputs_flat = inputs.reshape(-1, embed_dim)

            inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
            codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
            distances = torch.addmm(
                inputs_norm_sq + codebook_t_norm_sq,
                inputs_flat,
                codebook_t,
                alpha=-2.0,
            )
            distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, n_embed or n_embed+1]
        elif self.distance_type == 'cosine':
            # cosine distance
            codebook_vectors = self.weight[:-1, :]
            codebook_vectors_norm = F.normalize(codebook_vectors, p=2, dim=1)
            inputs_norm = F.normalize(inputs, p=2, dim=1)
            distances = -torch.matmul(inputs_norm, codebook_vectors_norm.t())
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

        return distances

    # def compute_distances_gard(self, inputs):
    #     # l2 distance
    #     codebook_t = self.weight[:-1, :].t().detach()

    #     (embed_dim, _) = codebook_t.shape
    #     inputs_shape = inputs.shape
    #     assert inputs_shape[-1] == embed_dim

    #     inputs_flat = inputs.reshape(-1, embed_dim)

    #     inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
    #     codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
    #     distances = torch.addmm(
    #         inputs_norm_sq + codebook_t_norm_sq,
    #         inputs_flat,
    #         codebook_t,
    #         alpha=-2.0,
    #     )
    #     distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, n_embed or n_embed+1]

    #     return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs, use_sinkhorn=True):
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]

        # sinkhorn
        if self.training and use_sinkhorn:
            distances = (distances - distances.mean()) / (distances.std() + 1e-6)
            distances = distances - distances.min()
            Q = sinkhorn(distances)
            embed_idxs = Q.argmax(dim=-1)
        else:
            embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        # embed_idxs = distances.argmin(dim=-1)

        # 依概率采样 + 退火算法
        # if self.training:
        #     # inverted_distances = 1.0 / (distances + 1e-8)
        #     inverted_distances = -distances
        #     # T = 0.1
        #     probabilities = F.softmax(inverted_distances / self.temperature, dim=-1)
        #     # probabilities = inverse_distances / inverse_distances.sum(dim=-1, keepdim=True)
        #     # 创建一个包含 [B, h, w] 的形状，存储的每个位置对应的概率分布中每个类别的概率
        #     embed_idxs = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        #     # self.temperature = max(self.prob_decay * self.temperature, self.eps)
        #     self.temperature = max(self.temperature - self.prob_decay, self.eps)
        # else:
        #     embed_idxs = distances.argmin(dim=-1)

        return embed_idxs, distances

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):

        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors)
                              )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)

        # 未使用码本重启 ：检测并重置长期未使用的码本，防止码本崩溃（codebook collapse）
        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        """
        Update the cluster center embeddings.
        更新码本：簇中心向量

        Args:
            None
        """

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
                n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs, reference_code=None, **kwargs):
        embed_idxs, distances = self.find_nearest_embedding(inputs, use_sinkhorn=kwargs.get('use_sinkhorn', True))

        if reference_code is not None:
            # 以一定概率选择参考code的index
            bs = embed_idxs.size(0)
            # 0.04 比较好
            p = 0.04
            random_probs = torch.rand(bs).to(inputs.device)
            embed_idxs = torch.where(random_probs < p, reference_code, embed_idxs)

        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs, distances

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        latent_shape (Tuple[int, int, int]): the shape of latents, denoted (H, W, D)
        code_shape (Tuple[int, int, int]): the shape of codes, denoted (h, w, d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 latent_shape,
                 code_shape,
                 n_embed,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 commitment_loss='cos',
                 VQ_ema=True,
                 latent_weight=[1, 0.5],
                 rotation_trick=False,
                 kmeans_init=True,
                 ):
        super().__init__()

        if not len(code_shape) == len(latent_shape) == 3:
            raise ValueError("incompatible code shape or latent shape")
        if any([y % x != 0 for x, y in zip(code_shape[:2], latent_shape[:2])]):
            raise ValueError("incompatible code shape or latent shape")

        # residual quantization does not divide feature dims for quantization.
        embed_dim = np.prod(latent_shape[:2]) // np.prod(code_shape[:2]) * latent_shape[2]

        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)
        self.shape_divisor = torch.Size([latent_shape[i] // code_shape[i] for i in range(len(latent_shape))])

        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.code_shape[-1])]
        # self.n_embed = [64]*4 + [2048]*4
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.code_shape[-1])]
        assert len(self.n_embed) == self.code_shape[-1]
        assert len(self.decay) == self.code_shape[-1]

        # distance_types = ['cosine'] * (self.code_shape[-1] - 1) +  ['l2']
        distance_types = ['l2'] * (self.code_shape[-1] - 1) + ['l2']
        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0],
                                    embed_dim,
                                    decay=self.decay[0],
                                    restart_unused_codes=restart_unused_codes,
                                    ema=VQ_ema
                                    )
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.code_shape[-1])])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx],
                                     embed_dim,
                                     decay=self.decay[idx],
                                     restart_unused_codes=restart_unused_codes,
                                     ema=VQ_ema,
                                     distance_type=distance_types[idx]
                                     ) for idx in range(self.code_shape[-1])]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss
        self.commitment_weight1, self.commitment_weight2 = latent_weight

        # kmeans初始化，默认false，使用VQEmbedding中的kaiming初始化
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.ema = VQ_ema

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        data = data.unsqueeze(0)
        embeds = kmeans.residual_kmeans(
            data,
            self.n_embed,
        )

        # move to gpu
        device = self.device
        embeds = [e.to(device) for e in embeds]
        if dist_utils.is_dist_avail_and_initialized():
            embeds = dist_utils.scaled_all_reduce(embeds)

        for idx in range(len(self.n_embed)):
            self.codebooks[idx].weight.data[:-1, :].copy_(embeds[idx])
        print("codebook kmeans init success")
        self.initted.data.copy_(torch.Tensor([True]))

    def to_code_shape(self, x):
        (B, D) = x.shape
        # (rH, rW, _) = self.shape_divisor

        # x = x.reshape(B, H//rH, rH, W//rW, rW, D)
        # x = x.permute(0, 1, 3, 2, 4, 5)
        # x = x.reshape(B, H//rH, W//rW, -1)

        return x

    @property
    def device(self):
        """返回当前模型所在的设备"""
        return next(self.parameters()).device

    def to_latent_shape(self, x):
        # (B, h, w, _) = x.shape
        # (_, _, D) = self.latent_shape
        # (rH, rW, _) = self.shape_divisor

        # x = x.reshape(B, h, w, rH, rW, D)
        # x = x.permute(0, 1, 3, 2, 4, 5)
        # x = x.reshape(B, h*rH, w*rW, D)

        return x

    def quantize(self, x, reference_code=None, **kwargs):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.

        Arguments:
            x (Tensor): bottleneck feature maps to quantize.

        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.

        Shape:
            - x: (B, h, w, embed_dim)
            - quant_list[i]: (B, h, w, embed_dim)
            - codes: (B, h, w, d)
        """
        B, embed_dim = x.shape

        residual_feature = x.detach().clone()  # 会自减，所以要clone
        # x_ = x.clone()
        feature_norm = [x.pow(2).sum(-1).pow(0.5).mean(-1).detach().cpu()]

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)

        # log
        quant_norm = []
        angle = []
        angle_aggregated = []
        all_distances = []

        for i in range(self.code_shape[-1]):
            if reference_code is not None:
                quant, code, distances = self.codebooks[i](residual_feature, reference_code=reference_code[:, i],
                                                           **kwargs)
            else:
                quant, code, distances = self.codebooks[i](residual_feature, **kwargs)

            # distances = self.codebooks[i].compute_distances_gard(x_)
            # x_ = x_ - quant
            # all_distances.append(distances.unsqueeze(-2))
            all_distances.append(distances)

            # log
            cosine_sim = F.cosine_similarity(residual_feature, quant, dim=-1)
            angle.append((torch.acos(cosine_sim) * 180 / 3.14159).mean(-1).detach().cpu())

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            # log
            cosine_sim2 = F.cosine_similarity(x, aggregated_quants, dim=-1)
            angle_aggregated.append((torch.acos(cosine_sim2) * 180 / 3.14159).mean(-1).detach().cpu())

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))

            # log
            quant_norm.append(quant.pow(2).sum(-1).pow(0.5).mean(-1).detach().cpu())
            feature_norm.append(residual_feature.pow(2).sum(-1).pow(0.5).mean(-1).detach().cpu())

        codes = torch.cat(code_list, dim=-1)
        # all_distances = torch.cat(all_distances, dim=-2)
        return quant_list, codes, feature_norm, quant_norm, angle, angle_aggregated, all_distances

    def forward(self, x, num_samples=0, reference_code=None, **kwargs):
        # x_reshaped = self.to_code_shape(x)
        # 不需要转换shape
        x_reshaped = x
        # if self.training:
        #     self.init_embed_(x_reshaped)

        quant_list, codes, feature_norm, quant_norm, angle, angle_aggregated, all_distances = self.quantize(x_reshaped,
                                                                                                            reference_code=reference_code,
                                                                                                            **kwargs)

        commitment_loss, commitment_loss_ze, commitment_loss_zq = self.compute_commitment_loss(x_reshaped, quant_list)
        # quants_trunc = self.to_latent_shape(quant_list[-1])
        # 不需要转换shape
        quants_trunc = quant_list[-1]
        quants_trunc = x + (quants_trunc - x).detach()

        commitment_loss = {
            'commitment_loss': commitment_loss,
            'commitment_loss_ze': commitment_loss_ze,
            'commitment_loss_zq': commitment_loss_zq
        }

        # 和hier版本输出保持对齐
        # quants_trunc2 = quant_list[-2]
        # quants_trunc2 = x + (quants_trunc2 - x).detach()
        quants_neg = None

        return quants_trunc, commitment_loss, codes, feature_norm, quant_norm, angle, angle_aggregated, quants_neg, all_distances

    def cosine_loss(self, x1, x2):
        cos_sim = F.cosine_similarity(x1, x2, dim=1)
        loss = 1 - cos_sim
        return loss.mean()

    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []

        for idx, quant in enumerate(quant_list):
            # for idx, quant in enumerate([quant_list[-1]]):
            if self.commitment_loss == 'cos':
                partial_loss1 = self.cosine_loss(x, quant.detach()) * self.commitment_weight1
                if self.ema:
                    partial_loss2 = torch.tensor(0.0).to(x.device)
                else:
                    partial_loss2 = self.cosine_loss(x.detach(), quant) * self.commitment_weight2
            else:
                # 只让x靠近codebook的相加
                # if idx != len(quant_list) - 1:
                #     partial_loss1 = 0.
                # else:
                #     partial_loss1 = (x - quant.detach()).pow(2.0).mean() * self.commitment_weight1
                partial_loss1 = (x - quant.detach()).pow(2.0).mean() * self.commitment_weight1  # * (0.5**idx)
                if self.ema:
                    partial_loss2 = torch.tensor(0.0).to(x.device)
                else:
                    partial_loss2 = (x.detach() - quant).pow(2.0).mean() * self.commitment_weight2  # * (0.5**idx)

            loss_list.append(partial_loss1 + partial_loss2)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss, partial_loss1, partial_loss2

    def embed_code_with_grad(self, code):
        assert code.shape[1:] == self.code_shape[-1:]

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        embeds = torch.cat(embeds, dim=-2).sum(-2)
        # embeds = self.to_latent_shape(embeds)

        return embeds

    @torch.no_grad()
    def embed_code(self, code):
        assert code.shape[1:] == self.code_shape[-1:]

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        embeds = torch.cat(embeds, dim=-2).sum(-2)
        # embeds = self.to_latent_shape(embeds)

        return embeds

    @torch.no_grad()
    def embed_code_with_depth(self, code, to_latent_shape=False):
        '''
        do not reduce the code embedding over the axis of code-depth.

        Caution: RQ-VAE does not use scale of codebook, thus assume all scales are ones.
        '''
        # spatial resolution can be different in the sampling process
        assert code.shape[-1] == self.code_shape[-1]

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        if to_latent_shape:
            embeds = [self.to_latent_shape(embed.squeeze(-2)).unsqueeze(-2) for embed in embeds]
        embeds = torch.cat(embeds, dim=-2)

        return embeds, None

    @torch.no_grad()
    def embed_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Decode the input codes, using [0, 1, ..., code_idx] codebooks.

        Arguments:
            code (Tensor): codes of input image
            code_idx (int): the index of the last selected codebook for decoding

        Returns:
            embeds (Tensor): quantized feature map
        """

        assert code.shape[-1] == self.code_shape[-1]
        assert code_idx < code.shape[-1]

        B, _ = code.shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        if decode_type == 'select':
            embeds = embeds[code_idx].view(B, -1)
        elif decode_type == 'add':
            embeds = torch.cat(embeds[:code_idx + 1], dim=-2).sum(-2)
        else:
            raise NotImplementedError(f"{decode_type} is not implemented in partial decoding")

        # embeds = self.to_latent_shape(embeds)

        return embeds

    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):

        x = self.to_code_shape(x)

        residual_feature = x.detach().clone()
        soft_code_list = []
        code_list = []

        n_codebooks = self.code_shape[-1]
        for i in range(n_codebooks):
            codebook = self.codebooks[i]
            distances = codebook.compute_distances(residual_feature)
            soft_code = F.softmax(-distances / temp, dim=-1)

            if stochastic:
                soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
                code = torch.multinomial(soft_code_flat, 1)
                code = code.reshape(*soft_code.shape[:-1])
            else:
                code = distances.argmin(dim=-1)
            quants = codebook.embed(code)
            residual_feature -= quants

            code_list.append(code.unsqueeze(-1))
            soft_code_list.append(soft_code.unsqueeze(-2))

        code = torch.cat(code_list, dim=-1)
        soft_code = torch.cat(soft_code_list, dim=-2)
        return soft_code, code
