#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Token masking module for Masked LM."""

import numpy
import torch
import logging


def mask_uniform(ys_pad, mask_token, eos, ignore_id):
    """Replace random tokens with <mask> label and add <eos> label.

    The number of <mask> is chosen from a uniform distribution
    between one and the target sequence's length.
    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int mask_token: index of <mask>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_out = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in = [y.clone() for y in ys]
    for i in range(len(ys)):
        num_samples = numpy.random.randint(1, len(ys[i]) + 1)
        idx = numpy.random.choice(len(ys[i]), num_samples, replace = False)

        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]

    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def mask_uniform_dlp(ys_pad, mask_token, eos, ignore_id):
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_len = len(ys)

    ys_in_del = [y.clone() for y in ys]
    dur_out_del = []
    for i in range(ys_len):
        y_len = len(ys[i])
        num_samples = numpy.random.randint(1, y_len + 1)
        idx = numpy.random.choice(y_len, num_samples)

        ys_in_del[i][idx] = mask_token
        
        unmask_idx = torch.arange(y_len).to(ys[i]).masked_fill((ys_in_del[i] == mask_token), ignore_id)
        _, dur = unmask_idx.unique_consecutive(return_counts = True)

        ys_in_del[i] = ys_in_del[i][dur.cumsum(0) - 1]
        dur_out_del.append(dur.masked_fill((ys_in_del[i] != mask_token), ignore_id))
    
    # [1,2,3,4] -> [0,1,0,2,0,3,0,4] 
    ys_in_ins = [torch.stack([y.new(y.size()).fill_(ignore_id), y.clone()]).t().flatten() for y in ys]
    dur_out_ins = [torch.stack([y.new(y.size()).fill_(0), y.new(y.size()).fill_(1)]).t().flatten() for y in ys]
    
    for i in range(len(ys)):
        ylen = len(ys[i])
        num_samples = numpy.random.randint(1, ylen + 1)
        idx = numpy.random.choice(ylen, num_samples) * 2

        # [0,1,0,2,0,3,0,4] ->(random) [0, 1, mask, 2,mask ,3, 0, 4]
        ys_in_ins[i][idx] = mask_token
        # [0, 1, mask, 2,mask ,3, 0, 4] -> [1, mask, 2, mask, 3, 4]
        
        tgt_idx = torch.where(ys_in_ins[i] != ignore_id)[0] # 取出Mask 跟 非Ignore id

        ys_in_ins[i] = ys_in_ins[i][tgt_idx] # 貼回去，得到有insert mask的sequence
        dur_out_ins[i] = dur_out_ins[i][tgt_idx] # 對應的位置的數字

    
    ys_in_del.extend(ys_in_ins)
    dur_out_del.extend(dur_out_ins)

    return pad_list(ys_in_del, eos), pad_list(dur_out_del, ignore_id)

    
def shrink(ys_in, mask_token, eos, ignore_id):
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    ys_copy = [y.clone() for y in ys_in]
    ys_num = [y.new(y.size()).fill_(ignore_id) for y in ys_in]
    ys_shrink = []

    for i in range(len(ys_copy)):
        shrink , cnt  = ys_copy[i].unique_consecutive(return_counts = True)
        mask_idx = torch.nonzero(shrink == mask_token)
        ys_num[i][mask_idx] = cnt[mask_idx]
        ys_shrink.append(shrink)
    # # ys_num = [y.new(y.size()).fill_(ignore_id) for y in ys_in] # fill with the number of mask
    # # ys_shrink = [y.new(y.size()).fill_(eos) for y in ys_in]
    # for i in range(len(ys_in)):
    #     have_mask = False
    #     mask_count = 0
    #     index = 0
    #     # logging.info(f'ys_in:{ys_in[i]}')
    #     for j in range(len(ys_in[i])):
    #         if (ys_in[i][j] == mask_token):
    #             mask_count += 1
    #             have_mask = True
    #         else :
    #             if (have_mask):
    #                 ys_shrink[i][index] = mask_token
    #                 ys_num[i][index] = mask_count
    #                 mask_count = 0
    #                 have_mask = False
    #                 index += 1
    #             ys_shrink[i][index] = ys_in[i][j]
    #             index += 1
    #         if (have_mask):
    #             ys_shrink[i][index] = mask_token
    #             ys_num[i][index] = mask_count
    return pad_list(ys_shrink, eos), pad_list(ys_num, ignore_id)

def insert_mask(ys_in, mask_token, eos, ignore_id):
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    ys_insert = [y.clone() for y in ys_in]
    ys_insert = [y[y != ignore_id] for y in ys_insert]
    ys_num = [y.new(y.size()).fill_(ignore_id) for y in ys_in]
    for i in range(len(ys_insert)):
        num_samples = numpy.random.randint(1, len(ys_insert[i]) + 1)
        idx = numpy.random.choice(len(ys_insert[i]), num_samples)
        
        for i in range(len(ys_insert)):
            for j in range(len(ys_insert[i])):
                if (ys_insert[i][j] == mask_token):
                    ys_num[i][j] = 1

        # print(f'ys_before_insert:{ys_insert[i]}')
        # print(f'ys_num_before_ins:{ys_num[i]}')
        # print(f'idx:{idx}')
        for k, n in enumerate(idx):
            # print(n)
            ys_insert[i] = torch.cat((ys_insert[i][:n], torch.tensor((1, )).fill_(mask_token).to(ys_in.device), ys_insert[i][n:]))
            ys_num[i] = torch.cat((ys_num[i][:n], torch.zeros((1, )).to(ys_in.device), ys_num[i][n:]))
        # print(f'ys_insert:{ys_insert[i]}')
        # print(f'ys_insert_num:{ys_num[i]}')
    return pad_list(ys_insert, eos), pad_list(ys_num, ignore_id)


def expand(ys_in, ys_nums, mask_token, eos, ignore_id):
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    ys_expand = [y.clone() for y in ys_in]
    ys_num = [y.clone() for y in ys_nums]
    print(ys_num)
    for i in range(len(ys_num)):
        ys_temp = [numpy.array([mask_token] * t) if t > 0 else ys_in[i][j].numpy() for j, t in enumerate(ys_num[i])]
        ys_expand = torch.tensor(numpy.hstack(ys_temp)).unsqueeze(0)
        # for j, t in enumerate(ys_num[i]):
        #     if (t == 0):
        #         ys_expand[i] = torch.cat((ys_expand[i][:j], ys_expand[i][j+1:])) # delete element
        #     elif (t > 1):
        #         print(f'before expand: {ys_expand}')
                
        #         mask_tensor = torch.zeros([t - 1]).fill_(mask_token).to(ys_in)
        #         print(mask_tensor.shape)
        #         ys_expand[i] = torch.cat((ys_expand[i][:j], mask_tensor, ys_expand[i][j:]))
        #         print(f'after expand : {ys_expand}')

    return pad_list(ys_expand, eos)