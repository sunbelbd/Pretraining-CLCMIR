"""Evaluation"""

from __future__ import print_function

from collections import OrderedDict

import numpy as np
import paddle
import sys
import time


# import torch
# from torch.autograd import Variable

# from model import SCAN, xattn_score_t2i, xattn_score_i2t

#
# def get_non_pad_mask(seq):
#     assert seq.dim() == 2
#     return seq.ne(0).type(torch.float).unsqueeze(-1)
#     # return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_non_pad_mask(seq):
    assert seq.ndim == 2
    # return seq.ne(0).type(torch.float).unsqueeze(-1).to(device=seq.device)
    return paddle.cast(seq != 0, 'float32').unsqueeze(axis=-1)


# def get_attn_key_pad_mask(seq_k, seq_q):
#     ''' For masking out the padding part of key sequence. '''
#     # Expand to fit the shape of key query attention matrix.
#     if seq_q.dim() == 1:
#         seq_q = seq_q.unsqueeze(0)
#     if seq_k.dim() == 1:
#         seq_k = seq_k.unsqueeze(0)
#     len_q = seq_q.size(1)
#     # padding_mask = seq_k.eq(Constants.PAD)
#     padding_mask = seq_k.eq(0)
#     padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
#     return padding_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            # print(k, v)
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    text_pads = None
    img_pads = None

    max_text_length = max([batch[1].shape[1] for batch in data_loader])
    max_boxes = max([batch[0].shape[1] for batch in data_loader])

    for i, (images, captions, target_mask, vision_mask, ids, *aux) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), max_boxes, images.shape[2]), dtype=np.float32)
            # img_embs = paddle.zeros((len(data_loader.dataset), max_boxes, images.size(2)), dtype="float32")
            cap_embs = np.zeros((len(data_loader.dataset), max_text_length), dtype=np.int64)
            # cap_embs = paddle.zeros((len(data_loader.dataset), max_text_length), dtype="int64")
            text_pads = np.zeros((len(data_loader.dataset), max_text_length), dtype=np.int64)
            # text_pads = paddle.zeros((len(data_loader.dataset), max_text_length), dtype="int64")
            img_pads = np.zeros((len(data_loader.dataset), max_boxes), dtype=np.int64)
            # img_pads = paddle.zeros((len(data_loader.dataset), max_boxes), dtype="int64")
            # cache embeddings
        img_embs[ids, :captions.shape[1]] = images.cpu().numpy().copy()
        cap_embs[ids, :captions.shape[1]] = captions.cpu().numpy().copy()
        text_pads[ids, :captions.shape[1]] = target_mask.cpu().numpy().copy()
        img_pads[ids, :vision_mask.shape[1]] = vision_mask.cpu().numpy().copy()
        # img_embs[ids, :images.shape[1]] = images.clone().detach()
        # cap_embs[ids, :captions.shape[1]] = captions.clone().detach()
        # text_pads[ids, :captions.shape[1]] = target_mask.clone().detach()
        # img_pads[ids, :vision_mask.shape[1]] = vision_mask.clone().detach()

    return img_embs, cap_embs, text_pads, img_pads


def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p


# def shard_xattn_t2i(images, captions, caplens, opt, shard_size=128):
#     """
#     Computer pairwise t2i image-caption distance with locality sharding
#     """
#     n_im_shard = (len(images) - 1) // shard_size + 1
#     n_cap_shard = (len(captions) - 1) // shard_size + 1
#
#     d = np.zeros((len(images), len(captions)))
#     for i in range(n_im_shard):
#         im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
#         for j in range(n_cap_shard):
#             sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i, j))
#             cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
#             im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
#             s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
#             l = caplens[cap_start:cap_end]
#             sim = xattn_score_t2i(im, s, l, opt)
#             d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
#     sys.stdout.write('\n')
#     return d
#
#
# def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
#     """
#     Computer pairwise i2t image-caption distance with locality sharding
#     """
#     n_im_shard = (len(images) - 1) / shard_size + 1
#     n_cap_shard = (len(captions) - 1) / shard_size + 1
#
#     d = np.zeros((len(images), len(captions)))
#     for i in range(n_im_shard):
#         im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
#         for j in range(n_cap_shard):
#             sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i, j))
#             cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
#             im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
#             s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
#             l = caplens[cap_start:cap_end]
#             sim = xattn_score_i2t(im, s, l, opt)
#             d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
#     sys.stdout.write('\n')
#     return d


def shard_xattn_t2i_model(model, images, captions, cap_masks, img_masks, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            n_cap = cap_end - cap_start
            n_img = im_end - im_start

            im = paddle.to_tensor(images[im_start:im_end]).cuda()
            # im = images[im_start:im_end].cuda()
            cap = paddle.to_tensor(captions[cap_start:cap_end]).cuda()
            # cap = captions[cap_start:cap_end].cuda()
            cap_mask = paddle.to_tensor(cap_masks[cap_start:cap_end]).cuda()
            # cap_mask = cap_masks[cap_start:cap_end].cuda()
            im = im.unsqueeze(1).expand((n_img, n_cap, im.shape[1], im.shape[2])).reshape(
                (-1, im.shape[1], im.shape[2]))
            cap = cap.unsqueeze(0).expand((n_img, n_cap, cap.shape[1])).reshape((-1, cap.shape[1]))  # .contiguous().

            cap_mask = cap_mask.unsqueeze(0).expand((n_img, n_cap, cap.shape[1])).reshape((-1, cap.shape[1]))
            cap_type = paddle.zeros_like(cap_mask)  # cap_mask.new_zeros(cap_mask.shape)

            # img_mask = Variable(torch.from_numpy(img_masks[im_start:im_end])).long().cuda()
            img_mask = img_masks[im_start:im_end].cuda()
            img_mask = img_mask.unsqueeze(1).expand((n_img, n_cap, im.shape[1])).reshape((-1, im.shape[1]))

            scores = model.txt_enc(cap, cap_type, cap_mask, im, img_mask, istest=True)  # .matmul(im,s)
            sim = scores.view(n_img, n_cap)

            d[im_start:im_end, cap_start:cap_end] = sim.cpu().numpy()
    sys.stdout.write('\n')
    return d


# def shard_xattn_i2t_model(model, images, captions, cap_masks, img_masks, opt, shard_size=128):
#     """
#     Computer pairwise i2t image-caption distance with locality sharding
#     """
#     n_im_shard = (len(images) - 1) // shard_size + 1
#     n_cap_shard = (len(captions) - 1) // shard_size + 1
#
#     d = np.zeros((len(images), len(captions)))
#     for i in range(n_im_shard):
#         im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
#         for j in range(n_cap_shard):
#             sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i, j))
#             cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
#             n_cap = cap_end - cap_start
#             n_img = im_end - im_start
#
#             im = paddle.to_tensor(images[im_start:im_end]).cuda()
#             # im = images[im_start:im_end].cuda()
#             cap = paddle.to_tensor(captions[cap_start:cap_end]).cuda()
#             # cap = captions[cap_start:cap_end].cuda()
#             cap_mask = paddle.to_tensor(cap_masks[cap_start:cap_end]).cuda()
#             # cap_mask = cap_masks[cap_start:cap_end].cuda()
#             im = im.unsqueeze(1).expand(n_img, n_cap, im.shape[1], im.shape[2]).reshape(-1, im.shape[1], im.shape[2])
#             cap = cap.unsqueeze(0).expand(n_img, n_cap, cap.shape[1]).reshape(-1, cap.shape[1])  # .contiguous().
#
#             cap_mask = cap_mask.unsqueeze(0).expand(n_img, n_cap, cap.shape[1]).reshape(-1, cap.shape[1])
#             cap_type = paddle.zeros_like(cap_mask)  # cap_mask.new_zeros(cap_mask.shape)
#
#             img_mask = img_masks[im_start:im_end].cuda()
#             img_mask = img_mask.unsqueeze(1).expand(n_img, n_cap, im.shape[1]).reshape(-1, im.shape[1])
#
#             scores = model.txt_enc(cap, cap_type, cap_mask, im, img_mask)  # .matmul(im,s)
#             sim = scores.view(n_img, n_cap)  # model.cross_att(img_emb,cap_emb,test=True)
#             d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
#     sys.stdout.write('\n')
#     return d


def i2t(images, captions, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
