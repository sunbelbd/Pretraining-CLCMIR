# -----------------------------------------------------------
# Part of the code is based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer
from paddlenlp.transformers import BertModel, BertTokenizer

from .evaluation import LogCollector


# from transformers import BertTokenizer


# from .modeling_bertnewsinglecut import BertModelNew


class Attention(Layer):
    """ Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = paddle.randn(5, 1, 256)
         >>> context = paddle.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias_attr=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias_attr=False)
        self.softmax = nn.Softmax(axis=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.shape
        query_len = context.shape[1]

        if self.attention_type == "general":
            query = query.reshape((batch_size * output_len, dimensions))
            query = self.linear_in(query)
            query = query.reshape((batch_size, output_len, dimensions))

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = paddle.bmm(query, context.transpose(perm=(0, 2, 1)))

        # Compute weights across every context sequence
        attention_scores = attention_scores.reshape((batch_size * output_len, query_len))
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.reshape((batch_size, output_len, query_len))

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = paddle.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = paddle.concat((mix, query), axis=2)
        combined = combined.reshape((batch_size * output_len, 2 * dimensions))

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).reshape((batch_size, output_len, dimensions))
        output = self.tanh(output)

        return output, attention_weights


def slice(x, idx):
    """
    Implement tensor slicing using bool array
    :param x:
    :param idx:
    :return:
    """
    assert idx.ndim == 1
    return paddle.gather(x, idx.nonzero())


def clamp(x, min):
    """
    clamp_x = where(x>=min, x, min)
    :param x:
    :param min:
    :return:
    """
    x[(x < min).numpy()] = min
    return x


def get_non_pad_mask(seq):
    assert seq.ndim == 2
    # return seq.ne(0).type(torch.float).unsqueeze(-1).to(device=seq.device)
    return paddle.cast(seq != 0, 'float32').unsqueeze(axis=-1)  # .cuda(device_id=seq.place)


def l1norm(X, dim, eps=1e-5):
    """L1-normalize columns of X
    """
    norm = X.abs().sum(axis=dim, keepdim=True) + eps
    X = paddle.divide(X, norm)
    return X


def l2norm(X, dim, eps=1e-5):
    """L2-normalize columns of X
    """
    norm = paddle.pow(X, 2).sum(axis=dim, keepdim=True).add(eps).sqrt() + eps
    X = paddle.divide(X, norm)
    return X


# class EncoderCross(nn.Module):
#     def __init__(self):
#         super(EncoderCross, self).__init__()
#         dropout = 0.1
#         self.opt = opt
#         self.margin = 0.2
#
#     def forward(self, scores, n_img, n_cap, test=False):
#         scores = scores.view(n_img, n_cap)
#         diagonal = scores.diag().view(scores.size(0), 1)
#         d1 = diagonal.expand_as(scores)
#         d2 = diagonal.t().expand_as(scores)
#
#         # compare every diagonal score to scores in its column
#         # caption retrieval
#         cost_s = (self.margin + scores - d1).clamp(min=0)
#         # compare every diagonal score to scores in its row
#         # image retrieval
#         cost_im = (self.margin + scores - d2).clamp(min=0)
#
#         # clear diagonals
#         # keep the maximum violating negative for each query
#         eps = 1e-5
#         cost_s = cost_s.pow(4).sum(1).add(eps).sqrt().sqrt()  # .sqrt()#.div(cost_s.size(1)).mul(2)
#         cost_im = cost_im.pow(4).sum(0).add(eps).sqrt().sqrt()  # .sqrt()#.div(cost_im.size(0)).mul(2)
#         return cost_s.sum() + cost_im.sum()

def soft_cross_entropy(input, target, reduction='mean'):
    """
    Cross entropy loss with input logits and soft target
    :param input: Tensor, size: (N, C)
    :param target: Tensor, size: (N, C)
    :param reduction: 'none' or 'mean' or 'sum', default: 'mean'
    :return: loss
    """
    eps = 1.0e-3
    # debug = False
    valid = (target.sum(1) - 1).abs() < eps
    # if debug:
    #     print('valid', valid.sum().item())
    #     print('all', valid.numel())
    #     print('non valid')
    #     print(target[valid == 0])
    if paddle.cast(valid, 'int32').sum() == 0:
        # return input.new_zeros(())
        return paddle.zeros(shape=(1,))
    if reduction == 'mean':
        return (- F.log_softmax(slice(input, valid), 1) * slice(target, valid)).sum(1).mean(0)
    elif reduction == 'sum':
        return (- F.log_softmax(slice(input, valid), 1) * slice(target, valid)).sum()
    elif reduction == 'none':
        # l = input.new_zeros((input.shape[0],))
        l = paddle.zeros((input.shape[0],))
        l[valid.numpy()] = (- F.log_softmax(slice(input, valid), 1) * slice(target, valid)).sum(1)
        return l
    else:
        raise ValueError('Not support reduction type: {}.'.format(reduction))


class EncoderText(Layer):

    def __init__(self, opt, logger=None):
        super(EncoderText, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.vocab = self.tokenizer.vocab.token_to_idx
        self.fc = nn.Linear(2048, 768)
        # self.fc2 = nn.Linear(768, 30600)
        self.fc2 = nn.Linear(768, 105879)
        self.fc3 = nn.Linear(768, 1601)
        self.vision_type_embeddings = nn.Embedding(1, 768)
        self.norm = nn.LayerNorm(768)
        self.relu = nn.ReLU()
        self.mlm = opt.mlm
        self.cm = opt.cm
        self.mrm = opt.mrm
        self.aux_txt_mlm = opt.aux_txt_mlm
        # self.task = opt.task
        self.aux_t2t_recovery = opt.aux_t2t_recovery
        self.i2t_recovery = opt.i2t_recovery
        self.logger = logger

        if self.aux_t2t_recovery:
            self.atten_align_t2t = Attention(768)

        if self.i2t_recovery:
            self.atten_align_i2t = Attention(768)

        # if self.task == 'finetune':
        #     self.finetune_fc = nn.Linear(768, 1)
        # else:
        #     self.proj = None

        # self.ff = EncoderLayerMinus(768,768,0.1)

    def calPrec(self, pred, grnd, ignore=-1):
        idx = grnd != ignore
        pred = paddle.masked_select(pred, idx)
        grnd = paddle.masked_select(grnd, idx)
        # pred = pred[idx]
        # grnd = grnd[idx]
        corr = paddle.cast(pred == grnd, dtype='int32')
        # prec = corr.sum()/corr.size(0)
        return corr.sum() * 1.0, corr.shape[0] * 1.0  # prec

    def sample_negative(self, vision_feat, text_output, non_pad_mask, vision_mask, head_mask, offset):
        offset = min(offset, vision_feat.shape[0])
        index = paddle.to_tensor([i for i in range(offset, vision_feat.shape[0])] + [i for i in range(offset)])
        # index = [i for i in range(1,vision_feat.size(0))] + [0]
        vision_feat = paddle.index_select(vision_feat, index)  # vision_feat[index]
        vision_mask = paddle.index_select(vision_mask, index)  # vision_mask[index]
        catfeat = paddle.concat([text_output, vision_feat], 1)
        vision_mask_cat = paddle.concat([non_pad_mask, vision_mask], 1).squeeze()
        extended_attention_mask_cat = vision_mask_cat.unsqueeze(axis=[1, 2])  # [:, None, None, :]
        extended_attention_mask_cat = (1.0 - extended_attention_mask_cat) * -10000.0
        catnewn = self.bert.encoder(catfeat, extended_attention_mask_cat)
        catnewn = catnewn[0]
        text_out = catnewn[:, 0]  # .view(bs,bs,-1)
        img_out = catnewn[:, text_output.shape[1]:].sum(1)
        return text_out, img_out

    def mlm_head(self, text_ids, mlm_labels, head_mask, is_TLM=False, lang1_len=None):
        # reset position ids for TLM task
        if is_TLM and lang1_len is not None:
            bs, max_length = text_ids.shape[:2]
            grid_ind, grid_pos = paddle.meshgrid(paddle.arange(bs, dtype='int64'),
                                                 paddle.arange(max_length, dtype='int64'))
            # .cuda(placedevice_id=text_ids.place)
            # print(grid_pos.shape, lang1_len.shape)
            # assert lang1_len.shape[0] == bs
            # position_ids = grid_pos
            # print(position_ids[grid_pos >= lang1_len].shape, lang1_len.expand((bs, max_length))[grid_pos >= lang1_len].shape)
            # position_ids[grid_pos >= lang1_len] -= \
            #                 lang1_len.expand((bs, max_length))[grid_pos >= lang1_len]
            position_ids = paddle.where(grid_pos >= lang1_len,
                                        grid_pos - lang1_len.expand((bs, max_length)), grid_pos)
        else:
            position_ids = None
        text_emb = self.bert.embeddings(input_ids=text_ids, position_ids=position_ids,
                                        token_type_ids=paddle.zeros_like(text_ids))
        # make extended mask for bert.encoder since it accepts processed attention mask
        text_attn_mask = (text_ids > 0).unsqueeze(axis=(1, 2)).cast(text_ids.dtype)
        text_attn_mask = (1.0 - text_attn_mask) * -10000.0
        text_hidden_states = \
            self.bert.encoder(text_emb, text_attn_mask)
        text_logits = self.fc2(text_hidden_states)
        loss_mlm = F.cross_entropy(text_logits.reshape((-1, text_logits.shape[-1])),
                                   mlm_labels.reshape((-1,)), ignore_index=-1)
        return loss_mlm

    def forward(self, input_ids, token_type_ids, non_pad_mask, vision_feat, vision_mask, gt_labels=None,
                vision_labels=None, aux=None, MLM=False, istest=False):

        # text embeddings using bert embedding func
        text_emb = self.bert.embeddings(input_ids=input_ids, position_ids=None,
                                        token_type_ids=token_type_ids.squeeze())
        head_mask = [None] * 20
        # vision embeddings, fc layer after frcnn 2048->768, followed by a normalization layer
        # Here vision_feat has no token_type_id or position emb,
        # but directly feed [text_output, vision_feat] to xatten layer
        vision_feat = self.fc(vision_feat)
        vision_type_emb = self.vision_type_embeddings(paddle.zeros(vision_feat.shape[:2], dtype='int64'))
        vision_feat += vision_type_emb
        vision_feat = self.norm(vision_feat)

        bs, tl = text_emb.shape[0:2]
        # tl = text_emb.size(1)
        vl = vision_feat.shape[1]

        if istest == False and MLM == False:
            text_emb = text_emb.unsqueeze(0).expand((bs, -1, -1, -1)).reshape((bs * bs, tl, -1))
            vision_feat = vision_feat.unsqueeze(1).expand((-1, bs, -1, -1)).reshape((bs * bs, vl, -1))
            non_pad_mask = non_pad_mask.unsqueeze(0).expand((bs, -1, -1)).reshape((bs * bs, -1))
            vision_mask = vision_mask.unsqueeze(1).expand((-1, bs, -1)).reshape((bs * bs, -1))

        catfeat = paddle.concat([text_emb, vision_feat], 1)
        vision_mask_cat = paddle.concat([non_pad_mask, vision_mask], 1).squeeze()
        extended_attention_mask_cat = vision_mask_cat.squeeze().unsqueeze((1, 2))
        extended_attention_mask_cat = (1.0 - extended_attention_mask_cat) * -10000.0

        catnew = self.bert.encoder(catfeat, extended_attention_mask_cat)
        # catnew = catnew[0]
        if MLM:
            if self.cm:
                text_global_neg1, img_global_neg1 = self.sample_negative(vision_feat, text_emb, non_pad_mask,
                                                                         vision_mask, head_mask, 1)
                text_global_neg2, img_global_neg2 = self.sample_negative(vision_feat, text_emb, non_pad_mask,
                                                                         vision_mask, head_mask, 2)
                text_global_neg3, img_global_neg3 = self.sample_negative(vision_feat, text_emb, non_pad_mask,
                                                                         vision_mask, head_mask, 3)
            # text_global_neg4, img_global_neg4 = self.sample_negative(vision_feat,text_output,non_pad_mask,vision_mask,head_mask,4)
            # text_global_neg5, img_global_neg5 = self.sample_negative(vision_feat,text_output,non_pad_mask,vision_mask,head_mask,5)

            text_out = catnew[:, :text_emb.shape[1]]
            img_out = catnew[:, text_emb.shape[1]:]
            # # text_global_pos = catnew[:, 0]  # .view(bs,bs,-1)
            # text_global_pos = text_out[:, 1:].mean(1)
            # img_global_pos = img_out.mean(1)

            pre_labels = self.fc2(text_out)
            pre_labels_vis = self.fc3(img_out)
            pre_vis = pre_labels_vis.argmax(axis=-1).reshape((-1,))
            pre_txt = pre_labels.argmax(axis=-1)
            # print(pre_txt.shape, gt_labels.shape)
            corr, total = self.calPrec(pre_txt.reshape((-1,)), gt_labels.reshape((-1,)))

            # corr_vis, total_vis = self.calPrec(pre_vis, vision_labels.view(-1), ignore=0)

            loss1 = F.cross_entropy(pre_labels.reshape((-1, pre_labels.shape[-1])),
                                    gt_labels.reshape((-1,)),
                                    ignore_index=-1)

            if self.logger:
                self.logger.update('MLM_loss:', loss1.mean().numpy()[0])
            # print('MLM_loss', loss1.mean().item())
            loss2 = soft_cross_entropy(pre_labels_vis.reshape((-1, pre_labels_vis.shape[-1])),
                                       vision_labels.reshape((-1, pre_labels_vis.shape[-1])))
            if self.logger:
                self.logger.update('MRM_loss:', loss2.mean().numpy()[0])
            # print('MRM_loss', loss2.mean().item())

            corr = loss1 / loss1 * corr
            total = loss1 / loss1 * total

            # corr_vis = loss1 / loss1 * corr_vis
            # total_vis = loss1 / loss1 * total_vis

            if self.i2t_recovery:
                # attn_weight = torch.matmul(self.proj(text_output), vision_feat.permute(0, 2, 1))
                # attn_weight = F.softmax(attn_weight, dim=1).to(dtype=vision_feat.dtype)
                # atten_cap_text_embedding = torch.matmul(attn_weight, vision_feat)
                # Fixed: should use text embedding, but text_output is overwriten by upper layers. Fix it.
                atten_cap_text_embedding, _ = self.atten_align_i2t(text_emb, vision_feat)
                attned_cap_attention_mask = non_pad_mask.unsqueeze((1, 2))  # [:, None, None, :]
                attned_cap_attention_mask = (1.0 - attned_cap_attention_mask) * -10000.0
                attn_cap_hidden_states = \
                    self.bert.encoder(atten_cap_text_embedding, attned_cap_attention_mask)
                recovered_cap_logits = self.fc2(attn_cap_hidden_states)
                i2t_recovery_labels = input_ids.clone().detach()
                i2t_recovery_labels[i2t_recovery_labels == self.vocab["[MASK]"]] = -1
                loss_i2t_recovery = F.cross_entropy(recovered_cap_logits.reshape((-1, recovered_cap_logits.shape[-1])),
                                                    i2t_recovery_labels.reshape((-1,)), ignore_index=-1)
                if self.logger:
                    self.logger.update('i2t_rec_loss: ', loss_i2t_recovery.mean().numpy()[0])
                # print('i2t_rec_loss', loss_i2t_recovery.mean().item())

            if self.aux_txt_mlm and len(aux) >= 5:
                mono_text_ids, mono_mlm_labels, para_text_ids, para_tlm_labels, text_en_ids = aux[0:5]
                # print(mono_text_ids.shape, mono_mlm_labels.shape)
                loss_mono_mlm = self.mlm_head(mono_text_ids, mono_mlm_labels, head_mask)
                lang1_len = paddle.cast(text_en_ids > 0, 'int64').sum(1, keepdim=True) + 2
                loss_para_tlm = self.mlm_head(para_text_ids, para_tlm_labels, head_mask,
                                              is_TLM=True, lang1_len=lang1_len)
                if self.logger:
                    self.logger.update('mono_mlm_loss: ', loss_mono_mlm.mean().numpy()[0])
                    self.logger.update('para_tlm_loss: ', loss_para_tlm.mean().numpy()[0])
                # print('mono_mlm_loss', loss_mono_mlm.mean().item())
                # print('para_tlm_loss', loss_para_tlm.mean().item())

            if self.aux_t2t_recovery:
                text_en_ids, text_other_ids = aux[-2:]
                text_en_emb = self.bert.embeddings(input_ids=text_en_ids, position_ids=None,
                                                   token_type_ids=paddle.zeros_like(text_en_ids))
                text_other_emb = self.bert.embeddings(input_ids=text_other_ids, position_ids=None,
                                                      token_type_ids=paddle.zeros_like(text_other_ids))
                # attn_en_weight = torch.matmul(self.proj(text_en_emb), text_other_emb.permute(0, 2, 1))
                # attn_en_weight = F.softmax(attn_en_weight, dim=1).to(dtype=text_en_emb.dtype)
                # atten_en_text_embedding = torch.matmul(attn_en_weight, text_other_emb)
                # Fixed: should use different attention matrix to align two langs and lang-vision
                atten_en_text_embedding, _ = self.atten_align_t2t(text_en_emb, text_other_emb)
                # attned_en_attention_mask = (text_en_ids > 0)[:, None, None, :].to()
                attned_en_attention_mask = paddle.cast((text_en_ids > 0).unsqueeze(axis=(1, 2)),
                                                       dtype=text_en_ids.dtype)
                attned_en_attention_mask = (1.0 - attned_en_attention_mask) * -10000.0
                attn_en_hidden_states = \
                    self.bert.encoder(atten_en_text_embedding, attned_en_attention_mask)
                recovered_en_logits = self.fc2(attn_en_hidden_states)
                t2t_recovery_labels = text_en_ids.clone().detach()
                loss_t2t_recovery = F.cross_entropy(recovered_en_logits.reshape((-1, recovered_en_logits.shape[-1])),
                                                    t2t_recovery_labels.reshape((-1,)), ignore_index=-1)
                if self.logger:
                    self.logger.update('t2t_rec_loss: ', loss_t2t_recovery.mean().numpy()[0])
                # print('t2t_rec_loss', loss_t2t_recovery.mean().item())

            # print(corr/total)
            loss = 0.0
            if self.mlm:
                loss += 4 * loss1
            if self.mrm:
                loss += 4 * loss2
            # computing overall loss
            if self.i2t_recovery:
                loss += loss_i2t_recovery
            if self.aux_t2t_recovery:
                loss += loss_t2t_recovery
            if self.aux_txt_mlm:
                loss += 4 * loss_mono_mlm + 4 * loss_para_tlm
            return loss, corr, total  # , corr_vis, total_vis  # + loss2

        # text_out = catnew[:, 0]  # .view(bs,bs,-1)
        # print("catnew shape ", catnew.shape)
        cls_out = catnew[:, 0]
        vision_output = catnew[:, text_emb.shape[1]:].sum(1)  # .view(bs,bs,-1)
        # print(text_out[:5], vision_output[:5])
        # if self.task == 'finetune':
        #     scores = self.finetune_fc(cls_out)
        # else:
        scores = paddle.nn.functional.cosine_similarity(vision_output, cls_out, -1)
        # print("scores shape ", scores.shape)
        margin = 0.2
        if istest:
            return scores  # .view(bs,bs,-1),vision_output.view(bs,bs,-1)#[0]
        else:
            scores = scores.reshape((bs, bs))
            # diagonal = paddle.diag(scores)
            # d1 = diagonal.unsqueeze(0).expand_as(scores)
            # d2 = diagonal.unsqueeze(1).expand_as(scores)
            # cost_s = clamp(margin + scores - d1, 0)
            # cost_im = clamp(margin + scores - d2, 0)
            # return cost_s.sum() + cost_im.sum()
            # cost_s = (margin + scores - d1).clamp(min=0)
            # cost_im = (margin + scores - d2).clamp(min=0)

            diagonal = paddle.diag(scores)  # scores.diag()
            scores_subtract_diag = scores - paddle.diag(diagonal)
            # hardest example hinge loss
            # Unlike pytorch, paddle max function only returns max value along input dimension
            cost_s = clamp(margin + scores_subtract_diag.max(1) - diagonal, 0)
            cost_im = clamp(margin + scores_subtract_diag.max(0) - diagonal, 0)
            # print(cost_s, cost_im)
            margin_loss = cost_s.sum() + cost_im.sum()
            return margin_loss


def cosine_similarity(x1, x2, dim=1, eps=1e-5):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = paddle.sum(x1 * x2, axis=dim)
    w1 = paddle.norm(x1, 2, axis=dim)
    w2 = paddle.norm(x2, 2, axis=dim)
    return clamp(w12 / (w1 * w2), eps).squeeze()


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(self, opt):
        # Build Models
        self.opt = opt

        self.grad_clip = opt.grad_clip

        self.logger = LogCollector()

        self.txt_enc = EncoderText(opt, self.logger)

        self.drop = nn.Dropout(p=0.15)

        # if torch.cuda.is_available():
        #     self.txt_enc.cuda()  # .cuda()
        #     cudnn.benchmark = True

        self.params = list(self.txt_enc.parameters())
        # self.params = params
        # self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        # self.txt_enc, self.optimizer = amp.initialize(self.txt_enc, self.optimizer, opt_level="O1")
        # self.txt_enc = torch.nn.DataParallel(self.txt_enc)
        # Loss and Optimizer
        self.Eiters = 0

    def get_params(self):
        return self.params

    # def to_cuda(self):
    #     self.txt_enc.cuda()  # .cuda()
    #     # cudnn.benchmark = True

    def state_dict(self):
        state_dict = [self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        cur_state_dict = self.txt_enc.state_dict()
        for k, v in state_dict[0].items():
            if k in cur_state_dict:
                cur_state_dict[k] = v
            else:
                print("Key %s not seen in target model's state dict" % k)
        # self.txt_enc.load_state_dict(state_dict[0])
        # paddle's way to load state_dict
        self.txt_enc.set_dict(cur_state_dict)

    def train_start(self):
        """switch to train mode
        """
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.txt_enc.eval()

    def forward_emb(self, images, captions, target_mask, vision_mask, volatile=False, istest=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        # images = Variable(images.float(), volatile=volatile)
        # captions = torch.LongTensor(captions)
        # captions = Variable(captions, volatile=volatile)
        # if torch.cuda.is_available():
        #     images = images.cuda()  # .cuda()
        #     captions = captions.cuda()  # .cuda()
        # Forward

        n_img = images.shape[0]
        n_cap = captions.shape[0]
        if istest:
            images = images.unsqueeze(1).expand((n_img, n_cap, images.shape[1],
                                                 images.shape[2])).reshape((-1, images.shape[1], images.shape[2]))
            # images = images.unsqueeze(1).expand(-1, n_cap, -1,
            #                                     -1).contiguous().view(-1, images.size(1), images.size(2))

            captions = captions.unsqueeze(0).expand((n_img, n_cap, captions.shape[1])).reshape((-1, captions.shape[1]))
            # captions = captions.unsqueeze(0).expand(n_img, -1, -1).contiguous().view(-1, captions.size(1))
        # attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        attention_mask = get_non_pad_mask(captions).squeeze()
        token_type_ids = paddle.zeros_like(attention_mask, dtype='int64')

        # video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()
        video_non_pad_mask = get_non_pad_mask(vision_mask).squeeze()
        if istest:
            video_non_pad_mask = video_non_pad_mask.unsqueeze(1).expand((n_img, n_cap, images.shape[1])).reshape((
                -1, images.shape[1]))

        scores = self.txt_enc(captions, token_type_ids, attention_mask, images, video_non_pad_mask, istest=istest)
        return scores

    def forward_embMLM(self, images, captions, target_mask, vision_mask, gt_labels, vision_labels, boxes, *aux,
                       volatile=False,
                       istest=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        assert len(aux) == 0 or len(aux) == 2 or len(aux) == 4 or len(aux) == 6
        aux_data = *aux,
        # images = Variable(images.float(), volatile=volatile)
        # captions = torch.LongTensor(captions)
        # captions = Variable(captions, volatile=volatile)
        # if torch.cuda.is_available():
        #     images = images.cuda()  # .cuda()
        #     captions = captions.cuda()  # .cuda()
        # Forward
        n_img = images.shape[0]
        n_cap = captions.shape[0]

        # attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        attention_mask = get_non_pad_mask(captions).squeeze()
        token_type_ids = paddle.zeros_like(attention_mask, dtype='int64')

        # video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()
        video_non_pad_mask = get_non_pad_mask(vision_mask).squeeze()
        # corr_vis, total_vis
        loss, corr, total = self.txt_enc(captions, token_type_ids, attention_mask, images,
                                         video_non_pad_mask, gt_labels=gt_labels,
                                         vision_labels=vision_labels, aux=aux_data,
                                         MLM=True)

        self.logger.update('Avg_MLM_corr', corr.sum() / total.sum(), n_cap)
        # self.logger.update('MRM', corr_vis.sum() / total_vis.sum(), n_img)
        self.logger.update('Overall_loss', loss.mean().numpy()[0])
        return loss

    # def forward_loss(self, img_emb, cap_emb, cap_len, text_non_pad_mask, text_slf_attn_mask, img_non_pad_mask,
    #                  img_slf_attn_mask, **kwargs):
    #     """Compute the loss given pairs of image and caption embeddings
    #     """
    #     scores = self.cross_att(img_emb, cap_emb, cap_len, text_non_pad_mask, text_slf_attn_mask, img_non_pad_mask,
    #                             img_slf_attn_mask)
    #     loss = self.criterion(scores)
    #     self.logger.update('Le', loss.numpy()[0], scores.shape[0])
    #     return loss

    def train_emb(self, images, captions, target_mask, vision_mask, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)

        # measure accuracy and record loss
        scores = self.forward_emb(images, captions, target_mask, vision_mask).mean()
        # measure accuracy and record loss

        # self.optimizer.zero_grad()
        if scores is not None:
            # print(scores)
            self.logger.update('Loss', scores.numpy()[0])
            return scores
        else:
            return None
        # compute gradient and do SGD step
        # loss.backward()
        # if self.opt.fp16:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        # if self.grad_clip > 0:
        #     clip_grad_norm(self.params, self.grad_clip)
        # self.optimizer.step()

    def train_embMLM(self, images, captions, text_mask, vision_mask, gt_labels, vision_labels, boxes, *aux):
        """
        One training step given images and captions and other possible input text. If args
        # aux texts are pure text, not in paired with any images
        # aux should be in the form of (ids, mlm_labels) or (ids_concat, mlm_labels, ids1, ids2)
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        # gt_labels = gt_labels.long().cuda()
        # measure accuracy and record loss for image-caption data
        loss = self.forward_embMLM(images, captions, text_mask, vision_mask, gt_labels, vision_labels, boxes, *aux)
        # measure accuracy and record loss
        # self.optimizer.zero_grad()
        loss = loss.mean()
        return loss
        # compute gradient and do SGD step
        # loss.backward()
        # if self.opt.fp16:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        # if self.grad_clip > 0:
        #     clip_grad_norm(self.params, self.grad_clip)
        # self.optimizer.step()
