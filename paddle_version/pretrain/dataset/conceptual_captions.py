import jsonlines
import logging
import numpy as np
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ConceptualCaptionsDataset(Dataset):
    def __init__(self, ann_file, image_set, root_path, data_path, seq_len=64,
                 with_precomputed_visual_feat=False, mask_raw_pixels=True,
                 with_rel_task=True, with_mlm_task=True, with_mvrc_task=True,
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False,
                 aspect_grouping=False, **kwargs):
        """
        Conceptual Captions Dataset: process both caption and image data

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to CC dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(ConceptualCaptionsDataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        assert not test_mode

        annot = {'train': 'train_frcnn.json',
                 'val': 'val_frcnn.json'}

        self.seq_len = seq_len
        self.max_vision = 16
        self.max_text = self.seq_len - self.max_vision
        self.with_rel_task = with_rel_task
        self.with_mlm_task = with_mlm_task
        self.with_mvrc_task = with_mvrc_task
        self.data_path = data_path
        self.root_path = root_path
        self.ann_file = os.path.join(data_path, annot[image_set])
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.mask_raw_pixels = mask_raw_pixels
        self.image_set = image_set
        self.transform = transform
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-multilingual-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)
        self.is_uncased = True if pretrained_model_name is not None and 'uncased' in pretrained_model_name else False

        # read image caption and frcnn file record file: {train/val]._frcnn.json
        # {"caption": ["a", "very", "typical", "bus", "station"], "image": "train_image.zip@/00000000.jpg",
        # "frcnn": "train_frcnn.zip@/00000000.json"}
        self.database = list(jsonlines.open(self.ann_file))
        if not self.zip_mode:
            for i, idb in enumerate(self.database):
                self.database[i]['frcnn'] = idb['frcnn'].replace('.zip@', '') \
                    .replace('.0', '').replace('.1', '').replace('.2', '').replace('.3', '')
                self.database[i]['image'] = idb['image'].replace('.zip@', '')

    @property
    def data_names(self):
        # Tan's list: image, target, target_mask, vision_mask, mlm_labels, vision_labels
        # My list: boxes_features, text, text_mask, vision_mask, mlm_labels, mvrc_labels, boxes
        # return ['image', 'boxes', 'im_info', 'text',
        #         'relationship_label', 'mlm_labels', 'mvrc_ops', 'mvrc_labels']
        return ['box_features', 'text', 'text_mask', 'vision_mask',
                'mlm_labels', 'mvrc_labels', 'boxes']

    def __getitem__(self, index):
        # idb format: {"caption": ["a", "very", "typical", "bus", "station"], "image": "train_image.zip@/00000000.jpg",
        # "frcnn": "train_frcnn.zip@/00000000.json"}
        idb = self.database[index]

        # image bbox feature data saved in {train/valid}_frcnn.zip
        # keys are: ['image_id', 'boxes', 'classes', 'attrs', 'image_w', 'num_boxes', 'image_h', 'features']
        # coco precomp: ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
        # change to cache[] to speedup
        frcnn_data = self._load_npz(os.path.join(self.data_path, idb['frcnn'].replace("json", "npz")))
        # print(frcnn_data)
        #  obj bbox coords [num_bbox, 4]
        boxes = frcnn_data['boxes'].reshape((frcnn_data['num_boxes'], -1)).astype(np.float32)
        # obj category prob score: [num_bbox, num_obj_categories]
        boxes_cls_scores = frcnn_data['classes'].reshape((frcnn_data['num_boxes'], -1))
        boxes_max_conf = boxes_cls_scores.max(axis=1)
        # rearrange boxes and cls_scores in descending order of detected obj_prob
        inds = np.argsort(boxes_max_conf)[::-1]  # [num_bbox]
        boxes = boxes[inds]
        boxes_cls_scores = boxes_cls_scores[inds]
        boxes = torch.as_tensor(boxes)

        w0, h0 = int(frcnn_data['image_w']), int(frcnn_data['image_h'])
        # [num_bboxes, frcnn_hidden_dim=2048 or other number]
        boxes_features = frcnn_data['features'].reshape((frcnn_data['num_boxes'], -1))
        boxes_features = boxes_features[inds]
        boxes_features = torch.as_tensor(boxes_features)

        # print(w0, h0, index)
        im_info = torch.as_tensor([w0, h0, 1.0, 1.0, index])

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        assert w > 0 and h > 0
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        caption = ' '.join(idb['caption'])
        if self.is_uncased:
            caption = caption.lower()
        # Task #2: Masked Language Modeling
        if self.with_mlm_task:
            caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
            caption_tokens, mlm_labels = self.random_word_wwm(caption_tokens)
        else:
            caption_tokens = self.tokenizer.tokenize(caption)
            mlm_labels = [-1] * len(caption_tokens)
        text_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        mlm_labels = [-1] + mlm_labels + [-1]

        # Task #3: Masked Visual Region Classification
        if self.with_mvrc_task:
            mvrc_ops, mvrc_labels = self.random_mask_region(boxes_cls_scores)
            boxes_features[mvrc_ops == 1] = 0.0
            assert len(mvrc_ops) == boxes.shape[0], \
                "Error: mvrc_ops have length {}, expected {}!".format(len(mvrc_ops), boxes.shape[0])
            assert len(mvrc_labels) == boxes.shape[0], \
                "Error: mvrc_labels have length {}, expected {}!".format(len(mvrc_labels), boxes.shape[0])
        else:
            mvrc_ops = [0] * boxes.shape[0]
            mvrc_labels = [np.zeros_like(boxes_cls_scores[0])] * boxes.shape[0]

        # [num_bboxes, num_obj_categories]
        mvrc_labels = np.stack(mvrc_labels, axis=0)

        text = self.tokenizer.convert_tokens_to_ids(text_tokens)

        # truncate seq to max len, separately truncate text and image boxes
        # self.max_vision self.max_text
        if len(text) > self.max_text:
            text = text[:self.max_text]
            # input_mask = input_mask[:self.max_seq_length]
            mlm_labels = mlm_labels[:self.max_text]

        if len(boxes) > self.max_vision:
            boxes = boxes[:self.max_vision]
            boxes_features = boxes_features[:self.max_vision]
            mvrc_ops = mvrc_ops[:self.max_vision]
            mvrc_labels = mvrc_labels[:self.max_vision]

        # if len(text) + len(boxes) > self.seq_len:
        #     text_len_keep = len(text)
        #     box_len_keep = len(boxes)
        #     while (text_len_keep + box_len_keep) > self.seq_len:
        #         if box_len_keep > text_len_keep:
        #             box_len_keep -= 1
        #         else:
        #             text_len_keep -= 1
        #     boxes = boxes[:box_len_keep]
        #     boxes_features = boxes_features[:box_len_keep]
        #     text = text[:text_len_keep]
        #     mlm_labels = mlm_labels[:text_len_keep]
        #     mvrc_ops = mvrc_ops[:box_len_keep]
        #     mvrc_labels = mvrc_labels[:box_len_keep]

        text_mask = [1] * len(text)
        vision_mask = [1] * boxes_features.size(0)
        # # image, target, target_mask, vision_mask, mlm_labels, vision_labels
        # return image, boxes, im_info, text, mlm_labels, mvrc_ops, mvrc_labels
        return boxes_features, text, text_mask, vision_mask, mlm_labels, mvrc_labels, boxes

    # def random_word(self, tokens):
    #     output_label = []
    #
    #     for i, token in enumerate(tokens):
    #         prob = random.random()
    #         # mask token with 15% probability
    #         if prob < 0.15:
    #             prob /= 0.15
    #
    #             # 80% randomly change token to mask token
    #             if prob < 0.8:
    #                 tokens[i] = "[MASK]"
    #
    #             # 10% randomly change token to random token
    #             elif prob < 0.9:
    #                 tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
    #
    #             # -> rest 10% randomly keep current token
    #
    #             # append current token to output (we will predict these later)
    #             try:
    #                 output_label.append(self.tokenizer.vocab[token])
    #             except KeyError:
    #                 # For unknown words (should not occur with BPE vocab)
    #                 output_label.append(self.tokenizer.vocab["[UNK]"])
    #                 logging.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
    #         else:
    #             # no masking token (will be ignored by loss function later)
    #             output_label.append(-1)
    #
    #     # if no word masked, random choose a word to mask
    #     if self.force_mask:
    #         if all([l_ == -1 for l_ in output_label]):
    #             choosed = random.randrange(0, len(output_label))
    #             output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]
    #
    #     return tokens, output_label

    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.9:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)

                        # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    try:
                        output_label.append(self.tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        ## if no word masked, random choose a word to mask
        # if all([l_ == -1 for l_ in output_label]):
        #    choosed = random.randrange(0, len(output_label))
        #    output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return output_tokens, output_label

    def random_mask_region(self, regions_cls_scores):
        """
        :param regions_cls_scores:
        :return:
        output_op [num_bboxes]: 0/1, 0 indicating no masking; 1 otherwise
        output_label [num_bboxes, num_obj_categories]:
        cls_scores if output_op is 1; otherwise 0 vector for 0 output_op
        """
        num_regions, num_classes = regions_cls_scores.shape
        output_op = []
        output_label = []
        for k, cls_scores in enumerate(regions_cls_scores):
            prob = random.random()
            # mask region with 15% probability
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.9:
                    # 90% randomly replace appearance feature by "MASK"
                    output_op.append(1)
                else:
                    # -> rest 10% randomly keep current appearance feature
                    output_op.append(0)

                # append class of region to output (we will predict these later)
                output_label.append(cls_scores)
            else:
                # no masking region (will be ignored by loss function later)
                output_op.append(0)
                # output_label.append(np.zeros_like(cls_scores))
                output_label.append(np.zeros_like(cls_scores))

        # # if no region masked, random choose a region to mask
        # if all([op == 0 for op in output_op]):
        #     choosed = random.randrange(0, len(output_op))
        #     output_op[choosed] = 1
        #     output_label[choosed] = regions_cls_scores[choosed]

        return output_op, output_label

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        """
        Add cache mode to avoid IO bottle neck
        :param path:
        :return:
        """
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_npz(self, path):
        """
        Add cache mode to avoid IO bottle neck
        :param path:
        :return:
        """
        npzfile = np.load(path)
        return dict(npzfile)
