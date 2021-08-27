import base64
import json
import numpy as np
import os
import paddle
from PIL import Image
from paddle.io import Dataset
from paddlenlp.transformers import BertTokenizer


class XLRETCOCODataset(Dataset):
    def __init__(self, ann_file, image_set, root_path, data_path, seq_len=64,
                 with_precomputed_visual_feat=False, mask_raw_pixels=True,
                 transform=None, test_mode=False, cache_mode=False,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False,
                 aspect_grouping=False, **kwargs):
        """
       Coco or flicker caption Dataset: process both caption and image data

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to CC dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(XLRETCOCODataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        self.seq_len = seq_len
        self.data_path = data_path
        self.root_path = root_path
        self.ann_file = os.path.join(data_path, ann_file)
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.mask_raw_pixels = mask_raw_pixels
        self.image_set = image_set
        self.transform = transform
        self.test_mode = test_mode
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.is_uncased = True if pretrained_model_name is not None and 'uncased' in pretrained_model_name else False

        # read image caption and mapping id file
        self.captions = self.load_captions()
        self.img_ids = [int(line.strip()) for line in open(self.ann_file.replace("caps", "ids"), encoding='utf-8')]
        self.img_feats = None
        if self.test_mode:
            # return a list of feat_dict
            self.img_ids = self._filer_duplicate_ids(self.img_ids)
            self.img_feats = self.load_img_feat(self.img_ids)
        if len(self.captions) != len(self.img_ids):
            self.im_div = 5
        else:
            self.im_div = 1
        if self.aspect_grouping:
            assert False, "not support aspect grouping currently!"
            self.group_ids = self.group_aspect(self.database)

        # print('mask_raw_pixels: ', self.mask_raw_pixels)

    @property
    def data_names(self):
        # images, captions, target_mask, vision_mask, ids
        # return ['image', 'boxes', 'im_info', 'text', 'index']
        return ['box_features', 'text', 'text_mask', 'vision_mask',
                'index', 'boxes']

    def __getitem__(self, index):
        # idb format: {"caption": ["a", "very", "typical", "bus", "station"], "image": "train_image.zip@/00000000.jpg",
        # "frcnn": "train_frcnn.zip@/00000000.json"}
        img_index = index // self.im_div
        img_id = self.img_ids[img_index]
        text = self.captions[index]
        if self.test_mode:
            feat_dict = self.img_feats[img_index]
            boxes_features, boxes, im_info = feat_dict['box_features'], feat_dict['boxes'], feat_dict['im_info']
        else:
            # image bbox feature data saved in {train/valid}_frcnn.zip
            # keys are: ['image_id', 'boxes', 'classes', 'attrs', 'image_w', 'num_boxes', 'image_h', 'features']
            # coco precomp: ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
            # change to cache[] to speedup
            frcnn_data = self._load_json(os.path.join(self.data_path, "precomputed/{:d}.json".format(img_id)))
            #  obj bbox coords [num_bbox, 4]
            boxes = np.frombuffer(self.b64_decode(frcnn_data['boxes']),
                                  dtype=np.float32).reshape((frcnn_data['num_boxes'], -1)).copy()
            if len(boxes) == 0:
                print("No bounding boxes detected in %s" % "./precomputed/{:d}.json".format(img_id))
            # obj category prob score: [num_bbox, num_obj_categories]
            if 'classes' in frcnn_data:
                boxes_cls_scores = np.frombuffer(self.b64_decode(frcnn_data['classes']),
                                                 dtype=np.float32).reshape((frcnn_data['num_boxes'], -1)).copy()
                boxes_max_conf = boxes_cls_scores.max(axis=1)
                # rearrange boxes and cls_scores in descending order of detected obj_prob
                inds = np.argsort(boxes_max_conf)[::-1]  # [num_bbox]
                boxes = boxes[inds]
                boxes_cls_scores = boxes_cls_scores[inds]

            w0, h0 = frcnn_data['image_w'], frcnn_data['image_h']
            # [num_bboxes, frcnn_hidden_dim=2048 or other number]
            boxes_features = np.frombuffer(self.b64_decode(frcnn_data['features']),
                                           dtype=np.float32).reshape((frcnn_data['num_boxes'], -1)).copy()
            if 'classes' in frcnn_data:
                boxes_features = boxes_features[inds]
            boxes_features = paddle.to_tensor(boxes_features)

            # transform: normalization etc
            im_info = np.array([w0, h0, 1.0, 1.0, index])

            # clamp boxes
            w = im_info[0]
            h = im_info[1]
            assert w > 0 and h > 0
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(min=0, max=w - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(min=0, max=h - 1)

            boxes = paddle.to_tensor(boxes)

        # truncate seq to max len
        if len(text) + len(boxes) > self.seq_len:
            text_len_keep = len(text)
            box_len_keep = len(boxes)
            while (text_len_keep + box_len_keep) > self.seq_len:
                if box_len_keep > text_len_keep:
                    box_len_keep -= 1
                else:
                    text_len_keep -= 1
            boxes = boxes[:box_len_keep]
            text = text[:text_len_keep]
        text_mask = [1] * len(text)
        vision_mask = [1] * boxes_features.shape[0]
        # return image, boxes, im_info, text, index
        return boxes_features, text, text_mask, vision_mask, index, boxes

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def __len__(self):
        return len(self.captions)

    def load_captions(self):
        caption_ids = []
        fp = open(self.ann_file, encoding='utf-8')
        for i, cap in enumerate(fp):
            cap = cap.strip()
            if self.is_uncased:
                cap = cap.lower()
            caption_tokens = self.tokenizer.tokenize(cap)
            text_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
            text = self.tokenizer.convert_tokens_to_ids(text_tokens)
            caption_ids.append(text)
        fp.close()
        return caption_ids

    @staticmethod
    def _filer_duplicate_ids(img_ids):
        filter_ids = []
        prev_id = -1
        for id in img_ids:
            if id != prev_id:
                filter_ids.append(id)
            prev_id = id
        return filter_ids

    def load_img_feat(self, img_ids):
        """
        :param img_ids: all image ids, load their data into list of dict format
        :return:
        """
        img_feats = []
        print("Loading image features from %d files" % len(img_ids))
        for index, img_id in enumerate(img_ids):
            if index % 200 == 0:
                print("Loading %dth image feature now" % index)
            frcnn_data = self._load_json(os.path.join(self.data_path, "precomputed/{:d}.json".format(img_id)))
            #  obj bbox coords [num_bbox, 4]
            boxes = np.frombuffer(self.b64_decode(frcnn_data['boxes']),
                                  dtype=np.float32).reshape((frcnn_data['num_boxes'], -1)).copy()
            if len(boxes) == 0:
                print("No bounding boxes detected in %s" % "./precomputed/{:d}.json".format(img_id))
            # obj category prob score: [num_bbox, num_obj_categories]
            if 'classes' in frcnn_data:
                boxes_cls_scores = np.frombuffer(self.b64_decode(frcnn_data['classes']),
                                                 dtype=np.float32).reshape((frcnn_data['num_boxes'], -1)).copy()
                boxes_max_conf = boxes_cls_scores.max(axis=1)
                # rearrange boxes and cls_scores in descending order of detected obj_prob
                inds = np.argsort(boxes_max_conf)[::-1]  # [num_bbox]
                boxes = boxes[inds]
                boxes_cls_scores = boxes_cls_scores[inds]

            w0, h0 = frcnn_data['image_w'], frcnn_data['image_h']
            # [num_bboxes, frcnn_hidden_dim=2048 or other number]
            boxes_features = np.frombuffer(self.b64_decode(frcnn_data['features']),
                                           dtype=np.float32).reshape((frcnn_data['num_boxes'], -1)).copy()
            if 'classes' in frcnn_data:
                boxes_features = boxes_features[inds]
            boxes_features = paddle.to_tensor(boxes_features)

            # transform: normalization etc
            im_info = np.array([w0, h0, 1.0, 1.0, index])

            # clamp boxes
            w = im_info[0]
            h = im_info[1]
            assert w > 0 and h > 0
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(min=0, max=w - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(min=0, max=h - 1)

            boxes = paddle.to_tensor(boxes)
            # 'box_features', 'im_info', 'boxes'
            img_feats.append({"boxes": boxes, "box_features": boxes_features, "im_info": im_info})
        return img_feats

    def _load_image(self, path):
        """
        Add cache mode to avoid IO bottle neck
        :param path:
        :return:
        """
        return Image.open(path).convert('RGB')

    def _load_npz(self, path):
        """
        Add cache mode to avoid IO bottle neck
        :param path:
        :return:
        """
        npzfile = np.load(path)
        return dict(npzfile)

    def _load_json(self, path):
        """
        Add cache mode to avoid IO bottle neck
        :param path:
        :return:
        """
        with open(path, 'r') as f:
            return json.load(f)


def main():
    # Save preloaded img features
    pass


if __name__ == '__main__':
    main()
