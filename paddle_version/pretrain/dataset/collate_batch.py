import paddle

from .clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        """
        process different input scenarios.
        cc data fields
        ['box_features', 'text', 'text_mask', 'vision_mask', 'mlm_labels', 'mvrc_labels', 'boxes']
        For
        For parallel corpus, data names are: ['text_en', 'text_other', 'text_concat']
        :param batch:
        :return:
        """
        # print(self.data_names)
        if not isinstance(batch, list):
            batch = list(batch)

        max_pair_text_length, max_single_text_length = 0, 0
        if 'boxes' in self.data_names:
            max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        if 'text' in self.data_names:
            max_pair_text_length = max(max_pair_text_length,
                                       max([len(data[self.data_names.index('text')]) for data in batch]))
        # ['text_en', 'text_other', 'text_concat']
        if 'text_en' in self.data_names:
            max_single_text_length = max(max_single_text_length,
                                         max([len(data[self.data_names.index('text_en')]) for data in batch]))
        if 'text_other' in self.data_names:
            max_single_text_length = max(max_single_text_length,
                                         max([len(data[self.data_names.index('text_other')]) for data in batch]))
        if 'text_concat' in self.data_names:
            max_pair_text_length = max(max_pair_text_length,
                                       max([len(data[self.data_names.index('text_concat')]) for data in batch]))

        for i, ibatch in enumerate(batch):
            out = {}

            if 'boxes' in self.data_names:
                boxes = ibatch[self.data_names.index('boxes')]
                box_features = ibatch[self.data_names.index('box_features')]
                # box coord padding symbol is -2, box_feature padding is 0
                out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)
                out['box_features'] = clip_pad_boxes(box_features, max_boxes, pad=0)

            if 'vision_mask' in self.data_names:
                vision_mask = ibatch[self.data_names.index('vision_mask')]
                out['vision_mask'] = clip_pad_1d(vision_mask, max_boxes, pad=0)

            if 'text' in self.data_names:
                text = ibatch[self.data_names.index('text')]
                out['text'] = clip_pad_1d(text, max_pair_text_length, pad=0)

            if 'text_mask' in self.data_names:
                text_mask = ibatch[self.data_names.index('text_mask')]
                out['text_mask'] = clip_pad_1d(text_mask, max_pair_text_length, pad=0)

            if 'text_en' in self.data_names:
                text = ibatch[self.data_names.index('text_en')]
                out['text_en'] = clip_pad_1d(text, max_single_text_length, pad=0)

            if 'text_other' in self.data_names:
                text = ibatch[self.data_names.index('text_other')]
                out['text_other'] = clip_pad_1d(text, max_single_text_length, pad=0)

            if 'text_concat' in self.data_names:
                text = ibatch[self.data_names.index('text_concat')]
                out['text_concat'] = clip_pad_1d(text, max_pair_text_length, pad=0)

            if 'mlm_labels' in self.data_names:
                mlm_labels = ibatch[self.data_names.index('mlm_labels')]
                out['mlm_labels'] = clip_pad_1d(mlm_labels, max_pair_text_length, pad=-1)

            if 'tlm_labels' in self.data_names:
                tlm_labels = ibatch[self.data_names.index('tlm_labels')]
                out['tlm_labels'] = clip_pad_1d(tlm_labels, max_pair_text_length, pad=-1)

            if 'mvrc_labels' in self.data_names:
                mvrc_labels = ibatch[self.data_names.index('mvrc_labels')]
                out['mvrc_labels'] = clip_pad_boxes(mvrc_labels, max_boxes, pad=0)

            # copy other data tuples w/o special care
            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                out[name] = paddle.to_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (paddle.to_tensor(i, dtype='int64'),)

        out_tuple = ()
        for items in zip(*batch):
            if items[0] is None:
                out_tuple += (None,)
            else:
                # if len(self.data_names) == 2:
                #     print(items)
                out_tuple += (paddle.stack(tuple(items), axis=0),)

        # print(out_tuple)
        return out_tuple
