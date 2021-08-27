import paddle

from paddle_version.pretrain.dataset.clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        """
        process different input scenarios.
        ['image', 'boxes', 'im_info', 'text', 'index']
        :param batch:
        :return:
        """
        # print(self.data_names)
        if not isinstance(batch, list):
            batch = list(batch)
        if 'boxes' in self.data_names:
            max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        if 'text' in self.data_names:
            max_text_length = max([len(data[self.data_names.index('text')]) for data in batch])

        for i, ibatch in enumerate(batch):
            out = {}
            if 'boxes' in self.data_names:
                boxes = ibatch[self.data_names.index('boxes')]
                box_features = ibatch[self.data_names.index('box_features')]
                out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)
                out['box_features'] = clip_pad_boxes(box_features, max_boxes, pad=0)

            if 'vision_mask' in self.data_names:
                vision_mask = ibatch[self.data_names.index('vision_mask')]
                out['vision_mask'] = clip_pad_1d(vision_mask, max_boxes, pad=0)

            if 'text' in self.data_names:
                text = ibatch[self.data_names.index('text')]
                out['text'] = clip_pad_1d(text, max_text_length, pad=0)

            if 'text_mask' in self.data_names:
                text_mask = ibatch[self.data_names.index('text_mask')]
                out['text_mask'] = clip_pad_1d(text_mask, max_text_length, pad=0)

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

        return out_tuple
