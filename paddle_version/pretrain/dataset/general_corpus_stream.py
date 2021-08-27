import logging
import random
import sys
import torch
from itertools import chain, cycle

sys.path.append('/mnt/home/hongliangfei/research/emnlp2020-vlbert-base')

from pretrain.data.collate_batch import BatchCollator
from external.pytorch_pretrained_bert import BertTokenizer
import os


class GeneralCorpusStream(torch.utils.data.IterableDataset):
    def __init__(self, ann_file, pretrained_model_name, data_path, tokenizer=None, seq_len=64, min_seq_len=64,
                 encoding="utf-8", on_memory=True, cache_db=False, **kwargs):

        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained(pretrained_model_name)
        self.vocab = self.tokenizer.vocab
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.data_path = data_path
        self.on_memory = on_memory
        self.ann_file = ann_file
        self.encoding = encoding
        self.test_mode = False
        self.cache_db = cache_db

        # load samples into memory
        # if on_memory:
        #     self.corpus = self.load_corpus()

        self.ann_file_list = self.ann_file.split('+')

    def get_stream(self):
        prob = random.random()
        if prob < 0.5:
            random.shuffle(self.ann_file_list)
        return chain.from_iterable(map(self.parse_file, cycle(self.ann_file_list)))

    def parse_file(self, ann_file):
        """
        Process one line and return ids, mlm_labels
        :param ann_file:
        :return:
        """
        with open(os.path.join(self.data_path, ann_file), 'r', encoding=self.encoding, errors='ignore') as f:
            for l in f.readlines():
                ls = l.strip('\n').strip('\r').strip('\n')
                if len(ls) == 0:
                    continue
                tokens = self.tokenizer.basic_tokenizer.tokenize(ls)
                tokens, mlm_labels = self.random_word_wwm(tokens)
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                # truncate
                if len(ids) > self.seq_len:
                    ids = ids[:self.seq_len]
                    mlm_labels = mlm_labels[:self.seq_len]
                yield ids, mlm_labels

    @property
    def data_names(self):
        return ['text', 'mlm_labels']

    def __iter__(self):
        return self.get_stream()

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(self.tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.tokenizer.vocab["[UNK]"])
                    logging.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        # # if no word masked, random choose a word to mask
        # if self.force_mask:
        #     if all([l_ == -1 for l_ in output_label]):
        #         choosed = random.randrange(0, len(output_label))
        #         output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return tokens, output_label


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset) // worker_info.num_workers
    dataset.data = dataset.data[worker_id * split_size:(worker_id + 1) * split_size]


if __name__ == '__main__':
    ann_files = 'de.valid+en.valid+zh.valid'
    pretrained_model_name = '/mnt/home/hongliangfei/research/emnlp2020-vlbert-base/model/pretrained_model/bert-base-multilingual-uncased'
    data_path = '/mnt/data/hongliangfei/emnlp2020-vlbert-base/data/mono_corpus/'
    dataset = GeneralCorpusStream(ann_files, pretrained_model_name, data_path)
    collator = BatchCollator(dataset=dataset, append_ind=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=16,
                                             num_workers=2,
                                             collate_fn=collator,
                                             drop_last=True,
                                             worker_init_fn=worker_init_fn)
    # print("length of dataset is: %d " % len(dataset))
    for i, batch in enumerate(dataloader):
        if i >= 50:
            break
        print(batch)
    print("Done")
