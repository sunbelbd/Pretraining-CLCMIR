import datetime
import logging
import random
import sys
from bisect import bisect
from torch.utils.data import Dataset

sys.path.append('/mnt/home/hongliangfei/research/emnlp2020-vlbert-base')

from external.pytorch_pretrained_bert import BertTokenizer
from copy import deepcopy
import os
import pickle


class ParallelCorpus(Dataset):
    def __init__(self, ann_file, pretrained_model_name, data_path, with_mlm_task=True, tokenizer=None,
                 seq_len=64, min_seq_len=64, encoding="utf-8", on_memory=True, cache_db=False, **kwargs):
        assert on_memory, "only support on_memory mode!"

        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained(pretrained_model_name)
        self.vocab = self.tokenizer.vocab
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.data_path = data_path
        self.on_memory = on_memory
        self.with_mlm_task = with_mlm_task
        self.ann_file = ann_file
        self.encoding = encoding
        self.test_mode = False
        self.cache_db = cache_db

        # load samples into memory
        if on_memory:
            print("Begin loading parallel corpus", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.corpus = self.load_corpus()
            print("End loading parallel corpus", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self.total_length = 0
        self.cum_corpus_lengths = []
        for corpus in self.corpus:
            self.total_length += len(corpus)
            self.cum_corpus_lengths.append(self.total_length)

    def load_corpus(self):
        """
        load all parallel data
        :return:
        """
        # check if cached file is available, if so, load it. Otherwise, read from text file
        if 'valid' in self.ann_file:
            partition = 'valid'
        elif 'test' in self.ann_file:
            partition = 'test'
        else:
            partition = 'train'
        cache_file = os.path.join(self.data_path, "parallel_corpus_" + partition + ".pickle")
        print("Desired cache file location %s " % cache_file)
        if os.path.exists(cache_file):
            return pickle.load(open(cache_file, "rb"))
        else:
            corpus = []
            for ann_file in self.ann_file.split('+'):
                print("loading parallel data %s now!" % ann_file)
                max_lines, i = 10000000, 0
                corpus_sub = []
                with open(os.path.join(self.data_path, ann_file), 'r', encoding=self.encoding, errors='ignore') as f:
                    for l in f.readlines():
                        ls = l.strip('\n').strip('\r').strip('\n')
                        if len(ls) == 0:
                            continue
                        i += 1
                        # ls = ls[:min(len(ls), 2*self.seq_len)]
                        s1, s2 = ls.split("\t")
                        s1 = s1[:min(len(s1), self.seq_len // 2)]
                        s2 = s2[:min(len(s2), self.seq_len) // 2]
                        ids1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s1))
                        ids2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s2))
                        corpus_sub.append((ids1, ids2))
                        if i % 10000 == 0:
                            print("Loaded %d lines now" % i)
                        if i >= max_lines:  break
                corpus.append(corpus_sub)
            if self.cache_db and not os.path.exists(cache_file):
                pickle.dump(corpus, open(cache_file, "wb"))
            return corpus

    @property
    def data_names(self):
        return ['text_concat', 'tlm_labels', 'text_en', 'text_other']

    def __len__(self):
        return self.total_length

    def _fetch_item(self, idx):
        dataset_idx = bisect.bisect_right(self.cum_corpus_lengths, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_corpus_lengths[dataset_idx - 1]
        return self.corpus[dataset_idx][sample_idx]

    def __getitem__(self, item):
        """
        split parallel sentences by token "\t"
        :param item: a line with a parallel sentence
        :return:
        """
        # raw1, raw2 = self.corpus[item].split("\t")
        # tokens_unmaked1 = self.tokenizer.tokenize(raw1)
        # tokens_unmaked2 = self.tokenizer.tokenize(raw2)
        # ids_unmask1 = self.tokenizer.convert_tokens_to_ids(tokens_unmaked1)
        # ids_unmask2 = self.tokenizer.convert_tokens_to_ids(tokens_unmaked2)

        # ids_unmask1, ids_unmask2 = self.corpus[item]
        ids_unmask1, ids_unmask2 = self._fetch_item(item)

        if self.with_mlm_task:
            # two step wordpiece tokenizer
            # tokens1 = self.tokenizer.basic_tokenizer.tokenize(raw1)
            # tokens2 = self.tokenizer.basic_tokenizer.tokenize(raw2)
            # tokens_masked1, mlm_labels1 = self.random_word_wwm(tokens1)
            # tokens_masked2, mlm_labels2 = self.random_word_wwm(tokens2)
            ids_mask1, mlm_labels1 = self.random_word_wwm(ids_unmask1)
            ids_mask2, mlm_labels2 = self.random_word_wwm(ids_unmask2)
            # convert token to its vocab id
            # ids_mask1 = self.tokenizer.convert_tokens_to_ids(tokens_masked1)
            # ids_mask2 = self.tokenizer.convert_tokens_to_ids(tokens_masked2)
        else:
            mlm_labels1 = [-1] * len(ids_unmask1)
            mlm_labels2 = [-1] * len(ids_unmask2)

        if len(ids_unmask1) > self.seq_len:
            ids_unmask1 = ids_unmask1[:self.seq_len]
        if len(ids_unmask2) > self.seq_len:
            ids_unmask2 = ids_unmask2[:self.seq_len]

        ids1 = deepcopy(ids_mask1 if self.with_mlm_task else ids_unmask1)
        ids2 = deepcopy(ids_mask2 if self.with_mlm_task else ids_unmask2)
        # truncate: reset positions for text_other
        if len(ids1) + len(ids2) > self.seq_len - 4:
            text1_len_keep = len(ids1)
            text2_len_keep = len(ids2)
            while (text1_len_keep + text2_len_keep) > self.seq_len - 4:
                if text2_len_keep > text1_len_keep:
                    text2_len_keep -= 1
                else:
                    text1_len_keep -= 1
            ids1 = ids1[:text1_len_keep]
            ids2 = ids2[:text2_len_keep]

        mlm_labels = [-1] + mlm_labels1 + [-1, -1] + mlm_labels2 + [-1]

        ids = [self.vocab['[CLS]']] + ids1 + [self.vocab['[SEP]']] + \
              [self.vocab['[SEP]']] + ids2 + [self.vocab['[SEP]']]

        return ids, mlm_labels, ids_unmask1, ids_unmask2

    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            # sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # for sub_token in sub_tokens:
                    output_tokens.append(self.tokenizer.vocab["[MASK]"])
                # 10% randomly change token to random token
                elif prob < 0.9:
                    # for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
                else:
                    # for sub_token in sub_tokens:
                    output_tokens.append(token)

                    # append current token to output (we will predict these later)
                # for sub_token in sub_tokens:
                try:
                    output_label.append(token)
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.tokenizer.vocab["[UNK]"])
                    logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                # for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(token)
                output_label.append(-1)

        ## if no word masked, random choose a word to mask
        # if all([l_ == -1 for l_ in output_label]):
        #    choosed = random.randrange(0, len(output_label))
        #    output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return output_tokens, output_label

    # def random_word_wwm(self, tokens):
    #     output_tokens = []
    #     output_label = []
    #
    #     for i, token in enumerate(tokens):
    #         sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
    #         prob = random.random()
    #         # mask token with 15% probability
    #         if prob < 0.15:
    #             prob /= 0.15
    #
    #             # 80% randomly change token to mask token
    #             if prob < 0.8:
    #                 for sub_token in sub_tokens:
    #                     output_tokens.append("[MASK]")
    #             # 10% randomly change token to random token
    #             elif prob < 0.9:
    #                 for sub_token in sub_tokens:
    #                     output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
    #                     # -> rest 10% randomly keep current token
    #             else:
    #                 for sub_token in sub_tokens:
    #                     output_tokens.append(sub_token)
    #
    #                     # append current token to output (we will predict these later)
    #             for sub_token in sub_tokens:
    #                 try:
    #                     output_label.append(self.tokenizer.vocab[sub_token])
    #                 except KeyError:
    #                     # For unknown words (should not occur with BPE vocab)
    #                     output_label.append(self.tokenizer.vocab["[UNK]"])
    #                     logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
    #         else:
    #             for sub_token in sub_tokens:
    #                 # no masking token (will be ignored by loss function later)
    #                 output_tokens.append(sub_token)
    #                 output_label.append(-1)
    #
    #     return output_tokens, output_label


if __name__ == '__main__':
    ann_files = 'en-de.train+en-fr.train+en-ja.train+en-ru.train+en-zh.train'
    pretrained_model_name = '/mnt/home/hongliangfei/research/emnlp2020-vlbert-base/model/pretrained_model/bert-base-multilingual-uncased'
    data_path = '/mnt/data/hongliangfei/emnlp2020-vlbert-base/data/para_corpus'
    pc = ParallelCorpus(ann_files, pretrained_model_name, data_path, cache_db=True)
    print("Done")
