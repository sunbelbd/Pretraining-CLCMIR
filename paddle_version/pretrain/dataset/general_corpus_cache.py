import datetime
import logging
import random
import sys
from bisect import bisect
from torch.utils.data import Dataset

sys.path.append('/mnt/home/hongliangfei/research/emnlp2020-vlbert-base')

from external.pytorch_pretrained_bert import BertTokenizer
import os
import pickle


class GeneralCorpus(Dataset):
    def __init__(self, ann_file, pretrained_model_name, data_path, tokenizer=None, seq_len=64, min_seq_len=64,
                 encoding="utf-8", on_memory=True, cache_db=False,
                 **kwargs):
        assert on_memory, "only support on_memory mode!"

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
        if on_memory:
            print("Begin loading mono corpus", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.corpus = self.load_corpus()
            print("End loading mono corpus", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self.total_length = 0
        self.cum_corpus_lengths = []
        for corpus in self.corpus:
            self.total_length += len(corpus)
            self.cum_corpus_lengths.append(self.total_length)

    def load_corpus(self):
        # check if cached file is available, if so, load it. Otherwise, read from text file
        if 'valid' in self.ann_file:
            partition = 'valid'
        elif 'test' in self.ann_file:
            partition = 'test'
        else:
            partition = 'train'
        cache_file = os.path.join(self.data_path, "general_corpus_" + partition + ".pickle")
        print("Desired cache file location %s " % cache_file)
        if os.path.exists(cache_file):
            return pickle.load(open(cache_file, "rb"))
        else:
            corpus = []
            for ann_file in self.ann_file.split('+'):
                print("loading monolingual data %s now!" % ann_file)
                max_lines, i = 10000000 if "en" in ann_file else 10000000, 0
                corpus_sub = []
                with open(os.path.join(self.data_path, ann_file), 'r', encoding=self.encoding, errors='ignore') as f:
                    for l in f.readlines():
                        ls = l.strip('\n').strip('\r').strip('\n')
                        if len(ls) == 0:    continue
                        i += 1
                        ls = ls[:min(len(ls), self.seq_len)]
                        # convert to ids
                        tokens = self.tokenizer.tokenize(ls)
                        ids = self.tokenizer.convert_tokens_to_ids(tokens)
                        corpus_sub.append(ids)
                        if i % 10000 == 0:
                            print("Loaded %d lines now" % i)
                        if i >= max_lines:  break
                corpus.append(corpus_sub)
            if self.cache_db and not os.path.exists(cache_file):
                pickle.dump(corpus, open(cache_file, "wb"))
            return corpus

    @property
    def data_names(self):
        return ['text', 'mlm_labels']

    def __len__(self):
        # total_len = 0
        # return len(self.corpus)
        return self.total_length

    def _fetch_item(self, idx):
        dataset_idx = bisect.bisect_right(self.cum_corpus_lengths, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_corpus_lengths[dataset_idx - 1]
        return self.corpus[dataset_idx][sample_idx]

    def __getitem__(self, item):
        # raw = self.corpus[item]
        # get bucket index from self.corpus_lengths
        ids = self._fetch_item(item)
        assert ids is not None
        # tokenize
        # tokens = self.tokenizer.basic_tokenizer.tokenize(raw)

        # add more tokens if len(tokens) < min_len
        _cur = (item + 1) % len(self.corpus)
        while len(ids) < self.min_seq_len:
            _cur_ids = self.corpus[_cur]
            ids.extend(_cur_ids)
            _cur = (_cur + 1) % len(self.corpus)

        # masked language modeling
        ids, mlm_labels = self.random_word_wwm(ids)

        # convert token to its vocab id
        # ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # truncate
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
            mlm_labels = mlm_labels[:self.seq_len]

        return ids, mlm_labels

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


if __name__ == '__main__':
    ann_files = 'de.train+en.train+fr.train+ja.train+ru.train+zh.train'
    pretrained_model_name = '/mnt/home/hongliangfei/research/emnlp2020-vlbert-base/model/pretrained_model/bert-base-multilingual-uncased'
    data_path = '/mnt/data/hongliangfei/emnlp2020-vlbert-base/data/mono_corpus/'
    gc = GeneralCorpus(ann_files, pretrained_model_name, data_path, cache_db=True)
    print("Done")
