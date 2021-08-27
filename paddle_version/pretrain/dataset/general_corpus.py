import datetime
import logging
import os
import random
from paddle.io import Dataset
from paddlenlp.transformers import BertTokenizer


class GeneralCorpus(Dataset):
    def __init__(self, ann_file, pretrained_model_name, data_path, tokenizer=None, seq_len=64, min_seq_len=64,
                 encoding="utf-8", on_memory=True, cache_db=False, **kwargs):
        super(GeneralCorpus, self).__init__()
        assert on_memory, "only support on_memory mode!"

        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained(
            'bert-base-multilingual-uncased')
        self.is_uncased = True if pretrained_model_name is not None and 'uncased' in pretrained_model_name else False
        self.vocab = self.tokenizer.vocab.token_to_idx
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

        # self.total_length = 0
        # self.cum_corpus_lengths = []
        # for corpus in self.corpus:
        #     self.total_length += len(corpus)
        #     self.cum_corpus_lengths.append(self.total_length)

    def load_corpus(self):
        # check if cached file is available, if so, load it. Otherwise, read from text file
        # if 'valid' in self.ann_file:
        #     partition = 'valid'
        # elif 'test' in self.ann_file:
        #     partition = 'test'
        # else:
        #     partition = 'train'
        # cache_file = os.path.join(self.data_path, "general_corpus_" + partition + ".pickle")
        # print("Desired cache file location %s " % cache_file)
        # if os.path.exists(cache_file):
        #     return pickle.load(open(cache_file, "rb"))
        # else:
        corpus = []
        for ann_file in self.ann_file.split('+'):
            print("loading monolingual data %s now!" % ann_file)
            # max_lines, i = 50000000, 0
            max_lines, i = 1000000, 0
            corpus_sub = []
            with open(os.path.join(self.data_path, ann_file), 'r', encoding=self.encoding, errors='ignore') as f:
                for l in f.readlines():
                    ls = l.strip('\n').strip('\r').strip('\n')
                    i += 1
                    if len(ls) == 0:    continue
                    if self.is_uncased:
                        ls = ls.lower()
                    ls = ls[:min(len(ls), self.seq_len)]
                    if len(corpus_sub) < max_lines:
                        # # convert to ids
                        # tokens = self.tokenizer.tokenize(ls)
                        # ids = self.tokenizer.convert_tokens_to_ids(tokens)
                        # corpus_sub.append(ids)
                        corpus_sub.append(ls)
                    else:
                        idx = random.randint(0, max_lines)
                        if idx < max_lines:
                            corpus_sub[idx] = ls
                    if i % 50000 == 0:
                        print("Loaded %d lines now" % i)
                    # turn on when debugging
                    if i >= 5 * max_lines:  break
            # corpus.append(corpus_sub)
            corpus.extend(corpus_sub)
        # if self.cache_db and not os.path.exists(cache_file):
        #     pickle.dump(corpus, open(cache_file, "wb"))
        return corpus

    @property
    def data_names(self):
        return ['text', 'mlm_labels']

    def __len__(self):
        # total_len = 0
        return len(self.corpus)
        # return self.total_length

    # def _fetch_item(self, idx):
    #     """
    #     Used for concatenated dataset.
    #     :param idx:
    #     :return:
    #     """
    #     dataset_idx = bisect.bisect_right(self.cum_corpus_lengths, idx)
    #     if dataset_idx == 0:
    #         sample_idx = idx
    #     else:
    #         sample_idx = idx - self.cum_corpus_lengths[dataset_idx - 1]
    #     return self.corpus[dataset_idx][sample_idx]

    def __getitem__(self, item):
        # get bucket index from self.corpus_lengths
        # ids = self._fetch_item(item)
        # raw = self._fetch_item(item)
        # print("Fetching %dth item: %s" % (item, raw))
        raw = self.corpus[item]
        # tokenize
        tokens = self.tokenizer.basic_tokenizer.tokenize(raw)

        # masked language modeling
        tokens, mlm_labels = self.random_word_wwm(tokens)

        # convert token to its vocab id
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # truncate
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
            mlm_labels = mlm_labels[:self.seq_len]

        return ids, mlm_labels

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
                        output_tokens.append(random.choice(list(self.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)

                        # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    try:
                        output_label.append(self.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(self.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        return output_tokens, output_label

    # def random_word_wwm(self, tokens):
    #     output_tokens = []
    #     output_label = []
    #
    #     for i, token in enumerate(tokens):
    #         # sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
    #         prob = random.random()
    #         # mask token with 15% probability
    #         if prob < 0.15:
    #             prob /= 0.15
    #
    #             # 80% randomly change token to mask token
    #             if prob < 0.8:
    #                 # for sub_token in sub_tokens:
    #                 output_tokens.append(self.tokenizer.vocab["[MASK]"])
    #             # 10% randomly change token to random token
    #             elif prob < 0.9:
    #                 # for sub_token in sub_tokens:
    #                 output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
    #                 # -> rest 10% randomly keep current token
    #             else:
    #                 # for sub_token in sub_tokens:
    #                 output_tokens.append(token)
    #
    #                 # append current token to output (we will predict these later)
    #             # for sub_token in sub_tokens:
    #             try:
    #                 output_label.append(token)
    #             except KeyError:
    #                 # For unknown words (should not occur with BPE vocab)
    #                 output_label.append(self.tokenizer.vocab["[UNK]"])
    #                 logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(token))
    #         else:
    #             # for sub_token in sub_tokens:
    #             # no masking token (will be ignored by loss function later)
    #             output_tokens.append(token)
    #             output_label.append(-1)
    #
    #     ## if no word masked, random choose a word to mask
    #     # if all([l_ == -1 for l_ in output_label]):
    #     #    choosed = random.randrange(0, len(output_label))
    #     #    output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]
    #
    #     return output_tokens, output_label


if __name__ == '__main__':
    ann_files = 'de.train+en.train+fr.train+ja.train+ru.train+zh.train'
    pretrained_model_name = '/mnt/home/hongliangfei/research/emnlp2020-vlbert-base/model/pretrained_model/bert-base-multilingual-uncased'
    data_path = '/mnt/data/hongliangfei/emnlp2020-vlbert-base/data/mono_corpus/'
    gc = GeneralCorpus(ann_files, pretrained_model_name, data_path, cache_db=True)
    print("Done")
