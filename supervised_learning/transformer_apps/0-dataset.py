#!/usr/bin/env python3
"""Class Dataset"""

import tensorflow_datasets as tfds


class Dataset:
    """Class Dataset"""

    def __init__(self):
        """Class constructor"""

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                        with_info=True,
                                        as_supervised=True)
        self.data_train = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """tokenize_dataset"""

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en
