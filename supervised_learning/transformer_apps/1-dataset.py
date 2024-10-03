#!/usr/bin/env python3
"""Class Dataset"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """ Class Dataset """
    def __init__(self):
        """Initializes the Dataset class by loading
        and preparing the dataset."""
        # Load the training and validation datasets
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        # Tokenize the datasets
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for
        the dataset using pre-trained models.

        Args:
            data: A tf.data.Dataset whose examples
            are formatted as a tuple (pt, en).

        Returns:
            tokenizer_pt: The Portuguese tokenizer.
            tokenizer_en: The English tokenizer.
        """
        # Load pre-trained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        # Define iterators for the Portuguese and English sentences
        def pt_iterator():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        # Train new tokenizers with the specified maximum vocabulary size
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), 2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), 2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.

        Args:
            pt: A tf.Tensor containing the Portuguese sentence.
            en: A tf.Tensor containing the corresponding English sentence.

        Returns:
            pt_tokens: list containing the Portuguese tokens.
            en_tokens: list containing the English tokens.
        """
        # Encode the sentences into tokens
        pt_tokens = self.tokenizer_pt.encode(pt.numpy().decode('utf-8'))
        en_tokens = self.tokenizer_en.encode(en.numpy().decode('utf-8'))

        # Add the start token (vocab_size) and end token (vocab_size + 1)
        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens
