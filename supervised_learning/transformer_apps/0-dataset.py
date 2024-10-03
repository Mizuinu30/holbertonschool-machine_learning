#!/usr/bin/env python3
"""Dataset class module"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class that loads and preps a dataset for machine translation.
    """
    def __init__(self):
        """
        Initializes the Dataset instance.
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data (tf.data.Dataset): Dataset whose examples are formatted as a tuple (pt, en).
                pt is the tf.Tensor containing the Portuguese sentence.
                en is the tf.Tensor containing the corresponding English sentence.

        Returns:
            tokenizer_pt (BertTokenizerFast): Portuguese tokenizer.
            tokenizer_en (BertTokenizerFast): English tokenizer.
        """
        tokenizer_pt = BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # Note: The actual training of the tokenizer is not shown here as it requires a different approach.
        # This is a placeholder to indicate where the training would occur.
        # tokenizer_pt.train_new_from_iterator((pt.numpy().decode('utf-8') for pt, _ in data), vocab_size=2**13)
        # tokenizer_en.train_new_from_iterator((en.numpy().decode('utf-8') for _, en in data), vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

if __name__ == "__main__":
    data = Dataset()
    for pt, en in data.data_train.take(1):
        print(pt.numpy().decode('utf-8'))
        print(en.numpy().decode('utf-8'))
    for pt, en in data.data_valid.take(1):
        print(pt.numpy().decode('utf-8'))
        print(en.numpy().decode('utf-8'))
    print(type(data.tokenizer_pt))
    print(type(data.tokenizer_en))
