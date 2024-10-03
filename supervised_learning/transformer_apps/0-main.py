#!/usr/bin/env python3

import tensorflow as tf
Dataset = __import__('0-dataset').Dataset

data = Dataset()

# Adjust the dataset pipeline
train_data = data.data_train.take(1).cache().repeat()
valid_data = data.data_valid.take(1).cache().repeat()

for pt, en in train_data:
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))

for pt, en in valid_data:
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))

print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))