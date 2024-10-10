#!/usr/bin/env python3
"""
Defines function that finds a snippet of text within a reference document
to answer a question
"""

import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf


def question_answer(question, reference):
    # Load pre-trained BERT tokenizer from transformers
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Load the BERT QA model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize input question and reference text (context)
    input_dict = tokenizer(question, reference, return_tensors='tf')

    # The model expects specific keys in the dictionary, so rename them
    input_dict = {
        'input_ids': input_dict['input_ids'],
        'input_mask': input_dict['attention_mask'],
        'segment_ids': input_dict['token_type_ids']
    }

    # Get the output of the model (logits for start and end positions of the answer)
    outputs = model(input_dict)
    start_logits, end_logits = outputs[0], outputs[1]

    # Get the most probable start and end positions
    start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=1).numpy()[0]

    # Convert the input IDs back to tokens, and then to the original text
    tokens = tokenizer.convert_ids_to_tokens(input_dict['input_ids'][0].numpy())

    # Reconstruct the answer from the tokens between start and end
    answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

    # If no valid answer is found, return None
    if start_idx == 0 and end_idx == 0:
        return None

    return answer
