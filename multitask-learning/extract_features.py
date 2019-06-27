# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch BERT model. The following code is inspired by the HuggingFace version
and it is built upon in places (license below). In particular it can be
used and modified for commercial use:
https://github.com/huggingface/pytorch-pretrained-BERT/
In this code we can extract the feature vectors from the BERT encodings.
"""

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class InputExample(object):
    """A class to store a single input example"""
    def __init__(self, unique_id, text_a, text_b=None, label=None):
        """A wrapper for an input example

        Parameters
        ----------
        object : [type]
            [description]
        unique_id : int
            The unique id for the example
        text_a : str
            The first sentence
        text_b : int, optional
            The second sentence, by default None
        label : str, optional
            label for the example, required for training and dev sets.
            Not required for the test set
        """
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """The convention in BERT is:
    a) For sequence pairs:
    tokens:      [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    segment_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1  1
    b) For single sequences:
    tokens:       [CLS] the dog is hairy . [SEP]
    segment_ids:   0     0   0   0   0   0  0

    Where "segment_ids" are used to indicate whether this is the first
    sequence or the second sequence. The embedding vectors for `type=0` and
    `type=1` were learned during pre-training and are added to the wordpiece
    embedding vector (and position vector). This is not *strictly* necessary
    since the [SEP] token unambigiously separates the sequences, but it makes
    it easier for the model to learn the concept of sequences.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """
    Loads a data file into a list of `InputBatch`s.
    Some potential speedups have been noted and commented but this code tried to remain
    faithful to the huggingface implementation


    Parameters
    ----------
    examples : list
        List of InputExample objects
    label_list : list
        List of labels typically returned from the data processors
    max_seq_length : int
        Maximum sequence length (to zero pad up to)
    tokenizer : object
        Tokenizer object, typically BERTTokenizer with do_lower_case = True/False
        as appropriate

    Returns
    -------
    list
        list of InputFeatures compatible with BERT
    """

    # Create unique dictionary of labels and their corresponding map
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # Below could definitely be made more efficient without for looping e.g
        # if not tokens_b:
        #     # Account for [CLS] and [SEP] tokens in the "+2"
        #     tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        #     segment_ids = [0] * (len(tokens_a) + 2)
        # else:
        #     tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        #     segment_ids = [0] * (len(tokens_a) + 2) + \
        #                   [1] * (len(tokens_b) + 1)
        tokens = []
        segment_ids = []
        # Start with the [CLS] token with id 0
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            # Each token in the first sentence has id 0
            tokens.append(token)
            segment_ids.append(0)
        # Add the [SEP] token, also with id 0
        tokens.append("[SEP]")
        segment_ids.append(0)

        # If we have tokens for text_b then append and label with id 1
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length. Could be sped up as below:
        # padding_length = max_seq_length - len(input_ids)
        # input_ids += [0]*padding_length
        # input_mask += [0]*padding_length
        # segment_ids += [0]*padding_length
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # Make sure everything has the right shape
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # Get the corresponding label id for the example
        label_id = label_map[example.label]

        # log some examples to make sure everything is working
        if ex_index < 3:
            # We use Python 3.6 f-strings for nice formatting!
            LOGGER.info("*** Example ***")
            LOGGER.info(f"unique_id: {example.unique_id}")
            LOGGER.info(f"tokens: {' '.join(str(token) for token in tokens)}")
            LOGGER.info(f"input_ids: {' '.join(str(input_id) for input_id in input_ids)}")
            LOGGER.info(f"input_mask: {' '.join(str(mask) for mask in input_mask)}")
            LOGGER.info(f"segment_ids: {' '.join(str(segment_id) for segment_id in segment_ids)}")
            LOGGER.info(f"label: {example.label} (id {label_id})")

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """A simple function to truncate a sequence pair in place to a max length
    We truncate the longer of the two sentences one token at a time in an
    attempt to maintain the most amount of information possible
    Parameters
    ----------
    tokens_a : array
        First sentence as a sequence of tokens
    tokens_b : array
        Second sentence as a sequence of tokens
    max_length : int
        Maximum length of the combined sentence lengths
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
