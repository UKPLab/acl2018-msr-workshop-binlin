#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import os

import numpy as np
import torch
from torch.autograd.variable import Variable
from components.data.base_data import BaseDataClass
from components.data.syn_fxtractor import SynFxtractor
from components.data.vocabulary.syn_vocab import SynVocabulary
from components.utils.combinatorics import get_combinations
from components.utils.graph import get_bfs_tuples, get_gold_label
from components.utils.graph import dg_from_tokens
from components.utils.readers import raw_data_to_tokens
from components.utils.readers import read_conll_data_file

logger = logging.getLogger('main')

class SynData(BaseDataClass):

    def training_setup(self, train_data_fname, dev_data_fname):

        assert train_data_fname is not None
        assert dev_data_fname is not None

        train_raw = read_conll_data_file(train_data_fname)
        dev_raw = read_conll_data_file(dev_data_fname)

        train_graphs = [dg_from_tokens(toks) for toks in raw_data_to_tokens(train_raw)]
        dev_graphs = [dg_from_tokens(toks) for toks in raw_data_to_tokens(dev_raw)]

        self.vocab.setup(vocab_path=self.vocab_fn,
                         data=train_graphs,
                         lower=self.lower,
                         source='depgraphs')

        # Vectorize data
        self.train = self.vectorize_graphs(train_graphs)
        self.dev = self.vectorize_graphs(dev_graphs)

        # Set the number of extracted features (need for the NN)
        random_instance_x = self.dev[0][0]
        self.num_features = len(random_instance_x)

    def prediction_setup(self):
        # load vocabulary file from the filesystem
        assert 'vocab_fn' in self.config, logger.error('Specify "vocab_fn" in the config file')
        self.vocab.setup(vocab_path=self.vocab_fn,
                         data=None,
                         lower=self.lower,
                         source='file')

        self.num_features = self.config['num_features']

    def set_vocabulary_class(self):
        self.vocab = SynVocabulary()

    def get_special_params_to_save(self):
        pd = {}
        pd['num_features'] = self.num_features
        pd['vocab_fn'] = self.vocab_fn
        return pd

    def vectorize_graphs(self, depgraphs):
        """
        Convert raw lists of tokens to numerical representations.

        :param raw_x: a list of N instances, each being a list of Token values; N = size(dataset)
        :return:
        """

        # we have two models: parent-child (PC) and siblings (SBL)
        x_vecs = []
        y_vecs = []

        for dg in depgraphs:

            # Order the nodes according to the reversed BFS order.
            bfs_tuples = get_bfs_tuples(dg, dg.graph['root'])

            # For each head node and its children,
            # extract features for the nodes and assign LEFT/RIGHT labels
            for (group_head_id, group) in bfs_tuples:

                head_features = SynFxtractor.extract_node_features(group_head_id, dg, self.vocab)
                children_features = [SynFxtractor.extract_node_features(child_node_id, dg, self.vocab) for child_node_id in
                                     group]

                gold_labels = [get_gold_label(dg, child_node_id, group_head_id) for child_node_id in group]
                for i, f in enumerate(children_features):
                    x_vecs.append(f + head_features)
                    y_vecs.append(gold_labels[i])

                # 2. Sibling model features
                # sbl_vecs now has the format:
                # [((n1, [featrues1]), (n2, [featrues2])),
                #  ((n1, [featrues1]), (n3, [featrues3])),
                #  ((n2, [featrues2]), (n3, [featrues3])),
                #  ...
                # ]
                for sbl_pair in get_combinations(zip(group, children_features), 2):

                    node1_id = sbl_pair[0][0]
                    node2_id = sbl_pair[1][0]

                    feats1 = sbl_pair[0][1]
                    feats2 = sbl_pair[1][1]

                    # correct way
                    label = get_gold_label(dg, node2_id, node1_id)  # comparing node_id's of two siblings -> label
                    x_vecs.append(feats2 + feats1)  # putting their features and gold label together
                    y_vecs.append(label)

        return (Variable(torch.LongTensor(x_vecs)).to(self.device),
                Variable(torch.FloatTensor(y_vecs)).to(self.device))

    def batchify_vectorized_data(self, xy_ids, batch_size, num_classes=1):

        """
        Cut the dataset into batches.
        :param xy_ids: a tuple of 2 lists:
            - a list of MR instances, each itself being a list of numerical ids
            - a list ot tokenized texts (same format)
        :param batch_size:
        :return:
        """
        num_batches = xy_ids[0].shape[0] // batch_size

        batches = [(xy_ids[0][bi * batch_size: (bi + 1) * batch_size],
                    xy_ids[1][bi * batch_size: (bi + 1) * batch_size])
                   for bi in range(num_batches)]

        return batches




component = SynData
