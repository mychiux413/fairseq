# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys

from fairseq.data import AddTargetDataset, Dictionary, FileAudioDataset

from . import LegacyFairseqTask, register_task


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("serving")
class ServingTask(LegacyFairseqTask):
    """"""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes volume",
        )
        parser.add_argument(
            "--dictionary",
            required=True,
            help="dictionary file",
        )

    def __init__(self, args, source_dictionary=None, target_dictionary=None):
        super().__init__(args)
        self._target_dictionary = target_dictionary
        self._source_dictionary = source_dictionary
        self.is_ctc = args.criterion == "ctc"

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (omegaconf.DictConfig): parsed command-line arguments
        """

        if args.dictionary:
            target_dictionary = Dictionary.load(args.dictionary)
        else:
            target_dictionary = None

        return cls(args, target_dictionary=target_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        raise AssertionError("load_dataset sould not be called at serving mode")

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices
