# encoding=utf-8

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from options import MonodepthOptions
from trainer import Trainer, StereoTrainer, MyStereoTrainer, MergerdTrainer

options = MonodepthOptions()
opts = options.parse()
print(opts)


if __name__ == "__main__":
    # trainer = Trainer(opts)
    # trainer = StereoTrainer(opts)
    # trainer = MyStereoTrainer(opts)
    trainer = MergerdTrainer(opts)
    trainer.train()
