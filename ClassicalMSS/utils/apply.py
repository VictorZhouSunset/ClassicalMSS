# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs
"""
Code to apply a model to a mix. It will handle chunking with overlaps and
inteprolation between chunks, as well as the "shift trick".
"""

import sys
from pathlib import Path
import typing as tp

root = Path(__file__).parent.parent
sys.path.append(str(root))
from ClassicalMSS.models.wave_u_net import WaveUNet
# from ClassicalMSS.models.conv_tasnet import ConvTasNet
# from ClassicalMSS.models.demucs import Demucs
# from ClassicalMSS.models.hdemucs import HDemucs
# from ClassicalMSS.models.htdemucs import HTDemucs

# Model = tp.Union[WaveUNet, ConvTasNet, Demucs, HDemucs, HTDemucs]

Model = WaveUNet
