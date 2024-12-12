"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.processors.alpro_processors import AlproVideoTrainProcessor, AlproVideoEvalProcessor
import numpy as np
from decord import VideoReader
import random as rnd
import torch

def load_video(video_path, n_frms=60, height=-1, width=-1, sampling="uniform"):
    vr = VideoReader(uri=video_path, height=height, width=width)
    fps = vr.get_avg_fps()  # Get the video's frame rate

    vlen = len(vr)  # Total number of frames in the video
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    # Determine frame indices based on the sampling strategy
    if sampling == "uniform":
        indices = np.linspace(start=start, stop=end, num=n_frms, endpoint=False).astype(int)
    elif sampling == "random":
        intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
        indices = [
            low if low == high else rnd.choice(range(low, high))
            for low, high in zip(intervals[:-1], intervals[1:])
        ]
    else:
        raise NotImplementedError(f"Sampling strategy '{sampling}' is not implemented.")

    # Load frames using the indices
    frms = vr.get_batch(indices).permute(3, 0, 1, 2).float()  # (C, T, H, W)

    return frms, indices, fps

class AlproVideoTrainProcessor_Stamps(AlproVideoTrainProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.9, max_scale=1.0, n_frms=60, full_video=True):
        super().__init__(image_size=image_size, mean=mean, std=std, n_frms=n_frms, min_scale=min_scale, max_scale=max_scale, full_video=full_video)

    def __call__(self, vpath):

        clip, indices, fps = load_video(
                video_path=vpath,
                n_frms=self.n_frms,
                height=self.image_size,
                width=self.image_size,
                sampling="random",
            )
        
        transformed = self.transform(clip)

        pad_size = self.n_frms - transformed.shape[1]
        if pad_size>0:
            last_frame = transformed[:, -1, :, :].unsqueeze(1)
            repeat_frames = last_frame.repeat(1, pad_size, 1, 1)
            transformed = torch.cat([transformed, repeat_frames], dim=1)

        return transformed, indices, fps
    
class AlproVideoEvalProcessor_Stamps(AlproVideoEvalProcessor):
    def __init__(self, image_size=224, mean=None, std=None, n_frms=60,  full_video=True):
        super().__init__(image_size=image_size, mean=mean, std=std, n_frms=n_frms, full_video=full_video)

    def __call__(self, vpath):
        clip, indices, fps = load_video(
                video_path=vpath,
                n_frms=self.n_frms,
                height=self.image_size,
                width=self.image_size,
                sampling="uniform",
            )
        
        transformed = self.transform(clip)

        pad_size = self.n_frms - transformed.shape[1]
        if pad_size>0:
            last_frame = transformed[:, -1, :, :].unsqueeze(1)
            repeat_frames = last_frame.repeat(1, pad_size, 1, 1)
            transformed = torch.cat([transformed, repeat_frames], dim=1)

        return transformed, indices, fps