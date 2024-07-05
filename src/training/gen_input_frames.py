from concurrent.futures import ProcessPoolExecutor
import copy
import datetime
import logging
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import json
from types import SimpleNamespace
logger = logging.getLogger(__name__)

import json
from decord import VideoReader

import os.path as osp
import io
from tqdm import tqdm
import webdataset as wds
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from braceexpand import braceexpand
import numpy as np
from PIL import Image
import random

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def read_frames_decord_stream(
        video_stream, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, trimmed30=False
    ):
    video_reader = VideoReader(io.BytesIO(video_stream), num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # only use top 30 seconds
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = torch.from_numpy(video_reader.get_batch(frame_indices).asnumpy())  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, duration

def custom_decoder(key, data):
    if key.endswith(".jpg") or key.endswith(".png"):
        return np.array(Image.open(io.BytesIO(data)))
    elif key.endswith(".pth"):
        return torch.load(io.BytesIO(data), map_location=torch.device('cpu'))
    elif key.endswith(".txt"):
        return data.decode("utf-8")
    elif key.endswith(".json"):
        return json.loads(data.decode('utf-8'))
    else:
        return data


def process(device_id, idx, url, output, maxcount=999999999, batch_size=100):
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')




    src = wds.WebDataset(url) \
            .decode(custom_decoder) \
            .rename(video="mp4", text="json")\
            .to_tuple("__key__", "video", "text", "pth")
    src = src.batched(batch_size)

    with wds.TarWriter(output) as dst:
        for batch in tqdm(src, desc=f"Processing Images on GPU {device_id}, Process {idx}"):
            keys, videos, texts, pths = batch
            new_pths = []

            for video, text, pth in zip(videos,texts,pths):
                frames, _, _ = read_frames_decord_stream(video_stream=video,num_frames=4,sample='middle',max_num_frames=-1,trimmed30=False)#frames=[4, 3, 240, 320]
                pth.update({
                    'frames':frames
                })
                new_pths.append(pth)
                
            for i, key in enumerate(keys):
                sample = {
                    "__key__":key,
                    "mp4":videos[i],
                    "json":texts[i],
                    "pth":new_pths[i]
                }
                dst.write(sample)


def extract_emb():
    num_gpus = 2  # 有两个GPU
    models_per_gpu = 1  # 每个GPU运行1个模型实例
    input_shards = braceexpand("{0..8}")
    output_shards = braceexpand("{0..8}")
    inputs = [f"/home/user/data/MSRVTT-videos/train_t_umt/train_{shard}.tar" for shard in input_shards]
    outputs = [f"/home/user/data/MSRVTT-videos/train_t_umt_preframes/train_{shard}.tar" for shard in output_shards]


    with ProcessPoolExecutor(max_workers=num_gpus * models_per_gpu) as executor:
        futures = []
        for i in range(len(inputs)):
            device_id = i % num_gpus
            proc_idx = i % models_per_gpu
            futures.append(executor.submit(process, device_id, proc_idx, inputs[i], outputs[i], batch_size=100))
        
        for future in tqdm(futures, desc="Total Progress"):
            future.result()

    print('done')

if __name__ == "__main__":
    extract_emb()