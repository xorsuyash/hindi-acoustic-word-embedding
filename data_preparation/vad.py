"""voice activity detection"""

import hashlib 
import os 
import urllib 
from typing import Callable, Optional, Text, Union
import urllib.request


import numpy as np 
import pandas as pd 
import torch 
from pyannote.audio import Model 
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from tqdm import tqdm 

VAD_SEGMENTATION_URL="https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"

def load_vad_model(device, vad_onset=0.500, vad_offset=0.363 , use_auth_token=None, model_fp=None):

    model_dir=torch.hub._get_torch_home()
    os.makedirs(model_dir,exist_ok=True)
    if model_fp is None:
        model_fp=os.path.join(model_dir,"vad-segmentation.bin")
    if os.path.exists(model_fp) and not os.path.isfile(model_fp):
        raise RuntimeError(f"{model_fp} exists and is not a regular file")
    
    if not os.path.isfile(model_fp):
        with urllib.request.urlopen(VAD_SEGMENTATION_URL) as source, open(model_fp,"wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024
            ) as loop:
                while True:
                    buffer=source.read(8192)
                    if not buffer:
                        break 

                    output.write(buffer)
                    loop.update(len(buffer))

    model_bytes=open(model_fp,"rb").read()
    if hashlib.sha256(model_bytes).hexdigest()!=VAD_SEGMENTATION_URL.split('/')[-2]:
        raise RuntimeError(
            "Model has been dowloaded but the SHA256 checksum does not match."

        )
    
    vad_model=Model.from_pretrained(model_fp,use_auth_token=use_auth_token)
    hyperparameters={
        "onset":vad_onset,
        "offset":vad_offset,
        "min_duration_on":0.1,
        "min_duration_off":0.1
    }

    vad_pipeline=VoiceActivityDetection(segmentation=vad_model,device=torch.device(device))
    vad_pipeline.instantiate(hyperparameters)

    return vad_pipeline


class Binarize:
    pass 

def merge_vad():
    pass 

def merge_chunks():
    pass   