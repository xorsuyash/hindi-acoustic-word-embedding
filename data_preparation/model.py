import torch 
from datasets import load_dataset
from transformers import AutoModelForCTC,AutoProcessor
import torchaudio.functional as F


class Model:
    _instance=None 
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_ID="ai4bharat/indicwav2vec-hindi"

    def __new__(cls,*args,**kwargs):

        if cls._instance is None:
            cls._instance=super().__new__(cls)
            cls._instance.model=AutoModelForCTC.from_pretrained(cls.MODEL_ID).to(cls.DEVICE)
            cls._instance.processor = AutoProcessor.from_pretrained(cls.MODEL_ID)

        return cls._instance
    
    def preprocess_audio(self,audio_array,orig_sr=None,target_sr=None):

        resampled_audio = F.resample(torch.tensor(audio_array), orig_sr, target_sr).numpy()
        input_values = self.processor(resampled_audio, return_tensors="pt").input_values
        return input_values.to(self.DEVICE)

    def inference(self,audio):

        with torch.no_grad():
            logits=self.model(audio)
        
        emission=logits[0]
        emission=emission.cpu().detach()

        return emission