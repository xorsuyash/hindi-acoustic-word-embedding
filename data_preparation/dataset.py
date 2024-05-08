from torch.utils.data import Dataset,DataLoader 
import pandas as pd 
from torchaudio import transforms


class AudioDataset(Dataset):

  def __init__(self,csv_file,batch_size,shuffle=False):

    self.data=csv_file
    self.batch_size=batch_size
    self.shuffle=shuffle 
    #self.transform=self._init_transform()
    #self.sample_rate=sample_rate
    self.data["preprocessed_transcripts"]=self.data.apply(preprocess_batch,axis=1)
    self.data["tokenized_transcripts"]=self.data.apply(generate_tokens_batch,axis=1)

  
  def __len__(self):
      return len(self.data)

  def __getitem__(self,idx):

    audio_path=self.data["audio_path"][idx]
    
    if isinstance(audio_path, int):
        audio_path = str(audio_path)

    tokenized_transcipt=torch.tensor(self.data["tokenized_transcripts"][idx])

    waveform,sample_rate=torchaudio.load(audio_path) 

    #if self.transform:
    #       waveform = self.transform(waveform)

    waveform=self._pad_waveform(waveform,target_length=59840)
    
    return waveform,tokenized_transcipt 
  
  def _init_transform(self):
        return transforms.Resample(orig_freq=16000, new_freq=8000)
  
  def _pad_waveform(self, waveform, target_length):
        if waveform.size(1) > target_length:
            # Truncate if longer than target length
            waveform = waveform[:, :target_length]
        elif waveform.size(1) < target_length:
            # Pad if shorter than target length
            pad_amount = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

  def create_dataloader(self):

    return DataLoader(self,batch_size=self.batch_size,shuffle=self.shuffle)