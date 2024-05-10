import torch 
import numpy as np
import json  
import os 
from scipy.io.wavfile import write

from model import Model 
from audio import load_audio,SAMPLE_RATE
from utils import Point,Segment

def _load_model():

    if Model._instance is not None:
        return Model._instance
    else:
        return Model() 

def _extract_base(path):
    
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]



def force_align(input_path:str,transcript:str,output_dir:str):
    
    model=_load_model()

    #loading_audio
    audio=load_audio(input_path)
    token_ids=model.tokenize(transcript)
    preprocessed_transcript=transcript.replace(" ","|")

    #force_alignment
    emission=model.inference(audio)
    graph=compose_graph(emission,token_ids)
    path=backtrack(graph,emission,token_ids)
    segments = merge_repeats(path,preprocessed_transcript)
    word_segments=merge_words(segments)

    #preparing audio segments
    json_dict={}
    segment_list=[]

    audio_segments=generate_audio_segments(audio,graph,word_segments,SAMPLE_RATE)
    
    #saving segments in the oupt_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    folder_path=os.path.join(output_dir,_extract_base(input_path))
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    json_path=os.path.join(folder_path,"metadata.json")
    segment_paths=[]
    for i in range(len(audio_segments)):
        segment_paths.append(os.path.join(folder_path,f"segment_{i}.wav"))
    
    
    for i in range(len(segment_paths)):
        segment_dict={}
        segment_dict["word_label"]=audio_segments[i][0]
        segment_dict["duration"]=audio_segments[i][1]
        segment_dict["file_path"]=segment_paths[i]
        segment_list.append(segment_dict)

        wave_form=audio_segments[i][2]
        write(segment_paths[i],8000,wave_form)
    
    json_dict["original_file_path"]=input_path
    json_dict["original_transcript"]=transcript
    json_dict["audio_segments"]=segment_list 

    with open(json_path,'w') as json_file:
        json.dump(json_dict,json_file,indent=4)

    print(f"Force Alignment complete files save at {folder_path}")

def compose_graph(emission,tokens,blank_id=0):

    num_frame=emission.size(0)
    num_tokens=len(tokens)

    graph=torch.zeros((num_frame,num_tokens))
    graph[1:,0]=torch.cumsum(emission[1:,blank_id],0)
    graph[0,1:]=-float("inf")
    graph[-num_tokens+1:,0]=float("inf")

    for t in range(num_frame-1):

        graph[t+1,1:]=torch.maximum(graph[t,1:]+emission[t,blank_id],
                                    graph[t,:-1]+emission[t,tokens[1:]],)
        
    return graph 

def backtrack(graph,emission,tokens,blank_id=0): 

    t,j=graph.size(0)-1,graph.size(1)-1

    path=[Point(j,t,emission[t,blank_id].exp().item())]
    while j>0:

        assert t>0

        p_stay=emission[t-1,blank_id]
        p_change=emission[t-1,tokens[j]]

        stayed=graph[t-1,j]+p_stay
        changed=graph[t-1,j-1]+p_change

        stayed=graph[t-1,j]+p_stay
        changed=graph[t-1,j-1]+p_change

        t-=1
        if changed>stayed:
            j -=1

        prob=(p_change if changed>stayed else p_stay).exp().item()
        path.append(Point(j,t,prob))

    while t>0:
        prob=emission[t-1,blank_id].exp().item()
        path.append(Point(j,t-1,prob))
        t-=1
    
    return path[::-1]

def merge_repeats(path,transcript):
    i1,i2=0,0
    segments=[]
    while i1<len(path):
        while i2<len(path) and path[i1].token_index == path[i2].token_index:
            i2+=1
        score=sum(path[k].score for k in range(i1,i2))/(i2-i1)

        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2-1].time_index+1,
                score,
            )
        )
        i1=i2

    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def generate_audio_segments(wave_form,graph,word_segments,sample_rate):

  ratio=wave_form.shape[0]/graph.size(0)
  audio_segments=[]

  for i in range(len(word_segments)):
    word=word_segments[i]
    x0=int(ratio*word.start)
    x1=int(ratio*word.end)
    time_interval=f"{x0/ sample_rate:.3f}-{x1/sample_rate:.3f} sec"
    audio_seg=wave_form[x0:x1]
    audio_segments.append((word.label,time_interval,audio_seg))

  return audio_segments

