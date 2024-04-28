from model import Model 
import torch 
import torchaudio 
from utils import Point,Segment

def load_audio_file(audio_path:str,sample_rate:int):

    waveform,rate=torchaudio.load(audio_path,sr=sample_rate)

    return waveform,rate

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
    """
    Backtracks the probability graph and returns 
    the most probable path using CTC and viterbi 
    algorithm.
    Args:
        graph: Tensor of probabilities of dim (len(emission),len(tokens)).
        emmision: Tensor containig actual set probabilities. 
        tokens: Tokenized transcript.
        blank_if: default 0.
    Returns:
        path: list of Point object containing most probable path.
    """

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
    words=[]
    i1,i2=0,0
    while i1<len(segments):
        if i2>= len(segments) or segments[i2].label==separator:
            if i1!=i2:
                segs=segments[i1:i2]
                word="".join([segs.label for seg in segs])
                score=sum(segs.score*segs.length for seg in segs)/sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1=i2+1
            i2=i1
        else:
            i2+=1
    
    return words 

def generate_alignment():
    """jo bhi alingnment bane hai usko save kardega"""
    
    pass 

