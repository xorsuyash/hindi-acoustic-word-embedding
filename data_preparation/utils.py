from dataclasses import dataclass 

@dataclass
class Point:
    token_index:int 
    time_index: int 
    score: float 

@dataclass
class Segment:
    label:str 
    start: int 
    end: int 
    score: float

    def __repr__(self) -> str:
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"
    
    @property
    def length(self):
        return self.end - self.start

