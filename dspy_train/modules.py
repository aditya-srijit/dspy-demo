import dspy
from .signatures import CoTSignature

class CoTModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(CoTSignature)
        
    def forward(self, question):
        return self.prog(question=question)
