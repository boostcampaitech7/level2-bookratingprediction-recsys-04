import torch
import torch.nn as nn

from ._helpers import Text_DeepFM

class ELECTRA(Text_DeepFM):
    def __init__(self, args, data):
        super(ELECTRA, self).__init__(args, data)
    
    def forward(self, x):
        return super().forward(x)
      
class RoBERTa(Text_DeepFM):
    def __init__(self, args, data):
        super(RoBERTa, self).__init__(args, data)
        
    def forward(self, x):
        return super().forward(x)