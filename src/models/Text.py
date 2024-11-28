import torch
import torch.nn as nn

from ._helpers import Text_DeepFM

class ELECTRA_DeepFM(Text_DeepFM):
    def __init__(self, args, data):
        super(ELECTRA_DeepFM, self).__init__(args, data)
    
    def forward(self, x):
        return super().forward(x)
      
class RoBERTa_DeepFM(Text_DeepFM):
    def __init__(self, args, data):
        super(RoBERTa_DeepFM, self).__init__(args, data)
        
    def forward(self, x):
        return super().forward(x)