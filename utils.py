# -*- coding: utf-8 -*-
from datetime import datetime

import torch
from torch._C import R
import numpy as np

def mask_fill(
    fill_value: float,
    tokens: torch.tensor,
    embeddings: torch.tensor,
    padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)

class RandomErasing():
    def __init__(self, p=0.5, scale=(0.1, 0.1)):
        super(RandomErasing, self).__init__()
        self.p = p
        self.scale = scale

    def __call__(self, x):
        input_size = x.shape
        assert len(input_size) == 2 or len(input_size) == 3

        if np.random.random() > self.p:
            if len(input_size) == 2:
                width, height = input_size 
            else:
                batch, width, height = input_size 
            
            # cut range
            cut_width = int(self.scale[0] * width)
            cut_height = int(self.scale[1] * height)

            x_axis = np.random.randint(0, width-cut_width+1)
            y_axis = np.random.randint(0, height-cut_height+1)

            assert x_axis+cut_width+1 <= width
            assert y_axis+cut_height+1 <= height
            if batch != None:
                x[:, x_axis: x_axis+cut_width+1, y_axis: y_axis+cut_height+1] = 0
            else:
                x[x_axis: x_axis+cut_width+1, y_axis: y_axis+cut_height+1] = 0
        else:
            pass
        
        return x 