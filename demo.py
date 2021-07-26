import torch 
import torchvision
import numpy as np 

from torchvision import transforms 

n = torch.ones([2,5,5])

def RandomErasing(x, p=0.5, scale=(0.2, 0.2)):
    input_size = x.shape
    assert len(input_size) == 2 or len(input_size) == 3

    if np.random.random() > p:
        if len(input_size) == 2:
            width, height = input_size 
        else:
            batch, width, height = input_size 
        
        # cut range
        cut_width = int(scale[0] * width)
        cut_height = int(scale[1] * height)

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

m = RandomErasing(n)


print(n)
print(m)
