from scipy import io

paths = ['/Users/allenhuang/works/GazeCapture/pytorch/mean_face_224.mat']
for path  in paths:
    mdict = io.loadmat(path)
    print(mdict.keys())
    print('path name %s :' % path[-18:], mdict['image_mean'].shape)
    print(mdict['image_mean'])


import torch

PATH = 'pytorch/checkpoint.pth.tar'

model_dict = torch.load(PATH, map_location='cpu')

print(model_dict['state_dict']['faceModel.conv.features.4.weight'].shape)
print(model_dict['state_dict']['faceModel.conv.features.0.bias'].shape)
