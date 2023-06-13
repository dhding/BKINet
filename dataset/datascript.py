# generate **.pth
import os
import sys
import shutil
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import operator
import argparse
import collections
import logging
import json
import re

input_txt = '../ln_data/anns/refcoco/testA.txt'
dataset = 'refcoco'
split = 'testA'

res = []
with open(input_txt) as f:
    lines = f.readlines()
    for line in lines:
        line = line.split() 
        stop = len(line)
        img_name = line[0]
        for i in range(1,len(line)):
            if (line[i]=='~'):
                stop=i
                break
        box_ = [list(map(int,box.split(','))) for box in line[1:stop]]
        box = box_[0][:4]
        seg_id=box_[0][-1]
        
        sent_stop=stop+1
        for i in range(stop+1,len(line)):
            if line[i]=='~': 
                des = ''
                for word in line[sent_stop:i]:
                    des = des + word + ' '
                sent_stop=i+1
                des = des.rstrip(' ')
                res.append((img_name, seg_id, box, des))
        des = ''
        for word in line[sent_stop:len(line)]:
            des = des + word + ' '
        des = des.rstrip(' ')
        res.append((img_name, seg_id, box, des))

imgset_path = '{0}_{1}.pth'.format(dataset, split)
images = torch.save(res, imgset_path)
