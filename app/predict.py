#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from PIL import Image
from torchvision import transforms


dic={0:'cent',1:'cfa',2:'cinq',3:'cinquante',4:'de',5:'deux',6:'dix',7:'douze',
     8:'et',9:'fcfa',10:'franc',11:'huit',12:'mille',13:'million',14:'neuf',15:'onze',
     16:'quarante',17:'quatorze',18:'quatre',19:'quinze',20:'seize',21:'sept',22:'six',
     23:'soixante',24:'treize',25:'trente',26:'trois',27:'un',28:'vingt'}


def predict(img,model):
    data_transforms = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225] )
    ])
    image=Image.fromarray(img.copy())
    image = image.convert('RGB')
    imgblob = data_transforms(image)
    imgblob.unsqueeze_(dim=0)
    output = model(imgblob)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    return dic[prediction]

