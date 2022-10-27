import cv2
from PIL import Image
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import os
import imagehash
from binary import preprocessing
from predict import predict
import tensorflow as tf
from text_to_num import text2num

with tf.device('/cpu:0'):
    digit = tf.keras.models.load_model('../models/digit.h5')

seg = torch.hub.load('yolov5/', 'custom', source='local',
                     path='../models/amount.pt', force_reload=True, device='cpu')
model = torch.hub.load('yolov5/', 'custom', source='local',
                       path='../models/best.pt', force_reload=True, device='cpu')
seg_dd=torch.hub.load('yolov5/', 'custom', source='local', path = '../models/seg_dd.pt', force_reload = True, device='cpu')
seg_date=torch.hub.load('yolov5/', 'custom',source='local', path='../models/seg_date.pt', force_reload=True, device='cpu')

def detect(path, model):
    start = time.time()
    img = cv2.imread(path)
    d = img.copy()
    result = model(d, size=640)
    result.show()
    print(time.time()-start)


def inference(img):
    result = model(img.copy(), size=640)
    df = result.pandas().xyxy[0]
    df = df.sort_values(by=['name', 'xmin'], ignore_index=True)
    return df


def get_bbox(img, df, name):
    Box = []
    if name == 'bar':
        bbox = df.loc[df['name'] == name][[
            'xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)

    elif (name == 'word1') or (name == 'word2'):
        bbox = df.loc[(df['name'] == name) & (df['confidence'] >= 0.40)][[
            'xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)

    else:
        bbox = df.loc[(df['name'] == name) & (df['confidence'] >= 0.45)][[
            'xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
    for i in range(len(bbox)):
        x1, y1, x2, y2 = bbox[i]
        Box.append(img.copy()[y1:y2, x1:x2])
    return Box


class detector:
    def __init__(self, path_image, vgg):
        self.img = cv2.imread(path_image)
        self.df = inference(self.img)
        self.sign = get_bbox(self.img, self.df, 'sign')
        self.amount = get_bbox(self.img, self.df, 'amount')
        self.word = get_bbox(self.img, self.df, 'word1')
        w = get_bbox(self.img, self.df, 'word2')
        self.word.extend(w)
        self.date = get_bbox(self.img, self.df, 'date')
        self.place = get_bbox(self.img, self.df, 'place')
        self.name = get_bbox(self.img, self.df, 'name')
        self.plt_img = Image.fromarray(self.img)
        self.word = [preprocessing(img.copy()) for img in self.word]
        self.montant = [predict(img, vgg) for img in self.word]

    def detect_bar(self):
        if len(self.df[self.df['name'] == 'bar']) == 0:
            return False
        else:
            return True

    def verif_sign(self, imgs_database):
        if len(self.sign) == 0:
            return 'Pas de signature trouve'
        else:
            score = np.zeros((len(self.sign), len(imgs_database)))
            for i in range(len(self.sign)):
                for j in range(len(imgs_database)):
                    img1 = Image.fromarray(self.sign[i])
                    hash1 = imagehash.average_hash(img1)
                    hash2 = imagehash.average_hash(
                        Image.open(imgs_database[j]))
                    # diff limite est 16 bits experimentalement
                    diff = abs(hash1 - hash2)
                    prob = 1-(float(diff)/32)
                    score[i][j] = prob*100

            score = score.max(axis=1).flatten()
            score[score < 0] = 0

            return list(score)

    def montant_lettre(self):

        return ' '.join(self.montant)

    def montant_chiffre(self):
        a = self.amount[0]
        a = preprocessing(a.copy())
        result = seg(a.copy(), size=640)
        df = result.pandas().xyxy[0]
        df = df.sort_values(by=['name', 'xmin'], ignore_index=True)
        chars = get_bbox(a, df, 'A')
        dic = dict(enumerate(list("0123456789A")))
        number = []
        for image in chars:
            img = cv2.resize(image, (28, 28))
            image = img.reshape(1, 28, 28, 1)
            pred = dic[digit.predict(image).argmax()]
            if pred == 'A':
                pred = ''
            number.append(pred)
        number = "".join(number)
        mont = int(number)
        return mont
    
    def pred_date(self):
        try:
            a=self.date[0]
            img=preprocessing(a.copy())
            result=seg_date(img,size=640)
            df=result.pandas().xyxy[0]
            df=df.sort_values(by='xmin',ignore_index=True)
            bbox=get_bbox(img,df,'d')
            final=[]
            for i in  range(len(bbox)):
                b=bbox[i]
                result=seg_dd(b.copy(),size=640)
                df=result.pandas().xyxy[0]
                df=df.sort_values(by='xmin',ignore_index=True)
                chars=get_bbox(b,df,'d')
                format=[]
                for image in chars:
                    img=cv2.resize(image,(28,28))
                    img=img.reshape(1,28,28,1)
                    if digit.predict(img).argmax()!=10:
                        pred=str(digit.predict(img).argmax())
                    else:
                        pred=''
                    format.append(pred)
                format="".join(format)
                final.append(format)
                if i!=len(bbox)-1:
                    final.append('/')
            final="".join(final)
        except:
            final=''
        return final

    def correct_mont(self,l): #nouveau
        f=[]
        i=0
        while i<len(l)-1:
            if (l[i] =='dix' and (l[i+1] in {'neuf','sept','huit'})) or (l[i] =='quatre'and l[i+1]=='vingt'):
                f.append(l[i]+'-'+l[i+1])
                i+=2
            else:
                f.append(l[i])
                i+=1
        if i==len(l)-1:
            f.append(l[i])
        return f

    def conforme(self):  #nouveau
        try:
            d=self.montant
            d=[i for i in d if i not in {'de','franc','fcfa','cfa','et'}]
            d=self.correct_mont(d)
            d=' '.join(d)
            d=text2num(d,'fr')
            
            if d==self.montant_chiffre():
                return True
            else:
                return False
        except:
            return False

############# Test ####################
# liste = []
# path = '../ecobank/'
# t = os.listdir(path)
# for i in t:
#     liste.append(os.path.join(path, i))

# a = liste[6]
# # detect(a,model) # Indique un apercu de comment l'algorithme récupère les données

# # Quelques cheques testés :146-147-148-150-184-188-189-190-191-192-195-203-205-209-210-213-215-216

# d = detector(a, vgg)
# d.plt_img

# d.montant_lettre()

# d.detect_bar()
