#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import Levenshtein as lev


# In[3]:


def answers(word):
    place=['DK','DL','FK','KA',"KL",'KE',"KD",'LG','SL',"SE","TC","TH",'ZG',"ZIG",'DKR',"MBR"
    ,"DAKAR","ZIGUINCHOR",'TOUBA',"MBOUR","FATICK","THIES","LOUGA","DIOUBEL",'KEDOUGOU',"KOLDA","TAMBACOUNDA","STLOUIS", "SAINTLOUIS",'KAOLACK','JOAL','GOREE','MATAM',"SEDHIOU","SALY"]
    dic=dict(enumerate(place))
    l=[lev.distance(word,ch) for ch in place]
    a=[i for i in range(len(l)) if l[i]==min(l)]
    p=dic[np.argmin(l)]
    if min(l)>=3:
        return 'Pas identifie'
    elif abs(len(word)-len(p))>2 and len(a)==1:
        return 'Pas identifie'
    elif len(a)!=1:
        new=[place[j] for j in a]
        v=[abs(len(word)-len(k)) for k in new ]
        if min(v)<=1:
            return new[np.argmin(v)]
        else:
            return 'Pas identifie'
    return p


# In[ ]:





# In[ ]:





# In[ ]:




