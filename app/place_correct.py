import numpy as np
import Levenshtein as lev

def levCalclulate(str1, str2):
    Distance = lev.distance(str1, str2)
    Ratio = lev.ratio(str1, str2)
    return Distance


def answers(word):
    place=['DK','DL','FK','KA',"KL",'KE',"KD",'LG','SL',"SE","TC","TH",'ZG',"ZIG",'DKR',"MBR"
    ,"DAKAR","ZIGUINCHOR",'TOUBA',"MBOUR","FATICK","THIES","LOUGA","DIOUBEL",'KEDOUGOU',"KOLDA","TAMBACOUNDA","STLOUIS", "SAINTLOUIS",'KAOLACK','JOAL','GOREE','MATAM',"SEDHIOU","SALY"]
    dic=dict(enumerate(place))
    l=[levCalclulate(word,ch) for ch in place]
    if min(l)>4:
        return 'Pas identifie'
    return dic[np.argmin(l)]