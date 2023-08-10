import os


batch1 = dict(name = 'CCKS',
              dataInput = dict(rootPath = '/Users/floyd/Desktop/Research/NER-CRF/cctner/', 
              fpath = 'dataset/annoted/batch1/',
              filenames= ['一般项目', '病史特点', '诊疗经过', '出院情况'],
              orig_iden = '.txtoriginal.txt',
              anno_iden = '.txt'),
              dataAnno = dict(sep = '\t',
                              fLabel = {'症状和体征': 'Sy','身体部位': 'Bo',
                                        '检查和检验': 'Ch','治疗': 'Tr', '疾病和诊断': 'Si'},
                              start = 0))

import os

def generateOriAn(fpath, filenames, orig_iden, anno_iden, rootPath):
    OriAn = dict()
    for filename in filenames:
        path = rootPath+fpath+filename
        L =  [i for i in os.listdir(path) if orig_iden in i or anno_iden in i]
        Orig = [i for i in L if orig_iden in i]
        Anno = [i for i in L if anno_iden in i]
        if len(Orig) == len(L)/2:
            Anno = [i for i in L if i not in Orig]
        elif len(Anno) == len(L)/2:
            Orig = [i for i in L if i not in Anno]
        else:
            print('Wrong !!!')

        assert len(Orig) == len(Anno)   
        
        d_orig = {}
        for i in Orig:
            d_orig[i.replace(orig_iden, '')] = i
            
        d_anno = {}
        for i in Anno:
            d_anno[i.replace(anno_iden, '')] = i
        assert d_anno.keys() == d_orig.keys()
        
        OriAn_L = []
        for k in d_anno:
            OriAn_L.append({'originalFilePath':path +'/'+ d_orig[k], 
                            'annotedFilePath' :path +'/'+ d_anno[k]})
        OriAn[filename] =  OriAn_L
    return OriAn

batch = batch1
d = generateOriAn(**batch['dataInput'])
d['出院情况'][:5]

batch2 = dict(name = 'LUOHU',
              dataInput = dict(rootPath = '/Users/floyd/Desktop/Research/NER-CRF/cctner/',
              fpath = 'dataset/annoted/batch2/',
              filenames= ['text'],
              orig_iden = '.txt',
              anno_iden = '_StandardFormat.txt'),
              dataAnno = dict(sep = '\t',
                              fLabel = {'症状': 'Sy', '身体部位以及器官': 'Bo',
                                        '检查项目': 'Ch',
                                        '治疗手段': 'Tr', '疾病名称': 'Si','疾病类型': 'DT',
                                        '不确定'  : 'unct' },
                              start = 1))

batch2

batch = batch2
d = generateOriAn(**batch['dataInput'])
d['text'][:5]

for k in d:
    for file_d in d[k]:
        anno = file_d['annotedFilePath']
        Lines = []
        with open(anno, 'r') as f:
            for i in f.readlines():
                Lines.append(i.replace(' ', '\t'))
        with open(anno, 'w') as f:
            for i in Lines:
                f.write(i)

