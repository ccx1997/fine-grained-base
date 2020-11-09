# -*- coding=utf-8 -*-
"""To convert CUB-200-2011 txt files into normal form as:
   train_list.txt  test_list.txt  class_map.txt
   where the content of each line is
   <image_name class_id> and <class_id class_name>
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def split(imageid_txt, splitid_txt, train_txt, test_txt):
    imageid = pd.read_table(imageid_txt, header=None, delim_whitespace=True)
    splitid = pd.read_table(splitid_txt, header=None, delim_whitespace=True)
    trainid = open(train_txt, 'w')
    testid = open(test_txt, 'w')
    for i in range(splitid.shape[0]):
        istrain = splitid.iloc[i, 1]
        name = imageid.iloc[i, 1]
        idc = int(name[:3]) - 1
        if istrain == 1:
            trainid.writelines(name + ' ' + str(idc) + '\n')
        else:
            testid.writelines(name + ' ' + str(idc) + '\n')
    trainid.close()
    testid.close()

def id_class(class_txt, mapping_txt):
    all_class = pd.read_table(class_txt, header=None, delim_whitespace=True)
    with open(mapping_txt, 'w') as m:
        for i in range(all_class.shape[0]):
            classid, classname = all_class.iloc[i, :]
            m.writelines(str(classid - 1) + ' ' + classname[4:] + '\n')

if __name__ == '__main__':
    imageid_txt, splitid_txt, train_txt, test_txt = [
        "/workspace/datasets/CUB_200_2011/CUB_200_2011/images.txt",
        "/workspace/datasets/CUB_200_2011/CUB_200_2011/train_test_split.txt",
        "/workspace/datasets/CUB_200_2011/CUB_200_2011/train_list.txt",
        "/workspace/datasets/CUB_200_2011/CUB_200_2011/test_list.txt"]
    class_txt, mapping_txt = [
        "/workspace/datasets/CUB_200_2011/CUB_200_2011/classes.txt",
        "/workspace/datasets/CUB_200_2011/CUB_200_2011/class_map.txt"]
    split(imageid_txt, splitid_txt, train_txt, test_txt)
    id_class(class_txt, mapping_txt)
    print("-----Successfully generation!-----")