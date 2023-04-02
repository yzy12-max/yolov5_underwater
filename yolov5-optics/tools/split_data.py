import os
import random
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--xml_path', type=str, help='input xml label path')
# parser.add_argument('--txt_path', type=str, help='output txt label path')
# opt = parser.parse_args()
# xml_path = '/data/datasets/yolo/Annotations'
# txt_path = '/data/datasets/yolo/ImageSets/Main'
xml_path = '/data/datasets/car/Annotations'
txt_path = '/data/datasets/car/ImageSets/Main'
trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = xml_path
txtsavepath = txt_path
total_xml = os.listdir(xmlfilepath)  # os.listdir是固定了随机值的,如果采用没固定的考虑加随机种子random.seed(11)
if not os.path.exists(txtsavepath):
  os.makedirs(txtsavepath)

num=len(total_xml)
list=range(num)

ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')

for i in list:
    name=total_xml[i][:-4]+'\n'
    ftrainval.write(name)
    if i >= int(num*train_percent):
        fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
