import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', type=str, help='input xml label path')
parser.add_argument('--txt_path', type=str, help='output txt label path')
opt = parser.parse_args()

xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
  os.makedirs(txtsavepath)

num=len(total_xml)
list=range(num)

# ftrainval = open(txtsavepath + '/trainval.txt', 'w')
# ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
# fval = open(txtsavepath + '/val.txt', 'w')

for i in list:
    name=total_xml[i][:-4]+'\n'
    # fval.write(name)
    ftrain.write(name)


ftrain.close()
# fval.close()
# ftest.close()