import json
import os
import argparse
import pdb
import shutil
from tqdm import tqdm

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']
sonar_class = ["cube", "ball", "cylinder", "human body", "tyre", "circle cage", "square cage", "metal bucket", "plane", "rov"]
# test_json_raw = json.load(open("/data/fanjin2022/guang_forUser_A/testA.json", "r"))
# imgid2name = {}
# for imageinfo in test_json_raw['images']:
#     imgid = imageinfo['id']
#     imgid2name[imgid] = imageinfo['file_name']
# pdb.set_trace()

def coco2vistxt(json_path, out_folder):
    labels = json.load(open(json_path, 'r', encoding='utf-8'))
    for i in tqdm(range(len(labels))):
        file_name = str(labels[i]['image_id']).zfill(5)
        # pdb.set_trace()
        file_name = str(file_name)+ '.txt'
        # pdb.set_trace()
        with open(os.path.join(out_folder, file_name), 'a+', encoding='utf-8') as f:
            l = labels[i]['bbox']
            s = [round(i) for i in l]
            # pdb.set_trace()
            # # s[2] += s[0]
            # s[3] += s[1]
            # pdb.set_trace()
            # line =str(s)[1:-1].replace(' ','')+ ',' + str(labels[i]['score'])[:6] + ',' + str(labels[i]['category_id']) + ',' + str('-1') + ',' + str('-1')
            # if labels[i]['score'] < 0.1:
            # if labels[i]['score'] < 0.01:
            #    continue
            # line = str(underwater_classes[labels[i]['category_id']]) + ',' + str(s)[1:-1].replace(' ','') + ',' + str(labels[i]['score'])[:5]
            line = str(sonar_class[labels[i]['category_id']]) + ',' + str(s)[1:-1].replace(' ', '') + ',' + ('%.3f'%float(labels[i]['score']))
            f.write(line+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coco test json to visdrone txt file.')
    if not os.path.exists('/home/user7/Desktop/answerB/sub_acoustics/'):
        os.mkdir('/home/user7/Desktop/answerB/sub_acoustics/')
    shutil.rmtree('/home/user7/Desktop/answerB/sub_acoustics/') 
    os.mkdir('/home/user7/Desktop/answerB/sub_acoustics/') 
    parser.add_argument('-j', help='JSON file', default='/home/user7/Desktop/test_run_dir/sonar/sonar_s6_predictions.json')
    parser.add_argument('-o', help='path to output folder', default='/home/user7/Desktop/answerB/sub_acoustics/')

    args = parser.parse_args()
    print('start to convert josn file to txt')
    coco2vistxt(json_path=args.j,out_folder=args.o)
    print('finished')
