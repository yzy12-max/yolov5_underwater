
import torch


path = '/home/user11/Desktop/yolov5-master/weights/bestg1.pt'
data = torch.load(path,map_location='cpu')
data['model'] = None
# data['optimizer'] = None
torch.save(data,'/home/user11/Desktop/yolov5-master/weights/bestg2.pt')
pass