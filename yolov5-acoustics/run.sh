# bash underwater-acoustics
cd /home/user11/Desktop/yolov5-acoustics/
# transform tensorrt
#python export.py --data data/underwater_voice.yaml --weights weights/bestsl69.pt --imgsz 960 --device 0 --workspace 7 --iou-thres 0.65 --conf-thres 0.001 --half --include engine
# don't need onnx,only save engine
#rm -r /home/user11/Desktop/yolov5-acoustics/weights/bestsl69.onnx
# if the file exists,rm ./path/file
rm -r /home/user11/Desktop/yolov5-acoustics/runs/detect/
# generate results and fps
python main_test.py --weights weights/bestsl69.engine --source /home/user11/Desktop/sound_images/ --data data/underwater_voice.yaml --imgsz 960 --conf-thres 0.001 --iou-thres 0.65 --device 0 --save-txt --save-conf --nosave --name sub_acoustics