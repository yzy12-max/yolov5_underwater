# bash underwater-acoustics
cd /home/user11/Desktop/yolov5-optics/
# transform tensorrt
#python export.py --data data/underwater.yaml --weights weights/bestgl63.pt --imgsz 1280 --device 0 --workspace 7 --iou-thres 0.65 --conf-thres 0.001 --half --include engine
# don't need onnx,only save engine
#rm -r /home/user11/Desktop/yolov5-optics/weights/bestgl63.onnx
# if the file exists,rm ./path/file
rm -r /home/user11/Desktop/yolov5-optics/runs/detect/
# generate results and fps
python main_test.py --weights weights/bestgl63.engine --source /home/user11/Desktop/light_images/ --data data/underwater.yaml --imgsz 1280 --conf-thres 0.001 --iou-thres 0.65 --device 0 --save-txt --save-conf --nosave --name sub_optics