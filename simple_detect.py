import torch
import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
dir = '/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/cat/JPEGImages/'
imgs = [dir + '000000.jpg']  # batch of images

#dir = '/home/zhaobotong/.cache/torch/hub/ultralytics_yolov5_master/data/images/'
#imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images
with torch.no_grad():
    # Inference
    start=time.clock()
    results = model(imgs)
    print(time.clock()-start)
results.print()  # or .show(), .save()
results.show()
print("123")