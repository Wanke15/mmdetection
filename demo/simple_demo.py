from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
import time

img = 'demo.jpg'
start = time.time()
result = inference_detector(model, img)
end = time.time()
print('Time elapsed: ', end - start)

show_result_pyplot(model, img, result)
