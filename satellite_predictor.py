from mmdet.apis import init_detector, inference_detector, show_result_pyplot, set_random_seed
import mmcv

cfg = mmcv.Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 10

checkpoint_file = './satellite_exps/epoch_6.pth'

model = init_detector(cfg, checkpoint_file, device='cuda:0')
model.cfg = cfg
model.CLASSES = ('airplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
                 'basketball_court', 'ground_track_field', 'harbor', 'bridge', 'vehicle')

import time

img = 'satellite_data/test/601.jpg'
start = time.time()
result = inference_detector(model, img)
end = time.time()
print('Time elapsed: ', end - start)

show_result_pyplot(model, img, result)
