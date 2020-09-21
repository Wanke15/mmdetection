import os

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, set_random_seed
import mmcv


class SatellitePredictor:
    def __init__(self):
        cfg = mmcv.Config.fromfile('../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

        # modify num classes of the model in box head
        cfg.model.roi_head.bbox_head.num_classes = 10

        checkpoint_file = '../satellite_exps/epoch_6.pth'

        self.model = init_detector(cfg, checkpoint_file, device='cuda:0')
        self.model.cfg = cfg
        self.model.CLASSES = ('airplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
                              'basketball_court', 'ground_track_field', 'harbor', 'bridge', 'vehicle')

        # first time initialization
        self.predict('./uploads/4c510a77-1777-48c7-878a-0ef93c2438d8.png', 0.3)

    def predict(self, img, threshold):

        result = inference_detector(self.model, img)

        res_img = self.model.show_result(img, result, score_thr=threshold, show=False)

        _image_name = img.split('\\')[-1].split('.')[0]
        # maybe only happen for the first time initialization
        if not _image_name:
            return

        _image_name = _image_name + '.png'

        out = os.path.join('results/', _image_name)

        mmcv.imwrite(res_img, out)

        return _image_name
