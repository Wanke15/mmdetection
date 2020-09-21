import os
import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class SatelliteDataset(CustomDataset):
    CLASSES = ('airplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
               'basketball_court', 'ground_track_field', 'harbor', 'bridge', 'vehicle')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file

        image_list = mmcv.list_from_file(self.ann_file)

        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.jpg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]

            data_info = dict(filename=f'{image_id}.jpg', width=width, height=height)

            # load annotations
            label_prefix = self.img_prefix.replace('train', 'gt_train')
            label_prefix = label_prefix.replace('test', 'gt_test')
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))

            content = [line.strip().replace('(', '').replace(')', '').replace(' ', '').split(',') for line in lines if
                       line.strip()]
            bbox_names = [self.CLASSES[int(x[-1]) - 1] for x in content]
            bboxes = [[float(info) for info in x[:4]] for x in content]

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos


from mmcv import Config

cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

from mmdet.apis import set_random_seed, inference_detector, show_result_pyplot

# Modify dataset type and path
cfg.dataset_type = 'SatelliteDataset'
cfg.data_root = 'satellite_data/'

cfg.data.train.type = 'SatelliteDataset'
cfg.data.train.data_root = 'satellite_data/'
cfg.data.train.ann_file = 'train_anno.txt'
cfg.data.train.img_prefix = 'train'
# cfg.data.train.ann_file = 'test_anno.txt'
# cfg.data.train.img_prefix = 'test'

cfg.data.val.type = 'SatelliteDataset'
cfg.data.val.data_root = 'satellite_data/'
cfg.data.val.ann_file = 'test_anno.txt'
cfg.data.val.img_prefix = 'test'

cfg.data.test.type = 'SatelliteDataset'
cfg.data.test.data_root = 'satellite_data/'
cfg.data.test.ann_file = 'test_anno.txt'
cfg.data.test.img_prefix = 'test'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 10
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './satellite_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 2
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 2

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')


from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

if __name__ == '__main__':
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

    model.cfg = cfg

    img = mmcv.imread('satellite_data/train/001.jpg')

    result = inference_detector(model, img)
    show_result_pyplot(model, img, result)

    img = mmcv.imread('satellite_data/test/601.jpg')

    result = inference_detector(model, img)
    show_result_pyplot(model, img, result)
