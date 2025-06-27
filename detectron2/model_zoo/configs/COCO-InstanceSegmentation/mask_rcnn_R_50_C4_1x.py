from ..common.train import train
from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.mask_rcnn_c4 import model

model.backbone.freeze_at = 2
train.init_checkpoint = "detectron24://ImageNetPretrained/MSRA/R-50.pkl"
