from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco_panoptic_separated import dataloader
from ..common.models.panoptic_fpn import model
from ..common.train import train

model.backbone.bottom_up.freeze_at = 2
train.init_checkpoint = "detectron24://ImageNetPretrained/MSRA/R-50.pkl"
