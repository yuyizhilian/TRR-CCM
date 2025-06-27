import time
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process

from .calibration_layer import PrototypicalCalibrationBlock

from PIL import Image
import numpy as np
# from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
import os
import matplotlib.pyplot as plt
import torchvision
from typing import Callable, List, Tuple


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    return image

class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        instances = model_outputs["instances"]
        boxes = instances.pred_boxes.tensor
        labels = instances.pred_classes
        scores = instances.scores

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, boxes)
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and labels[index] == label:
                score = ious[0, index] + scores[index]
                output = output + score
        return output

def inference_on_dataset(model, data_loader, evaluator, cfg=None):

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    # # 清空
    # with open('test/false_detection.txt', 'w') as file:
    #     file.write('')
    # with open('test/mis_detection.txt', 'w') as file:
    #     file.write('')

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            #visualize(outputs=outputs, inputs=inputs, cfg=cfg)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()

    # with inference_context(model):
    #     for idx, inputs in enumerate(data_loader):
    #         image_path = inputs[0]['file_name']
    #         image_name = image_path.split('/')[-1]
    #         image = preprocess_image(image_path)

    #         outputs = model(inputs)
    #         if cfg.TEST.PCB_ENABLE:
    #             outputs = pcb.execute_calibration(inputs, outputs)
    #         # 获取目标框和标签
    #         instances = outputs[0]["instances"].to("cpu")
    #         boxes = instances.pred_boxes.tensor.detach().numpy()
    #         labels = instances.pred_classes.detach().numpy()
    #         scores = instances.scores.detach().numpy()

    #         # 设置目标层
    #         target_layer = model.backbone.res4[-1].conv3
    #         # target_layer = model.affine_rcnn
    #         for p in model.affine_rcnn.parameters():
    #             p.requires_grad = True

    #         # 初始化 GradCAM
    #         cam = GradCAM_(model=model, target_layers=[target_layer], use_cuda=False)

    #         # 生成热力图
    #         targets = [FasterRCNNBoxScoreTarget(labels=torch.tensor(labels), bounding_boxes=torch.tensor(boxes))]
    #         grayscale_cam = cam(input_tensor=inputs, targets=targets)

    #         # 只取第一张图片的结果
    #         grayscale_cam = grayscale_cam[0, :]

    #         # 加载原始图像用于叠加热力图
    #         rgb_img = image / 255.0
    #         cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    #         tar_dir = f'{cfg.OUTPUT_DIR}/heat_map'
    #         os.makedirs(tar_dir, exist_ok=True)
    #         plt.imsave(f'{tar_dir}/{image_name}', cam_image)
    #filter_vis(cfg=cfg)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
