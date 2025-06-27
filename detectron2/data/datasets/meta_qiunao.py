import os
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
import math

__all__ = ["register_meta_qiunao"]

num_base_class = 5

def load_filtered_voc_instances(
    name: str, dirname: str, split: str, classnames: str
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "vocsplit_qiunao")
        shot = name.split("_")[-2].split("shot")[0]
        seed = int(name.split("_seed")[-1])
        split_dir = os.path.join(split_dir, "seed{}".format(seed))
        # 让新类在前面
        for cls in classnames[::-1]:

            with PathManager.open(
                os.path.join(
                    split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                )
            ) as f:
                fileids_ = np.loadtxt(f, dtype=str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
                ]
                for fileid in fileids_:
                    if fileid in fileids:
                        fileids[fileid].append(cls)
                    else:
                        fileids[fileid] = [cls]
    else:
        with PathManager.open(
            os.path.join(dirname, "ImageSets", "Main", split + ".txt")
        ) as f:
            fileids = np.loadtxt(f, dtype=str)

    dicts = []
    if is_shots:
        shot = int(name.split("_")[-2].split("shot")[0])
        cls_shots = [0 for x in range(0, len(classnames))]
        filecount = 0
        for fileid, classes in fileids.items():
            filecount += 1

            dicts_ = []
            year = "qiunao"
            dirname = os.path.join("datasets", "VOC_{}".format(year))
            anno_file = os.path.join(
                dirname, "Annotations", fileid + ".xml"
            )
            jpeg_file = os.path.join(
                dirname, "JPEGImages", fileid + ".jpg"
            )

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
                "annotations": [],
            }

            for obj in tree.findall("object"):
                
                cls_ = obj.find("name").text

                if filecount > (len(classnames) - num_base_class) * shot:
                    # 当前为基类的one-shot图片
                    if cls_ not in classes:
                        continue
                    index = classnames.index(cls_)
                    if cls_shots[index] >= shot:
                        continue
                    
                else:
                    if cls_ in classes:
                        index = classnames.index(cls_)
                        if cls_shots[index] == shot:
                            continue
                    elif classnames.index(cls_) >= num_base_class:
                        # 不需要保留的新类
                        continue
                    # else:
                    #     index = classnames.index(cls_)
                    #     if cls_shots[index] >= shot:
                    #         continue

                index = classnames.index(cls_)
                cls_shots[index] += 1

                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instance = {
                    "category_id": classnames.index(cls_),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                }
                
                r["annotations"].append(instance)
            if len(r["annotations"]) == 0:
                continue
            dicts.append(r)

    else:
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append(
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                )
            r["annotations"] = instances
            dicts.append(r)
    return dicts


def register_meta_qiunao(
    name, metadata, dirname, split, year, keepclasses, sid
):
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"][sid]

    DatasetCatalog.register(
        name,
        lambda: load_filtered_voc_instances(
            name, dirname, split, thing_classes
        ),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        year=year,
        split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid],
    )
