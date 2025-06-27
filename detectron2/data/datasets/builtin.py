import os
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from .meta_qiunao import register_meta_qiunao
from .meta_4ch import register_meta_4ch
from .meta_3vt import register_meta_3vt
from .meta_head import register_meta_head


def register_all_qiunao(root="datasets"):

    METASPLITS = [
        ("qiunao_trainval_base1", "VOC_qiunao", "trainval", "base1", 1),
        ("qiunao_trainval_base2", "VOC_qiunao", "trainval", "base2", 2),
        ("qiunao_trainval_base3", "VOC_qiunao", "trainval", "base3", 3),
        ("qiunao_trainval_all1", "VOC_qiunao", "trainval", "base_novel_1", 1),
        ("qiunao_trainval_all2", "VOC_qiunao", "trainval", "base_novel_2", 2),
        ("qiunao_trainval_all3", "VOC_qiunao", "trainval", "base_novel_3", 3),
        ("qiunao_test_base1", "VOC_qiunao", "test", "base1", 1),
        ("qiunao_test_base2", "VOC_qiunao", "test", "base2", 2),
        ("qiunao_test_base3", "VOC_qiunao", "test", "base3", 3),
        ("qiunao_test_novel1", "VOC_qiunao", "test", "novel1", 1),
        ("qiunao_test_novel2", "VOC_qiunao", "test", "novel2", 2),
        ("qiunao_test_novel3", "VOC_qiunao", "test", "novel3", 3),
        ("qiunao_test_all1", "VOC_qiunao", "test", "base_novel_1", 1),
        ("qiunao_test_all2", "VOC_qiunao", "test", "base_novel_2", 2),
        ("qiunao_test_all3", "VOC_qiunao", "test", "base_novel_3", 3),
    ]
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in ['qiunao']:
                    for seed in range(30):
                        seed = "_seed{}".format(seed)
                        name = "{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC_{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )
                        name = "{}_gcn_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007
        register_meta_qiunao(
            name,
            _get_builtin_metadata("qiunao"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"




# register_all_coco()
# register_all_voc()
register_all_qiunao()
