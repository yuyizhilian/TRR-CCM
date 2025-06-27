import os
import math
import argparse
import numpy as np
import pandas as pd
import os

novel_categories_all = {
    '4ch': {
        1: ['atrium dextrum', 'ventriculus sinister'],
        2: ['interventricular septum','spine'],
        3: ['ventriculus dexter','aorta descendens'],
    },
    'qiunao': {
        1: ['septum pellucidum cavity', 'The thalamus'],
        2: ['the brain lateral fissure', "choroid plexus"],
        3: ["head line", 'the cerebellum'],
    },
    '3vt': {
        1: ['T', 'AOA'],
        2: ['DAO', 'PTDA'],
        3: ['SP', 'SVC'],
    },
    'head':{
        1: ['C','BM'],
        2: ['CP', 'CSP'],
        3: ['S', 'LS'],
    }
}

def conLineToList(line):
    values = line.split('|')
    if '' in values:
        values.remove('')
    if '' in values:
        values.remove('')
    values = [value.strip() for value in values]
    return values

def getResult(datasets, classes_map, output_map, lib_map):
    columns = [['model'], ['class']]
    for split in range(2, 4):
        for shot in [1,2,3,5,10]:
            column = [f'split{split}', f'shot{shot}']
            columns.append(column)
    columns = [tuple(column) for column in columns]

    for dataset in datasets:
        all_results = []
        classes = classes_map[dataset]
        all_map_results = []
        for model in output_map:
            results = []
            results.append([model])
            for i in range(1, len(classes)):
                results[0].append('')
            template = output_map[model]
            results.append(classes.copy())
            map_result = []
            for split in range(2,4):
                for shot in [1,2,3,5,10]:
                    output_dir = f'output45/{dataset}/finetune{split}/{shot}shot_seed1/s2'
                    novel_classes = novel_categories_all[dataset][split]
                    base_map = 0
                    novel_map = 0
                    if model == 'digeo':
                        files = os.listdir(output_dir)
                        for filename in files:
                            if 'distill' in filename:
                                output_dir = f'{output_dir}/{filename}'
                    if lib_map[model] == 'detectron2':
                        log_path = f'{output_dir}/log.txt'
                        print(log_path)
                        lineinfos = open(log_path).readlines()
                        best_val = ''
                        with open(f'{output_dir}/best_AP50.txt', 'r') as f:
                            best_val = f.read()
                        resultIndex = 0
                        lineinfos = open(f'{output_dir}/log.txt').readlines()
                        for i, line in enumerate(lineinfos):
                            if f',{float(best_val):.4f},' in line:
                                resultIndex = i
                        # 每一种类别
                        resultIndex = resultIndex - 8
                        res_info = lineinfos[resultIndex].strip()
                        values_temp = conLineToList(res_info)
                        values_temp = [float(value) for value in values_temp]
                        res_info = lineinfos[resultIndex - 2].strip()
                        classes_temp = conLineToList(res_info)
                        values = []
                        for cla in classes:
                            index = classes_temp.index(cla)
                            value = values_temp[index]
                            if cla in novel_classes:
                                novel_map += value
                            else:
                                base_map += value
                            values.append(value)
                        results.append(values.copy())
                        # map
                        res_info = lineinfos[resultIndex + 4].strip()
                        values = conLineToList(res_info)
                        map = float(values[1])
                    else:
                        files = os.listdir(output_dir)
                        files.sort()
                        log_path = ''
                        for filename in files:
                            if filename.endswith('.log'):
                                log_path = filename
                        lineinfos = open(f'{output_dir}/{log_path}').readlines()
                        print(f'{output_dir}/{log_path}')
                        # if 'icpe/output1/qiunao/split1/shot10/20231224_004135.log' == f'{output_dir}/{log_path}':
                        #     print('lll')
                        # 每一种类别
                        aps = [0 for i in range(0, len(classes))]
                        resultIndex = -8 - len(classes)
                        base_map = 0
                        novel_map = 0
                        if model in ['vfa']:
                            resultIndex = resultIndex + 1
                        if lineinfos[-1] != '\n':
                            resultIndex = resultIndex + 1
                        for i in range(resultIndex, resultIndex + len(classes)):
                            res_info = lineinfos[i].strip()
                            values_temp = conLineToList(res_info)
                            classname = values_temp[0]
                            ap = float(values_temp[-1])
                            index = classes.index(classname)
                            aps[index] = ap
                            if classname in novel_classes:
                                novel_map += ap
                            else:
                                base_map += ap
                        results.append(aps)
                        # map
                        res_info = lineinfos[resultIndex + len(classes) + 1].strip()
                        values = conLineToList(res_info)
                        map = float(values[-1])
                    base_map = round(base_map / (len(classes) - len(novel_classes)) , 3)
                    novel_map = round(novel_map / len(novel_classes) , 3)
                    scale = 1
                    if map < 1:
                        scale = 100
                    map_result.append(map * scale)
                    map_result.append(base_map * scale)
                    map_result.append(novel_map * scale)
            results[0].append(model)
            results[1].append('map')
            results[0].append(model)
            results[1].append('base_map')
            results[0].append(model)
            results[1].append('novel_map')
            for i in range(2, len(results)):
                case_index = i - 2
                results[i].append(map_result[case_index * 3])
                results[i].append(map_result[case_index * 3 + 1])
                results[i].append(map_result[case_index * 3 + 2])
            if len(all_results) == 0:
                all_results = results
            else:
                for column in range(0, len(all_results)):
                    all_results[column] += results[column]
            all_map_results.append(map_result)
        all_map_results = np.array(all_map_results)
        # maxs = np.argmax(all_map_results, axis=0)
        # maxs = [value+1 for value in maxs]
        # all_results[0].append('')
        # all_results[1].append('max')
        # for i in range(0, len(maxs)):
        #     all_results[i + 2].append(maxs[i])
        # print(all_results)
        np_array = np.array(all_results).transpose()
        df = pd.DataFrame.from_records(np_array)
        df.columns = pd.MultiIndex.from_tuples(columns)
        with pd.ExcelWriter(f"{dataset}_result_complete_base_novel.xlsx") as writer:
            df.to_excel(writer, sheet_name=dataset)


output_map = {
    # 'dcfs': '{dataset}/dcfs_gfsod_r101_novel{split}/tfa-like-DC/{shot}shot_seed1',
    'defrcn': '{dataset}/defrcn_gfsod_r101_novel{split}/{shot}shot_seed1_s1',
    # 'digeo': '{dataset}/{split}_{shot}',
    # 'icpe': '{dataset}/split{split}/shot{shot}',
    # 'mfdc': '{dataset}/mfdc_gfsod_novel{split}/tfa-like/{shot}shot_seed1',
    # 'mmfewshot': '{dataset}/split{split}/shot{shot}',
    # 'tfa': '{dataset}/split{split}/shot{shot}',
    # 'vfa': '{dataset}/split{split}/{shot}shot',
}

lib_map = {
    # 'dcfs': 'detectron2',
    'defrcn': 'detectron2',
    # 'digeo': 'detectron2',
    # 'icpe': 'mmfewshot',
    # 'mfdc': 'detectron2',
    # 'mmfewshot': 'mmfewshot',
    # 'tfa': 'detectron2',
    # 'vfa': 'mmfewshot',
}
#'3vt', 'qiunao',
datasets = ['head']

classes_map = {
    'fetus': ['thalami', 'midbrain', 'NT', 'IT', 'CM', 'palate', 'nasal bone', 'nasal tip', 'nasal skin'],
    'qiunao': ['The thalamus', 'the brain lateral fissure', 'choroid plexus', 'septum pellucidum cavity', 'head line', 'skull strong echoes ring', 'the cerebellum'],
    '4ch': ['atrium sinistrum','atrium dextrum','ventriculus sinister','ventriculus dexter','interventricular septum','atrioventricular septum','spine','aorta descendens','rib'],
    '3vt': ['DAO', 'SP', 'PTDA', 'T', 'SVC', 'AOA'],
    'head':['T', 'LS', 'CP', 'CSP', 'S', 'C','BM']
}

getResult(datasets, classes_map, output_map, lib_map)