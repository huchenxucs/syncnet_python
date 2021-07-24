import os
import subprocess
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import traceback

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description="SyncNet for Neural Dubber")
parser.add_argument('--dataset', type=str, default='chem', help='Dataset type')
parser.add_argument('--metafile', type=str, default='', help='metadata csv file path')
parser.add_argument('--pathfile', type=str, default='', help='file contain the test sample path')
parser.add_argument('--model', type=str, default='', help='model name')
parser.add_argument('--work_num', type=int, default=8, help='cpu number')
parser.add_argument('--gpu_num', type=int, default=1, help='cpu number')

opt = parser.parse_args()
assert opt.dataset in ['chem', 'lrs2']
dataset = opt.dataset

"""
python run_pipeline.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_syncnet.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
python run_visualise.py --videofile /path/to/video.mp4 --reference name_of_video --data_dir /path/to/output
"""


def run_syncnet_all(videofile, ref_name, data_dir, stdout=None):
    cmd1 = f"python3 run_pipeline.py --videofile {videofile} --reference {ref_name} --data_dir {data_dir}"
    cmd2 = f"python3 run_syncnet.py --videofile {videofile} --reference {ref_name} --data_dir {data_dir}"
    subprocess.check_call(cmd1, stdout=stdout, shell=True)
    subprocess.check_call(cmd2, stdout=stdout, shell=True)


def get_path_format(pathfile):
    res = dict()
    with open(pathfile, 'r') as f:
        paths = f.readlines()
    for p in paths:
        k, v = p.strip().split('|')
        res[k] = v
    return res


metadata_path = opt.metafile
with open(metadata_path, 'r') as f:
    item_ids = f.readlines()
item_ids = [x.strip().split('|')[0].strip() for x in item_ids]

test_data_path_formats = get_path_format(opt.pathfile)
os.makedirs(f"data/nb_eval/{dataset}", exist_ok=True)
stdout_file = open(f"data/nb_eval/{dataset}/stdout_{opt.model}.txt", 'w')

new_path_formats = {}
m = opt.model
new_path_formats[m] = test_data_path_formats[m]

# pool = Pool(opt.work_num)
# jobs = []
for model_name, path_format in new_path_formats.items():
    print(f"Dataset: {dataset}, model_name: {model_name}")
    for idx, name in enumerate(tqdm(item_ids)):
        videofile = path_format.format(name)
        if not os.path.exists(videofile):
            print(f"| {dataset} {model_name} {name} is not exist!")
            continue

        try:
            ref_name = name
            data_dir = f"data/nb_eval/{dataset}/{model_name}"
            result_path = os.path.join(data_dir, 'pywork', ref_name, 'result.pckl')
            if not os.path.exists(result_path):
                run_syncnet_all(videofile, ref_name, data_dir, stdout_file)
                # jobs.append(pool.apply_async(run_syncnet_all,
                #                              args=(idx % opt.gpu_num, videofile, ref_name, data_dir, stdout_file)))
        except Exception as e:
            traceback.print_exc()
            print(e)

# pool.close()
# result_list_tqdm = []
# for job in tqdm(jobs):
#     result_list_tqdm.append(job.get())
# pool.join()
stdout_file.close()
print("Finish!")
