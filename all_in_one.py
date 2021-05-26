import os
import subprocess
import numpy as np
import argparse
from tqdm import tqdm

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description="SyncNet for Neural Dubber")
parser.add_argument('--dataset', type=str, default='chem', help='Dataset type')
parser.add_argument('--metafile', type=str, default='', help='metadata csv file path')
parser.add_argument('--pathfile', type=str, default='', help='file contain the test sample path')
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
stdout_file = open(f"data/nb_eval/{dataset}/stdout.txt", 'w')
for model_name, path_format in test_data_path_formats.items():
    print(f"Dataset: {dataset}, model_name: {model_name}")
    for name in tqdm(item_ids):
        videofile = path_format.format(name)
        ref_name = name
        data_dir = f"data/nb_eval/{dataset}/{model_name}"
        run_syncnet_all(videofile, ref_name, data_dir, stdout_file)

stdout_file.close()
print("Finish!")
