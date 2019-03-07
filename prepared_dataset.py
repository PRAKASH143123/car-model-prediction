'''
Change the data structure to a way that can be used by tensorflow
'''
import pandas as pd
import os
from shutil import copyfile
from tqdm import tqdm

names = pd.read_csv("data/names.csv")[["name"]]
ann_train = pd.read_csv("data/anno_train.csv")

folders = set()

for idx, row in tqdm(ann_train.iterrows(), total=len(ann_train)):
    class_name = names.iloc[int(row["label"]) - 1]["name"]
    class_name = class_name.replace(" ", "_").replace("/", "_").lower()
    if class_name not in folders:
        os.mkdir(path=os.getcwd() + "/data/prepared_training_set/{}".format(class_name))
        folders.add(class_name)
    copyfile(
        src=os.getcwd() + "/data/train/{}".format(row["name"]),
        dst=os.getcwd()
        + "/data/prepared_training_set/{}/{}".format(class_name, row["name"]),
    )
