import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import json
import time
import os
import argparse
from model import NCD_con,NCD_gen, NCD
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import *
logger = Logger()
def log(str):
    print(str)
    logger.log(str + '\n')




if __name__ == '__main__':
    # argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epoch', default=40, type=int)
    argparser.add_argument('-lr', '--learning-rate', default=1e-3, type=float)
    argparser.add_argument('-b', '--batch-size', default=256, type=int)
    argparser.add_argument('-g','--gpu', type=str, default='cuda')
    argparser.add_argument('-i', '--index', default=0, type=int)
    argparser.add_argument('-mn', '--model-name', default="def", type=str)
    argparser.add_argument('-ds', '--dataset', default="python-large", type=str)
    args = argparser.parse_args()

    # parameters
    model_arch = args.model_name
    model_name = model_arch+"-NCD"
    dataset_name = args.dataset
    batch_size = args.batch_size
    epoch = args.epoch
    index = args.index
    learning_rate = args.learning_rate
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')


    # set log files
    current_time = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
    log_dir = f'../logs/{model_name}/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_name = log_dir+f"{model_name}_{current_time}.log"
    logger.set_filename(log_name)

    log(f"start training {model_name}, batchsize: {batch_size}, epoch: {epoch}, lr: {learning_rate}, device: {device}")
    log(f"log file saved in {log_name}")


    # get q-matrix in target course
    df_item = pd.read_csv(f"../data/{dataset_name}/item.csv")
    item2knowledge = {}
    knowledge_set = set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    dst_info = f"../data/{dataset_name}/info.json"
    with open(dst_info, 'r') as f:
        info = json.load(f)
    user_n = info['student_cnt']
    item_n = info['problem_cnt']
    knowledge_n = info['concept_cnt']
    log(f"dst_data: user_n: {user_n}, item_n: {item_n}, konwledge_n: {knowledge_n}")

    # read test dataset
    test_data = pd.read_csv(f"../data/{dataset_name}/test.csv")
    test_set = transform(test_data["user_id"], test_data["item_id"], item2knowledge, test_data["score"], batch_size)

    log(f"transform test_data done.")



    dir_path = f"../data/{dataset_name}/"
    
    train_data = pd.read_csv(dir_path+"train.csv")
    valid_data = pd.read_csv(dir_path+"valid.csv")

    similar_students = load_from_json(dir_path+"similar_students.json")
    similar_students = sort_and_fill_dict(similar_students, user_n)
    similar_exercises = load_from_json(dir_path+"similar_exercises.json")
    similar_exercises = sort_and_fill_dict(similar_exercises, item_n)
    log(f"data_dir: {dir_path}")

    train_set = transform(train_data["user_id"], train_data["item_id"], item2knowledge, train_data["score"], batch_size)
    log(f"transform train_data done.")
    valid_set = transform(valid_data["user_id"], valid_data["item_id"], item2knowledge, valid_data["score"], batch_size)
    log(f"transform valid_data done.")

    usrprf_embeds_path = f"../data/{dataset_name}/user_emb.pkl"
    itmprf_embeds_path = f"../data/{dataset_name}/item_emb.pkl"
    with open(usrprf_embeds_path, 'rb') as f:
        usr_emb = pickle.load(f)
    with open(itmprf_embeds_path, 'rb') as f:
        itm_emb = pickle.load(f)

    with open(dir_path+"stu_activity.json", 'r') as f:
        stu_ratio = json.load(f)
    
    with open(dir_path+"exer_activity.json", 'r') as f:
        exer_ratio = json.load(f)
    
    stu_ratio = [1 - x for x in stu_ratio]
    exer_ratio = [1 - x for x in exer_ratio]

    model_dir =  f"../model/{model_name}/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    best_model = model_dir+f"{current_time}_{ratio}_{index}_best_model"
    if model_arch=="def":
        cdm = NCD(knowledge_n, item_n, user_n, log_name, best_model, usr_emb, itm_emb)
    if model_arch=="gen":
        cdm = NCD_gen(knowledge_n, item_n, user_n, log_name, best_model, usr_emb, itm_emb, stu_ratio, exer_ratio)
    if model_arch=="con":
        cdm = NCD_con(knowledge_n, item_n, user_n, log_name, best_model, usr_emb, itm_emb, similar_students, similar_exercises)
    cdm.train(train_set, test_set, epoch=epoch, device=device, lr = learning_rate)

    cdm.load(best_model)
    auc, accuracy, rmse = cdm.eval(test_set, device=device, save=True)
    log("on test data: auc: %.6f, accuracy: %.6f, rmse: %.6f" % (auc, accuracy, rmse))
