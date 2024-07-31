import copy
import math
import time
import arff
import torch
import csv
import numpy as np
import torch.nn as nn
import skmultiflow
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import itertools
from utility import DDMdetection
from utility import EDDMdetection
from utility import ADWINdetection
from function.Dataset import MyDataset
from function.model import MLP, weights_init
from function.SGDtrain import client_update
from function.test import test1, test
from function.fedavg import server_aggregate, server_aggregate_random, server_aggregate_top
import pandas as pd
import random
from torch.utils.data import TensorDataset, DataLoader
import wandb

# initial wandb
wandb.init(project='fed_mcd', name='localused')

time_start = time.time()  # 记录开始时间
# initial parameter
drift_type = 'normal_data'
NOISE_PROB = drift_type

csv_name = f"{drift_type}"
gpu = 0
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
print(device)
optimizer = 'adagrad'
criterion = nn.CrossEntropyLoss()

num_clients = 10
batch_size = 100
lr = 0.005
num_epochs = 5

testnum = 'kddcup data'
train_num = 10
adjust_scale = 2
converge_acc = []
client_model_num = 2

#MLP   # 54,1000,8      8,1000,2         41,1000,23
input_size = 2
hidden_size = 128
output_size = 2



# # real dataset load code
# train_datasets = []
# train_dataset = MyDataset(f'data\Real-world datasets\covType\covtypeNorm.arff')   #data\Real-world datasets\electricity\modified_elecNorm.arff
# split_features, split_labels = train_dataset.split_dataset(num_clients)
#
# for i in range(num_clients):
#     train_dataset = MyDataset(f'data\Real-world datasets\covType\covtypeNorm.arff') #data\Real-world datasets\covType\covtypeNorm.arff
#     train_dataset.features = split_features[i]
#     train_dataset.labels = split_labels[i]
#     train_datasets.append(train_dataset)

# artificial datalaod
train_datasets = []
for i in range(num_clients):
    # if i == num_clients-1:
    #     csv_name = f"data/rotation/abrupt_data_H_9.csv"
    #     train_dataset = MyDataset(csv_name)
    #     train_datasets.append(train_dataset)
    # # elif i == num_clients-2:
    # #     csv_name = f"data/rotation/abrupt_data_H_8.csv"
    # #     train_dataset = MyDataset(csv_name)
    # #     train_datasets.append(train_dataset)
    # # elif i == num_clients-3:
    # #     csv_name = f"data/rotation/abrupt_data_H_7.csv"
    # #     train_dataset = MyDataset(csv_name)
    # #     train_datasets.append(train_dataset)
    # # elif i == num_clients-4:
    # #     csv_name = f"data/rotation/abrupt_data_H_6.csv"
    # #     train_dataset = MyDataset(csv_name)
    # #     train_datasets.append(train_dataset)
    # # elif i == num_clients-5:
    # #     csv_name = f"data/rotation/abrupt_data_H_5.csv"
    # #     train_dataset = MyDataset(csv_name)
    # #     train_datasets.append(train_dataset)
    # else:
        csv_name = f"{drift_type}"
        csv_name = f"data/rotation/{csv_name}_{i}.csv"
        train_dataset = MyDataset(csv_name)
        train_datasets.append(train_dataset)

min_length = 200000
shortest_dataset = None
for dataset in train_datasets:
    if len(dataset) < min_length:
        min_length = len(dataset)
        shortest_dataset = dataset

print(f"The shortest dataset has length {min_length}.")
globle_train_dataset = shortest_dataset

data_streams = []
for i in range(num_clients):
    data_stream = DataLoader(train_datasets[i], batch_size=batch_size, shuffle=False)
    data_streams.append((data_stream))



client_test_f1ss = []
client_test_losses = []
client_test_accs = []
global_test_accs = []
global_test_losses = []
global_test_f1s = []
pre_global_datas = [[] for _ in range(num_epochs)]
pre_client_datas = [[[] for _ in range(num_clients)] for _ in range(num_epochs)]
pre_client_data = [[[[] for _ in range(2)] for _ in range(num_clients)] for _ in range(num_epochs)]

for j in range(num_epochs):
    model = MLP(input_size, hidden_size, output_size)
    model = model.to(device)
    model.apply(weights_init)
    aggregation_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    print(aggregation_weights)
    print(j)
    local_models_used = [[] for _ in range(num_clients)]
    global_data_stream = iter(DataLoader(globle_train_dataset, batch_size=batch_size))
    global_model = copy.deepcopy(model)
    global_model_random = copy.deepcopy(model)
    global_model_top =  copy.deepcopy(model)
    best_acc_clients = [0 for _ in range(num_clients)]
    data_streams = []
    client_test_f1s = [[] for _ in range(num_clients)]
    client_test_loss = [[] for _ in range(num_clients)]
    client_test_acc = [[] for _ in range(num_clients)]
    global_test_f1 = []
    global_test_acc = []
    global_test_loss = []
    global_data_batch = []
    local_modellist = [[] for _ in range(num_clients)]
    if_detection = [[False] for _ in range(num_clients)]


    ddm_instances = {}
    for client_id in range(num_clients):
        ddm = DDMdetection.DDM(client_id=client_id, min_num_instances=100, warning_level=2.0, out_control_level=3.0)
        ddm_instances[client_id] = ddm


    local_modellist = [[MLP(input_size, hidden_size, output_size).to(device) for _ in range(client_model_num)] for _ in range(num_clients)]
    # print(len(local_modellist))

    for i in range(num_clients):
        data_stream = DataLoader(train_datasets[i], batch_size=batch_size, shuffle=False)
        data_streams.append((data_stream))

    change_detection_count = 0
    for i in range(len(global_data_stream)):
        # print('\ntime stamp:',i)
        local_models = []
        local_train_losses = []
        local_train_accs = []
        global_data = []
        for client_idx in range(num_clients):
            # print('\n\tclient {}'.format(client_idx))
            data_streams[client_idx] = iter(data_streams[client_idx])
            client_data = next(data_streams[client_idx])
            if len(client_data[1]) >= 10:
                batch_size_10_percent = int(len(client_data[1]) * 0.1)
                selected_data = []
                # print(len(client_data[1]))
                selected_data = random.sample(list(zip(*client_data)), batch_size_10_percent)
                selected_data = list(zip(*selected_data))
                tensor_list = [tensor.unsqueeze(0) for tensor in selected_data[0]]
                combined_tensor = torch.cat(tensor_list, dim=0)
                scalar_tensor = torch.stack(selected_data[1], dim=0)
                # combine data,label
                selected_data_result = [combined_tensor, scalar_tensor]
                selected_data_result = [t.to(device) for t in selected_data_result]
                selected_data = [torch.stack(d).to(device) for d in selected_data]
                if global_data == []:
                    global_data = selected_data_result
                else:
                    global_data = [torch.cat([global_data[0], selected_data_result[0]], dim=0),
                                   torch.cat([global_data[1], selected_data_result[1]], dim=0)]

                if global_data_batch == []:
                    global_data_batch = selected_data_result
                else:
                    global_data_batch = [torch.cat([global_data_batch[0], selected_data_result[0]], dim=0),
                                         torch.cat([global_data_batch[1], selected_data_result[1]], dim=0)]

            client_data = [d.to(device) for d in client_data]


            worst_loss = float('-inf')
            worst_acc = 100.0
            worst_f1 = 100.0
            worst_client_test_data = None
            worst_client_index = 0


            best_loss = float('inf')
            best_acc = 0.0
            best_f1 = 0.0
            best_client_test_data = None
            best_client_index = 0

            best_model_indices = []
            worst_model_indices = []
            for n in range(client_model_num):
                test_loss, test_acc, test_f1, client_test_data = test1(local_modellist[client_idx][n], device, client_data)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_acc = test_acc
                    best_f1 = test_f1
                    best_client_test_data = client_test_data
                    best_client_index = n
                    best_acc_clients[client_idx] = best_acc
                    best_model_indices.append(n)
                else:
                    worst_model_indices.append(n)

                if test_loss > worst_loss:
                    worst_loss = test_loss
                    worst_acc = test_acc
                    worst_f1 = test_f1
                    worst_client_test_data = client_test_data
                    worst_client_index = n
                    worst_model_indices.append(n)
            wandb.log({
                f'client{client_idx}_test_acc_{j}': best_acc,
                f'client{client_idx}_test_loss_{j}': best_loss,
                f'client{client_idx}_test_f1_{j}': best_f1,
                "time stamp":i,
            })
            worst_model_indices = worst_model_indices[::-1] + best_model_indices
            last_worst_model_indices = []
            [last_worst_model_indices.append(x) for x in worst_model_indices if x not in last_worst_model_indices]
            local_modellist[client_idx][last_worst_model_indices[0]].load_state_dict(global_model_random.state_dict())
            # select worst model load global model

            # #ddm detection
            cpu_data = []
            for tuple_item in best_client_test_data:
                cpu_tuple = tuple(tensor.cpu() for tensor in tuple_item)
                cpu_data.append(cpu_tuple)
            pre_client_datas[j][client_idx].extend(cpu_data)

            ddm_dect = 0
            for detecti in range(len(cpu_data[0][0].numpy())):
                    _ = ddm_instances[client_idx].add_element(cpu_data[0][1][detecti])
                    if ddm_instances[client_idx].detected_change():
                        ddm_dect = ddm_dect + 1
                        print(f"ddm Change detected at clinet{client_idx} ,timestamp{i}, index {detecti}")
                        if_detection[client_idx] = True
                        wandb.log({
                                      f"DDM detected": f"ddm Change detected at clinet{client_idx} ,timestamp{i}, index {detecti}",
                                      "time stamp": i, })
                        # if ddm_dect%5 == 0:
                        for n in range(client_model_num):
                            local_modellist[client_idx][n].apply(weights_init)


                     ts_init)


            client_test_acc[client_idx].append(best_acc)
            client_test_loss[client_idx].append(best_loss)
            client_test_f1s[client_idx].append(best_f1)
            # client train
            for n in range(client_model_num):
                client_optimizer = torch.optim.Adagrad(local_modellist[client_idx][n].parameters(), lr=0.005)


            for e in range(train_num):
                for n in range(client_model_num):
                    client_update(local_modellist[client_idx][n], client_optimizer, client_data, local_train_losses, local_train_accs)

            worst_loss = float('-inf')
            worst_acc = 100.0
            worst_f1 = 100.0
            worst_client_test_data = None
            worst_client_index = 0


            best_loss = float('inf')
            best_acc = 0.0
            best_f1 = 0.0
            best_client_test_data = None
            best_client_index = 0

            best_model_indices = []
            worst_model_indices = []
            for n in range(client_model_num):
                test_loss, test_acc, test_f1, client_test_data = test1(local_modellist[client_idx][n], device, client_data)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_acc = test_acc
                    best_f1 = test_f1
                    best_client_test_data = client_test_data
                    best_client_index = n
                    best_acc_clients[client_idx] = best_acc
                    best_model_indices.append(n)
                else:
                    worst_model_indices.append(n)

                if test_loss > worst_loss:
                    worst_loss = test_loss
                    worst_acc = test_acc
                    worst_f1 = test_f1
                    worst_client_test_data = client_test_data
                    worst_client_index = n
                    worst_model_indices.append(n)

            worst_model_indices = worst_model_indices[::-1] + best_model_indices
            last_worst_model_indices = []
            [last_worst_model_indices.append(x) for x in worst_model_indices if x not in last_worst_model_indices]



            local_models.append(local_modellist[client_idx][best_client_index])
            local_models_used[client_idx].append(best_client_index)


        # global data
        global_loss, global_acc, global_f1, pre_global_datas[j] = test1(global_model, device, global_data_batch)
        wandb.log({
            f'global_loss_{j}': global_loss,
            f'global_acc_{j}': global_acc,
            f'global_f1_{j}': global_f1,
            "time stamp": i,
        })

        server_aggregate_random(global_model_random, local_models, aggregation_weights)


        global_test_acc.append(global_acc)
        global_test_loss.append(global_loss)
        global_test_f1.append(global_f1)
    print(local_models_used)


time_end = time.time()
time_sum = time_end - time_start
print(time_sum)

wandb.log({"runtime": time_sum,
           "NOISE_PROB":NOISE_PROB,
           "optimizer": optimizer,
           'local_models_used':local_models_used,}
          )
print('finished')

