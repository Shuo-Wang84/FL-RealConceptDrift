import torch
import math
import random
# fedavg
# def server_aggregate(global_model, local_models,_):
#     for global_param, local_param_list in zip(global_model.parameters(),zip(*[local_model.parameters() for local_model in local_models])):
#         global_param.data = torch.mean(torch.stack(local_param_list), dim=0)


# FedAvg with dynamic aggregation weights
def server_aggregate(global_model, local_models, aggregation_weights):
    for global_param, local_param_list in zip(global_model.parameters(),
                                                      zip(*[local_model.parameters() for local_model in
                                                            local_models])):
        weighted_local_params = [param * weight for param,weight in zip(local_param_list, aggregation_weights)]

        nan_tensor = torch.tensor(math.nan, device='cuda:0')
        non_zero_tensors = [torch.where(tensor == 0, nan_tensor, tensor) for tensor in weighted_local_params]
        global_param.data = torch.nanmean(torch.stack(non_zero_tensors), dim=0)
        global_param.data[torch.isnan(global_param.data)] = 0

        # try:
        #     global_param.data[torch.isnan(global_param.data)] = 0
        # except:
        #     print(global_param.data)

        # global_param.data = torch.mean(torch.stack(weighted_local_params), dim=0)
        # print(global_param)

#随机选择聚合全局模型
def server_aggregate_random(global_model, local_models, aggregation_weights):
    num_local_models = len(local_models)
    num_models_to_aggregate = random.randint(1, num_local_models) #选择最少多少客户端参与
    selected_local_models = random.sample(local_models, num_models_to_aggregate)

    for global_param, selected_local_params in zip(global_model.parameters(),
                                                      zip(*[local_model.parameters() for local_model in
                                                            selected_local_models])):
        weighted_local_params = [param * weight for param,weight in zip(selected_local_params, aggregation_weights)]

        nan_tensor = torch.tensor(math.nan, device='cuda:0')
        non_zero_tensors = [torch.where(tensor == 0, nan_tensor, tensor) for tensor in weighted_local_params]
        global_param.data = torch.nanmean(torch.stack(non_zero_tensors), dim=0)
        global_param.data[torch.isnan(global_param.data)] = 0


#根据客户端表现选择聚合模型
def server_aggregate_top(global_model, local_models, aggregation_weights, clients_scores):
    indices = list(range(len(local_models)))
    sorted_clients, sorted_indices = zip(*sorted(zip(clients_scores, indices), reverse=True))
    num_best_models = int(0.7*len(local_models))
    best_local_models = [local_models[i] for i in sorted_indices[:num_best_models]]

    for global_param, best_local_params in zip(global_model.parameters(),
                                                      zip(*[local_model.parameters() for local_model in
                                                            best_local_models])):
        weighted_local_params = [param * weight for param,weight in zip(best_local_params, aggregation_weights)]

        nan_tensor = torch.tensor(math.nan, device='cuda:0')
        non_zero_tensors = [torch.where(tensor == 0, nan_tensor, tensor) for tensor in weighted_local_params]
        global_param.data = torch.nanmean(torch.stack(non_zero_tensors), dim=0)
        global_param.data[torch.isnan(global_param.data)] = 0