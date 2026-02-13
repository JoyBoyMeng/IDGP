import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
from pathlib import Path

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics, get_popularity_prediction_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from utils.visualize import visualize_fluxons

def evaluate_model_popularity_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler,
                                         evaluate_idx_data_loader: DataLoader,
                                         evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20,
                                         time_gap: int = 2000, infer: str = 'val', epoch: int = -1, exp: str = ''):
    """
    evaluate models on the popularity prediction task
    :param epoch: epoch num
    :param infer: 'val' or 'test'
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()
    valid_batch_num = 0

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120, disable=True)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            # if batch_idx == 10:
            #     break
            # if model_name in ['Fluxion']:
            #     if batch_idx % 500 == 0:
            #         states = model[0].fluxon_bank.get_all_fluxon().detach().clone()
            #         Path("./visualization").mkdir(parents=True, exist_ok=True)
            #         base_v = f'./visualization/{exp}/fluxon_basis_D128_seed42.pt'
            #         image_path = f'./visualization/{exp}/{infer}_epoch{epoch}_batch{batch_idx}.jpg'
            #         visualize_fluxons(
            #             states,
            #             save_path=image_path,
            #             seed=42,
            #             basis_file=base_v,
            #             normalize=False,
            #             color_by="row",
            #             title=f'{batch_idx}'
            #         )
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[
                    evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]
            if model_name in ['Fluxion']:
                batch_slow_feature = evaluate_data.slow_features[evaluate_data_indices]
                batch_sp_feature = evaluate_data.sp_features[evaluate_data_indices]
                batch_time_feature = evaluate_data.timeinterval_features[evaluate_data_indices]

            # print(f'eval batch: {batch_idx}')
            valid_index = np.where(batch_labels > 0)
            if sum(batch_labels) == 0:
                # print('no loss this batch')
                evaluate_idx_data_loader_tqdm.set_description(
                    f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: no loss')
            else:
                valid_batch_num = valid_batch_num + 1

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                _, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN', 'TGN-id']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                _, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                _, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                _, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            elif model_name in ['Fluxion']:
                batch_dst_node_embeddings = model[0].compute_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    dst_slow_feature=batch_slow_feature,
                    valid_index=valid_index,
                    dst_sp_feature=batch_sp_feature,
                    dst_time_feature=batch_time_feature,
                    visual='True'
                )
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            if sum(batch_labels) != 0:
                if model_name in ['Fluxion']:
                    predicts = model[1](x=batch_dst_node_embeddings).squeeze(dim=-1)
                else:
                    predicts = model[1](x=batch_dst_node_embeddings[valid_index]).squeeze(dim=-1)
                labels = torch.from_numpy(batch_labels[valid_index]).float().to(predicts.device)
                labels = torch.log2(labels)
                # print(f'labels:{labels}')
                # print(f'predts:{predicts}')

                loss = loss_func(input=predicts, target=labels)

                evaluate_total_loss += loss.item()

                evaluate_y_trues.append(labels)
                evaluate_y_predicts.append(predicts)

                evaluate_idx_data_loader_tqdm.set_description(
                    f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= valid_batch_num
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        # print(f'y_trues: {evaluate_y_trues}')
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)
        # print(f'y_predicts: {evaluate_y_predicts}')

        evaluate_metrics = get_popularity_prediction_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics
