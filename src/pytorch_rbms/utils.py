# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:00:29 2023

@author: COND04848
"""
import torch
from itertools import permutations


def batched_array(array, batch_size):
    for first in torch.arange(0, array.size(dim=0), batch_size):
        yield array[first : first + batch_size]


def batched_outer(a, b):
    return torch.einsum("ni,nj->nij", a, b)


def function_10dimCAR_control(v):
    a = (
        (torch.sum(v[0:5], -1) == 0)
        & (torch.sum(v[5:10], -1) == 0)
        & (torch.sum(v[10:], -1) == 1)
    )
    b = (
        (torch.sum(v[0:5], -1) == 1)
        & (torch.sum(v[5:10], -1) <= 1)
        & (torch.sum(v[10:], -1) == 1)
    )
    if a or b:
        return True
    else:
        return False


def all_combination():

    all_CAR1 = permutations([1, 0, 0, 0, 0])
    all_CAR0 = [0, 0, 0, 0, 0]
    all_CAR0_10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    ala = []
    ala0 = []

    for i in list(all_CAR1):
        ala.append(i)
        ala0.append(i)
    ala0.append(all_CAR0)

    all_CAR = torch.tensor(ala)
    all_CAR0 = torch.tensor(ala0)
    ALL_CAR = torch.unique(all_CAR, dim=0)
    ALL_CAR0 = torch.unique(all_CAR0, dim=0)

    ALL_list_car = []
    ALL_list_car.append(torch.tensor(all_CAR0_10))
    for i in range(ALL_CAR.shape[0]):
        for j in range(ALL_CAR0.shape[0]):
            ALL_list_car.append(torch.hstack((ALL_CAR[i], ALL_CAR0[j])))

    ALL_CAR1 = torch.zeros(len(ALL_list_car), 10)
    for i in range(len(ALL_list_car)):
        ALL_CAR1[i] = ALL_list_car[i]

    all_cell1 = permutations([1.0, 0, 0, 0, 0, 0, 0, 0])

    ala1 = []
    for i in list(all_cell1):
        ala1.append(i)
    all_cell = torch.tensor(ala1)
    ALL_cell = torch.unique(all_cell, dim=0)

    ALL_list = []
    for i in range(ALL_CAR1.shape[0]):
        for j in range(ALL_cell.shape[0]):
            ALL_list.append(torch.hstack((ALL_CAR1[i], ALL_cell[j])))

    ALL = torch.zeros(len(ALL_list), 18)
    for i in range(len(ALL_list)):
        ALL[i] = ALL_list[i]

    return ALL, ALL_CAR1, ALL_cell


def conditional_probabilities(data_CAR, data_cell, ALL_CAR1, ALL_cell, sample):
    # data_CAR are the CAR in 10dim, data_cell are the cell_type that I have separately
    # sample are a sample from the RBM

    CAR, counts_CAR = torch.unique(data_CAR, dim=0, return_counts=True)
    CELL, counts_cell = torch.unique(data_cell, dim=0, return_counts=True)

    prob_true = torch.zeros((ALL_cell.shape[0], ALL_CAR1.shape[0]))
    prob_pred = torch.zeros((ALL_cell.shape[0], ALL_CAR1.shape[0]))

    sample_get, counts_sample = torch.unique(sample, dim=0, return_counts=True)
    freq_sample = counts_sample / torch.sum(counts_sample)

    data_binary = torch.hstack((data_CAR, data_cell))
    unique_data, counts_data = torch.unique(data_binary, dim=0, return_counts=True)

    d_sample = {tuple(u.tolist()): c for u, c in zip(sample_get, freq_sample)}
    d_unique = {tuple(u.tolist()): c for u, c in zip(unique_data, counts_data)}

    for k in d_unique:
        if k in d_sample:
            for j in range(ALL_CAR1.shape[0]):
                for i in range(ALL_cell.shape[0]):
                    if torch.equal(
                        torch.tensor(k), torch.hstack((ALL_CAR1[j], ALL_cell[i]))
                    ):
                        for t in range(CAR.shape[0]):
                            if torch.equal(CAR[t], ALL_CAR1[j]):
                                prob_true[i, j] = d_unique[k] / counts_CAR[t]
                        prob_pred[i, j] = d_sample[k]
        if k not in d_sample:
            for j in range(ALL_CAR1.shape[0]):
                for i in range(ALL_cell.shape[0]):
                    if torch.equal(
                        torch.tensor(k), torch.hstack((ALL_CAR1[j], ALL_cell[i]))
                    ):
                        for t in range(CAR.shape[0]):
                            if torch.equal(CAR[t], ALL_CAR1[j]):
                                prob_true[i, j] = d_unique[k] / counts_CAR[t]

    for k in d_sample:
        if k not in d_unique:
            for j in range(ALL_CAR1.shape[0]):
                for i in range(ALL_cell.shape[0]):
                    if torch.equal(
                        torch.tensor(k), torch.hstack((ALL_CAR1[j], ALL_cell[i]))
                    ):
                        prob_pred[i, j] = d_sample[k]

    prob_pred1 = torch.zeros((ALL_cell.shape[0], ALL_CAR1.shape[0]))
    for j in range(prob_pred.shape[1]):
        for i in range(prob_pred.shape[0]):
            if torch.sum(prob_pred, 0)[j] != 0:
                prob_pred1[i, j] = prob_pred[i, j] / torch.sum(prob_pred, 0)[j]

    return prob_true, prob_pred1
