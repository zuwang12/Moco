from glob import glob
import numpy as np
import argparse
from tqdm import tqdm
# from experiments.evaluate_tsp import is_valid
from moco.utils_ddrl import TSP_2opt
import pandas as pd
import os

def is_valid(tour):
    if tour[0]!=tour[-1]:
        return False
    if set(tour)!=set(np.arange(len(tour)-1)):
        return False
    return True

parser = argparse.ArgumentParser()
args = parser.parse_args()

exp_list = [x.split('/')[-1].split('.')[0].split('_tours')[0] for x in glob('./results/tmp_npy/*') if 'tour' in x]
for exp_name in exp_list:
    args.constraint_type = exp_name.split('_')[2]
    args.result_file_name = f'./results/{args.constraint_type}/{exp_name.split("_stacked")[0]}.csv'
    if os.path.exists(args.result_file_name):
        print(f'{args.result_file_name} already exist, so skip')
        continue
    print(f'start saving csv.. about - {args.result_file_name}')
    problem_size = int(exp_name.split('_')[1].replace('tsp', ''))

    npy_list = [x for x in glob('./results/tmp_npy/*') if exp_name in x]
    stacked_sample_idx = np.load([x for x in npy_list if 'stacked_sample_idx.npy' in x][0])
    stacked_points = np.load([x for x in npy_list if 'stacked_points.npy' in x][0])
    stacked_tours = np.load([x for x in npy_list if 'stacked_tours.npy' in x][0])
    stacked_gt = np.load([x for x in npy_list if 'stacked_gt.npy' in x][0])
    stacked_constraint = np.load([x for x in npy_list if 'stacked_constraint.npy' in x][0])
    stacked_constraint_matrix = np.load([x for x in npy_list if 'stacked_constraint_matrix.npy' in x][0])
    
    sample_idx, gt_tours, solved_tours, last_tours, gt_costs, gt_counts, basic_costs, solved_costs, last_costs, mean_costs, is_valid_list, penalty_counts, last_counts = [], [], [], [], [], [], [], [], [], [], [], [], []
    print('start making csv ... ')
    for idx, points, tour, gt_tour, constraint in tqdm(zip(stacked_sample_idx, stacked_points, stacked_tours, stacked_gt, stacked_constraint)):
        penalty_const = 1.0
        solver = TSP_2opt(points, args.constraint_type, constraint)
        gt_cost = solver.evaluate(gt_tour - 1)
        gt_count = solver.count_constraints(gt_tour-1)

        total_tour = [x for x in np.concatenate([tour.reshape(-1, problem_size), tour.reshape(-1, problem_size)[:,:1]], axis=-1) if is_valid(x)]
        if len(total_tour)==0:
            continue
        total_cost = [solver.evaluate(x) for x in total_tour]
        total_count = [solver.count_constraints(x) for x in total_tour]
        total_solved_cost = [cost + penalty_const*count for cost, count in zip(total_cost, total_count)]
        solved_cost = min(total_solved_cost)
        mean_cost = sum(total_solved_cost)/len(total_solved_cost)
        total_idx = total_solved_cost.index(solved_cost)
        total_solved_tour = total_tour[total_idx].tolist()
        total_penalty_count = total_count[total_idx]
        
        last_tour = [x for x in np.concatenate([tour[:,-1,:,:].reshape(-1, problem_size), tour[:,-1,:,:].reshape(-1, problem_size)[:,:1]], axis=-1) if is_valid(x)]
        last_cost = [solver.evaluate(x) for x in last_tour]
        last_count = [solver.count_constraints(x) for x in last_tour]
        last_solved_cost = [cost + penalty_const*count for cost, count in zip(last_cost, last_count)]
        if len(last_solved_cost)!=0:
            last_cost = min(last_solved_cost)
            last_idx = last_solved_cost.index(last_cost)
            last_solved_tour = last_tour[last_idx].tolist()
            last_penalty_count = last_count[last_idx]
        else:
            last_cost = -1000
            last_solved_tour = []
            last_penalty_count = -1000
        
        
        sample_idx.append(idx)
        gt_tours.append(gt_tour.tolist())
        gt_costs.append(gt_cost)
        gt_counts.append(gt_count)
        solved_tours.append(total_solved_tour)
        solved_costs.append(solved_cost)
        mean_costs.append(mean_cost)
        penalty_counts.append(total_penalty_count)
        last_tours.append(last_solved_tour)
        last_costs.append(last_cost)
        last_counts.append(last_penalty_count)
        is_valid_list.append(is_valid(total_solved_tour))

        if idx%100==0:
            df = pd.DataFrame([sample_idx, gt_tours, gt_costs, gt_counts, solved_tours, solved_costs, mean_costs, penalty_counts, last_tours, last_costs, last_counts, is_valid_list]).T
            df.columns = ['sample_idx', 'gt_tour', 'gt_cost', 'gt_count', 'solved_tour', 'solved_cost', 'mean_cost', 'penalty_count', 'last_tours', 'last_cost', 'last_count', 'is_valid_list']
                
            df.to_csv(args.result_file_name, index=0)
            
    df = pd.DataFrame([sample_idx, gt_tours, gt_costs, gt_counts, solved_tours, solved_costs, mean_costs, penalty_counts, last_tours, last_costs, last_counts, is_valid_list]).T
    df.columns = ['sample_idx', 'gt_tour', 'gt_cost', 'gt_count', 'solved_tour', 'solved_cost', 'mean_cost', 'penalty_count', 'last_tours', 'last_cost', 'last_count', 'is_valid_list']
        
    df.to_csv(args.result_file_name, index=0)
    print('file_name : ', args.result_file_name)
    print('complete inference')