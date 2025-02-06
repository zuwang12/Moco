import os
import argparse
import sys
import random
from functools import partial
import time
import timeit
from datetime import timedelta
import pandas as pd
import tensorflow as tf
import numpy as np

import jax
import jax.numpy as jnp
from jax import disable_jit
from chex import dataclass, PRNGKey, Array 
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow

from moco.tasks import TspTaskFamily, train_task, TspTaskParams
from moco.lopt import HeatmapOptimizer
from moco.utils import jax_has_gpu
from moco.data_utils import load_data
from moco.utils_ddrl import TSP_2opt, calculate_distance_matrix2
from learned_optimization.optimizers.optax_opts import Adam
from learned_optimization.learned_optimizers import base as lopt_base, mlp_lopt

def is_valid(tour):
    if tour[0]!=tour[-1]:
        return False
    if set(tour)!=set(np.arange(len(tour)-1)):
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", default="./data/tsp/test-100-coords.npy", help="path to the data", type=str)
    parser.add_argument("--task_batch_size", "-tb", help="batch size", type=int, default=32)
    parser.add_argument("--batch_size_eval", "-be", help="batch size", type=int, default=32)
    parser.add_argument("--num_steps", "-n", help="number of training steps", type=int, default=200)
    parser.add_argument("--learning_rate", "-l", help="learning rate", type=float, default=1e-2)
    parser.add_argument("--seed", "-s", help="seed", type=int, default=random.randrange(sys.maxsize))
    parser.add_argument("--verbose", "-v", help="print training progress", action="store_true")
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--heatmap_init_strategy", type=str, choices=["heuristic", "constant"], default="heuristic") # gets overwritten by the model
    parser.add_argument("--rollout_actor", type=str, choices=["softmax", "entmax"], default="softmax")
    parser.add_argument("--k", "-k", help="number of nearest neighbors", type=int, default=10)
    parser.add_argument("--causal", "-c", help="use causal accumulation of rewards for policy gradient calc", action="store_true")
    parser.add_argument("--baseline", "-b", help="specify baseline for policy gradient calc", type=str, default=None, choices=[None, "avg"])
    parser.add_argument("--mlflow_uri", help="mlflow uri", type=str, default="logs")
    parser.add_argument("--experiment", help="experiment name", type=str, default="tsp")
    parser.add_argument("--num_starting_nodes", "-ns", help="number of starting nodes", type=int, default=2)
    parser.add_argument("--checkpoint_folder", "-cf", help="folder to load checkpoint from", type=str, default="./checkpoints/tsp20_200_32")
    parser.add_argument("--two_opt_t_max", type=int, default=None)
    parser.add_argument("--first_accept", action="store_true")
    parser.add_argument("--run_name", type=str, default='test')
    parser.add_argument("--num_cities", type=int, default=20)
    parser.add_argument("--constraint_type", type=str, default='cluster')
    parser.add_argument("--run_time", type=str, default='test')
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--testYn", type=str, default='Y')
    parser.add_argument("--save_csv", default=False)
    
    args = parser.parse_args()
    
    # Pretty print arguments
    print("Arguments:")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    # test gpu
    print("jax has gpu:", jax_has_gpu())
    
    # Set seed for reproducibility
    seed = 2025
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    tf.random.set_seed(seed)  # TensorFlow
    key = jax.random.PRNGKey(args.seed)

    date_per_type = {
        'basic': '',
        'box': '240710',
        'path': '240711',
        'cluster': '240721',
    }

    now = time.strftime('%y%m%d_%H%M%S')
    if args.run_name is None:
        args.run_name = f'test_{now}'

    if args.constraint_type == 'basic':
        if args.testYn == 'Y':
            args.data_path = f'./data/tsp{args.num_cities}_test_concorde_test.txt'
        else:
            args.data_path = f'./data/tsp{args.num_cities}_test_concorde.txt'
        args.result_file_name = f'./results/{args.constraint_type}/moco_tsp{args.num_cities}_{args.constraint_type}_{args.run_time}.csv'
    else:
        if args.testYn == 'Y':
            args.data_path = f'./data/tsp{args.num_cities}_{args.constraint_type}_constraint_{date_per_type.get(args.constraint_type)}_test.txt'
        else:
            args.data_path = f'./data/tsp{args.num_cities}_{args.constraint_type}_constraint_{date_per_type.get(args.constraint_type)}.txt'
        args.result_file_name = f'./results/{args.constraint_type}/moco_tsp{args.num_cities}_{args.constraint_type}_constraint_{args.run_time}.csv'
    print(f'Result file : {args.result_file_name}')

    dataset, tour, constraint, constraint_matrix, sample_idx = load_data(args.data_path, batch_size=args.batch_size_eval, constraint_type=args.constraint_type)
    _, problem_size, _ = dataset.element_spec.shape
    dataset_size = sum([i.shape[0] for i in dataset.as_numpy_iterator()])
    print("Dataset size: ", dataset_size, "Problem size: ", problem_size)

    # load optimizer
    args.checkpoint_folder = f"./checkpoints/tsp{args.num_cities}_200_32"
    if args.checkpoint_folder is not None:
        import orbax.checkpoint as ocp
        restore_options = ocp.CheckpointManagerOptions(
            best_mode='min',
            best_fn=lambda x: x['val_last_best_reward'],
            )
        restore_mngr = ocp.CheckpointManager(
            args.checkpoint_folder,
            ocp.PyTreeCheckpointer(),
            options=restore_options
            )
        
        # overwrite command line arguments with checkpoint metadata
        metadata = restore_mngr.metadata()
        if 'top_k' in metadata and metadata['top_k'] != args.top_k:
            metadata['top_k'] = args.top_k
        args.heatmap_init_strategy = metadata['heatmap_init_strategy']
        args.rollout_actor = metadata['rollout_actor']
        args.k = metadata['k']
        args.causal = metadata['causal']
        args.baseline = metadata['baseline']
        lopts = {
            "adam": lopt_base.LearnableAdam(),
            "gnn": HeatmapOptimizer(
                embedding_size=metadata["embedding_size"], 
                num_layers_init=metadata["num_layers_init"], 
                num_layers_update=metadata["num_layers_update"], 
                aggregation=metadata["aggregation"], 
                update_strategy=metadata["update_strategy"], 
                normalization=metadata["normalization"]
                )
            }
        l_optimizer = lopts[metadata['lopt']]
        optimizer_params = restore_mngr.restore(restore_mngr.best_step())
        optimizer = l_optimizer.opt_fn(optimizer_params, is_training=False)
        print(f"Running {metadata['lopt']} optimizer from checkpoint {args.checkpoint_folder} step {restore_mngr.best_step()}")

    else:
        optimizer = Adam(learning_rate=args.learning_rate)
        print(f"Running Adam with lr {args.learning_rate}")

    task_family = TspTaskFamily(
        problem_size=problem_size,
        batch_size=args.task_batch_size,
        k=args.k, 
        baseline=args.baseline, 
        causal=args.causal, 
        top_k=args.top_k, 
        heatmap_init_strategy=args.heatmap_init_strategy, 
        rollout_actor=args.rollout_actor, 
        two_opt_t_max=args.two_opt_t_max, 
        first_accept=args.first_accept,
        constraint_type = args.constraint_type,
    )

    print("Task family: ", task_family)

    @jax.jit
    def train_task_from_multiple_starts(coordinates, key: PRNGKey, constraint_matrix=None):
        """Train a task from multiple starting nodes"""
        # create task params
        key, subkey = jax.random.split(key)
        starting_nodes = jax.random.choice(subkey, problem_size, shape=(args.num_starting_nodes,), replace=False) # sample n non repeating starting nodes
        coordinates = jnp.tile(coordinates[None,:,:], (args.num_starting_nodes,1,1)) # repeat coordinates n times
        if args.constraint_type != 'basic':
            constraint_matrix = jnp.tile(constraint_matrix[None, :], (args.num_starting_nodes, 1))

        task_params = TspTaskParams(coordinates=coordinates, starting_node=starting_nodes, constraint_matrix=constraint_matrix)
        keys = jax.random.split(key, args.num_starting_nodes)
        results = jax.vmap(train_task, in_axes=(0, 0, None, None, None))(task_params, keys, args.num_steps, optimizer, task_family)
        return results
    
    stacked_sample_idx = []
    stacked_points = []
    stacked_gt = []
    stacked_constraint = []
    stacked_constraint_matrix = []
    stacked_tours = []
    
    if args.constraint_type == 'basic':
        for i, (batch_sample_idx, batch_points, batch_gt) in tqdm(enumerate(zip(sample_idx.as_numpy_iterator(), dataset.as_numpy_iterator(), tour.as_numpy_iterator()))):
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch_points.shape[0])
            res = jax.vmap(train_task_from_multiple_starts)(jnp.array(batch_points, dtype=jnp.float32), keys)
            
            stacked_sample_idx.append(batch_sample_idx)
            stacked_points.append(batch_points)
            stacked_gt.append(batch_gt)
            stacked_constraint.append(None)
            stacked_constraint_matrix.append(None)
            stacked_tours.append(res.tours)
    
    else:
        for i, (batch_sample_idx, batch_points, batch_gt, batch_constraint, batch_constraint_matrix) in tqdm(enumerate(zip(sample_idx.as_numpy_iterator(), dataset.as_numpy_iterator(), tour.as_numpy_iterator(), constraint.as_numpy_iterator(), constraint_matrix.as_numpy_iterator()))):
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch_points.shape[0])
            res = jax.vmap(train_task_from_multiple_starts)(jnp.array(batch_points, dtype=jnp.float32), keys, jnp.array(batch_constraint_matrix, dtype=jnp.float32))

            stacked_sample_idx.append(batch_sample_idx)
            stacked_points.append(batch_points)
            stacked_gt.append(batch_gt)
            stacked_constraint.append(batch_constraint)
            stacked_constraint_matrix.append(batch_constraint_matrix)
            stacked_tours.append(res.tours)
            
    sample_idx, gt_tours, solved_tours, last_tours, gt_costs, gt_counts, basic_costs, solved_costs, last_costs, mean_costs, is_valid_list, penalty_counts, last_counts = [], [], [], [], [], [], [], [], [], [], [], [], []

    stacked_sample_idx = np.concatenate(stacked_sample_idx, axis=0)
    stacked_points = np.concatenate(stacked_points, axis=0)
    stacked_gt = np.concatenate(stacked_gt, axis=0)
    stacked_constraint = np.concatenate(stacked_constraint, axis=0)
    stacked_tours = np.concatenate(stacked_tours, axis=0)
    np.save(f'./results/tmp_npy/{args.result_file_name.split("/")[-1].split(".")[0]}_stacked_sample_idx', stacked_sample_idx)
    np.save(f'./results/tmp_npy/{args.result_file_name.split("/")[-1].split(".")[0]}_stacked_points', stacked_points)
    np.save(f'./results/tmp_npy/{args.result_file_name.split("/")[-1].split(".")[0]}_stacked_gt', stacked_gt)
    np.save(f'./results/tmp_npy/{args.result_file_name.split("/")[-1].split(".")[0]}_stacked_constraint', stacked_constraint)
    np.save(f'./results/tmp_npy/{args.result_file_name.split("/")[-1].split(".")[0]}_stacked_constraint_matrix', stacked_constraint_matrix)
    np.save(f'./results/tmp_npy/{args.result_file_name.split("/")[-1].split(".")[0]}_stacked_tours', stacked_tours)
    
    if args.save_csv:
        print('start making csv ... ')
        for idx, points, tour, gt_tour, constraint, constraint_matrix in tqdm(zip(stacked_sample_idx, stacked_points, stacked_tours, stacked_gt, stacked_constraint, stacked_constraint_matrix)):
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
            solved_costs.append(solved_cost) # already involve penalty term
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
    else:
        print('skip making csv')
    print('complete inference')