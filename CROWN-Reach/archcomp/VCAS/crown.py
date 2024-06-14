#########################################################################
##   This file is part of the CROWN-Reach verifier                     ##
##                                                                     ##
##   Copyright (C) 2024 The CROWN-Reach Team                           ##
##   Primary contacts: Xiangru Zhong <xiangruzh0915@gmail.com>         ##
##                     Yuhao Jia <yuhaojia98@g.ucla.edu>               ##
##                     Huan Zhang <huan@huan-zhang.com>                ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

import os
import sys
CROWN_DIR = "../../Verifier_Development/complete_verifier"
sys.path.append(CROWN_DIR)

import time
import torch
import onnx
import onnx2pytorch
import numpy as np
from collections import defaultdict, OrderedDict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from random_points_simulation import random_attack
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
satisfying_constraints = {0: [-float('inf'), float('inf')],     # COC
                          1: [-float('inf'), 0           ],     # DNC
                          2: [0            , float('inf')],     # DND
                          3: [-float('inf'), -1500       ],     # DES1500
                          4: [1500         , float('inf')],     # CL1500
                          5: [-float('inf'), -1500       ],     # SDES1500
                          6: [1500         , float('inf')],     # SCL1500
                          7: [-float('inf'), -2500       ],     # SDES2500
                          8: [2500         , float('inf')]}     # SCL2500

g = 32.2
h0_dot_dot_choices = {0: [-g/8, 0, g/8],
                      1: [-g/3, -7*g/24, -g/4],
                      2: [g/3, 7*g/24, g/4],
                      3: [-g/3, -7*g/24, -g/4],
                      4: [g/3, 7*g/24, g/4],
                      5: [-g/3],
                      6: [g/3],
                      7: [-g/3],
                      8: [g/3]}

def load_models(model_dir):
    model_list = []
    model_name_list = os.listdir(model_dir)
    model_name_list.sort()
    for model_name in model_name_list:
        onnx_model = onnx.load(os.path.join(model_dir, model_name))
        model_ori = onnx2pytorch.ConvertModel(onnx_model)
        model_ori.training = False
        lirpa_model = BoundedModule(model_ori, torch.zeros(1, 3), device=device, bound_opts={'conv_mode': 'matrix'})
        model_list.append(lirpa_model)
    return model_list

def get_acceleration(adv, h0_dot_lb, h0_dot_ub, strategy="middle"):
    if adv == 0:
        h0_dot_dot = 0
    range_of_adv = satisfying_constraints[adv]
    if strategy == "middle":
        h0_dot_dot = h0_dot_dot_choices[adv][int((len(h0_dot_dot_choices[adv])-1)/2)]   # Choose the middle one
    elif strategy == "worst":
        h0_dot_dot = h0_dot_dot_choices[adv][-1]  # Choose the "worst" acceleration
    else:
        raise("Strategy not implemented")
    mask_no_change = (h0_dot_lb > range_of_adv[0]) * (h0_dot_ub < range_of_adv[1])
    mask_need_change = (h0_dot_ub < range_of_adv[0]) + (h0_dot_lb > range_of_adv[1])
    if mask_no_change:
        h0_dot_dot = 0  # If the current climb rate complies with the adv, no acceleration is needed
    elif mask_need_change:
        pass
    else:
        raise("Can't decide the acceleration due to too large range of h0_dot.")
    return h0_dot_dot

def dynamics(lb, ub, h0_dot_dot):
    # Use IBP to get the next state bounds
    weight = torch.tensor([[1.0, -1.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]).to(device)
    bias = torch.tensor([-0.5 * h0_dot_dot, h0_dot_dot, -1]).to(device)
    mid = (lb + ub) / 2
    diff = (ub - lb) / 2
    weight_abs = weight.abs()
    center = torch.addmm(bias, mid, weight.t())
    deviation = diff.matmul(weight_abs.t())
    lower = center - deviation
    upper = center + deviation
    return lower, upper

def check_constraints(lb, ub, unsafe_lb, unsafe_ub):
    if (ub < unsafe_lb) or (lb > unsafe_ub):
        return 0    # Safe
    elif (lb > unsafe_lb) and (ub < unsafe_ub):
        return 1    # Unsafe
    else:
        return 2    # Unknown    

def visualization(lb_records, ub_records, output_file):
    with open(output_file, 'w') as f:
        for i in range(len(lb_records)):
            f.write(f"plot([{i}, {i+1}, {i+1}, {i}, {i}], [{lb_records[i].item()}, {lb_records[i].item()}, {ub_records[i].item()}, {ub_records[i].item()}, {lb_records[i].item()}], 'color', '[0, 0.4, 0]');\n")
            f.write("hold on;\n")
    print(f"Figure saved as {output_file}.")

def visualization_sample(h_records, output_file):
    with open(output_file, 'w') as f:
        for i in range(h_records.shape[0]):
            f.write(f"plot([{i}, {i+1}], [{h_records[i].item()}, {h_records[i].item()}], 'color', '[0, 0.4, 0]');\n")
            f.write("hold on;\n")
    print(f"Figure saved as {output_file}.")
        
def verifier(models, input_lb, input_ub, h0_dot_initial, strategy):
    adv = 0     # COC
    lb_records = [input_lb[:,0,0,0].clone()]
    ub_records = [input_ub[:,0,0,0].clone()]
    for step in range(10):
        print(f"Step {step}")
        nn_input_lb = (input_lb - scale_mean) / scale_range
        nn_input_ub = (input_ub - scale_mean) / scale_range           
        ptb = PerturbationLpNorm(x_L=nn_input_lb, x_U=nn_input_ub)
        bounded_tensor = BoundedTensor(nn_input_lb, ptb)
        required_A = defaultdict(set)
        lirpa_model = models[adv]
        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
        lb, ub, A_dict = lirpa_model.compute_bounds(x=(bounded_tensor,), method='CROWN', return_A=True, needed_A_dict=required_A)
        max_lb = torch.max(lb)
        potential_adv = torch.where(ub[0] > max_lb)     # To check whether the optimal interval is not overlapped with others.
        if len(potential_adv) > 1:
            print("Not unique.")
            break
        adv = int(potential_adv[0][0].item())
        # print(f"adv: {adv}")
        h0_dot_dot = get_acceleration(adv, h0_dot_lb=input_lb[0,0,0,1], h0_dot_ub = input_ub[0,0,0,1], strategy=strategy)
        input_lb[:,0,0,:], input_ub[:,0,0,:] = dynamics(input_lb[:,0,0,:], input_ub[:,0,0,:], h0_dot_dot)
        print(f"[{input_lb[0,0,0,0].item()}, {input_ub[0,0,0,0].item()}]")
        lb_records.append(input_lb[:,0,0,0].clone())
        ub_records.append(input_ub[:,0,0,0].clone())

        b = check_constraints(input_lb[0,0,0,0].item(), input_ub[0,0,0,0].item(), -100, 100)
        if b == 1:
            print("Unsafe.")
            break
        elif b == 2:
            print("Unknown.")
    return b, lb_records, ub_records

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h0_dot', type=float, choices=[-19.5, -22.5, -25.5, -28.5], help='Set h0_dot_initial')
    parser.add_argument('--strategy', type=str, choices=['worst', 'middle'], help='The strategy for picking up next acceleration')
    args = parser.parse_args()
    
    benchmark_dir = "../../ARCH-COMP2024/benchmarks/VCAS"
    model_dir = os.path.join(benchmark_dir, "onnx_networks")
    models = load_models(model_dir)
    scale_mean = torch.tensor([0.0,0.0,20.0]).to(device)
    scale_range = torch.tensor([16000.0,5000.0,40.0]).to(device)
    
    # h0_dot_initial = [-19.5, -22.5, -25.5, -28.5]
    h0_dot_initial = args.h0_dot
    strategy = args.strategy
    print(f"h0_dot: {h0_dot_initial}, strategy: {strategy}")

    input_lb = torch.tensor([-133, h0_dot_initial, 25]).view(-1, 1, 1, 3).to(device)
    input_ub = torch.tensor([-129, h0_dot_initial, 25]).view(-1, 1, 1, 3).to(device)

    final_result = 0

    start_time = time.time()
    attack_res, cex = random_attack(models, input_lb, input_ub, h0_dot_initial, strategy)

    if attack_res:
        final_result = 1
    else:
        final_result, lb_records, ub_records = verifier(models, input_lb, input_ub, h0_dot_initial, strategy)

    if final_result == 0:
        print("VERIFIED")
    elif final_result == 1:
        print("FALSIFIED")
    else:
        print("UNKNOWN")

    print(f"time cost: {(time.time() - start_time):.3f}")
    
    # print(h_records)
    # if attack_res:
    #     visualization_sample(cex, f"VCAS_{h0_dot_initial}_{strategy}.m")
    # else:
    #     visualization(lb_records, ub_records, f"VCAS_{h0_dot_initial}_{strategy}.m")
