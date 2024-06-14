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

import torch
import onnx
import onnx2pytorch
import random
import numpy as np

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

# def load_models(model_dir):
#     model_list = []
#     model_name_list = os.listdir(model_dir)
#     model_name_list.sort()
#     for model_name in model_name_list:
#         onnx_model = onnx.load(os.path.join(model_dir, model_name))
#         model_ori = onnx2pytorch.ConvertModel(onnx_model)
#         model_ori.training = False
#         lirpa_model = BoundedModule(model_ori, torch.zeros(1, 3), device=device, bound_opts={'conv_mode': 'matrix'})
#         model_list.append(lirpa_model)
#     return model_list

def get_acceleration_random(adv, h0_dot, strategy="middle"):
    h0_dot_dot = torch.zeros_like(h0_dot).to(device)
    mask_COC = adv == 0
    h0_dot_dot[mask_COC] = 0
    for adv_idx in range(1, 9):
        range_of_adv = torch.tensor([satisfying_constraints[a.item()] for a in adv]).to(device)
        mask_adv = (adv == adv_idx)
        if not any(mask_adv):
            continue
        if strategy == "middle":
            h0_dot_dot[mask_adv] = torch.tensor([h0_dot_dot_choices[a.item()][int((len(h0_dot_dot_choices[a.item()])-1)/2)] for a in adv[mask_adv]]).to(device)  # Choose the middle one
        elif strategy == "worst":
            h0_dot_dot[mask_adv] = torch.tensor([h0_dot_dot_choices[a.item()][-1] for a in adv[mask_adv]]).to(device)  # Choose the "worst" acceleration
        else:
            raise("Strategy not implemented")
        mask_no_change = (adv == adv_idx) * (h0_dot > range_of_adv[:,0]) * (h0_dot < range_of_adv[:,1])
        h0_dot_dot[mask_no_change] = 0  # If the current climb rate complies with the adv, no acceleration is needed
    return h0_dot_dot

def dynamics_random(state, h0_dot_dot):
    state[:,0] = state[:,0] - state[:,1] - 0.5 * h0_dot_dot
    state[:,1] = state[:,1] + h0_dot_dot
    state[:,2] -= 1

# def visualization(h_records, output_file):
#     stacked_records = torch.stack(h_records)
#     array_records = stacked_records.cpu().numpy()
#     x = np.linspace(0, 10, 100)
#     y1 = -100 * np.ones_like(x)
#     y2 = 100 * np.ones_like(x)
#     plt.fill_between(x, y1, y2, color='red', alpha=0.3)
#     for one_record in array_records.T:
#         plt.plot(one_record)
#     plt.xlabel('t')
#     plt.ylabel('h')
#     plt.savefig(output_file)
#     plt.show()
        

def random_attack(models, input_lb, input_ub, h0_dot_initial, strategy):
    scale_mean = torch.tensor([0.0,0.0,20.0]).to(device)
    scale_range = torch.tensor([16000.0,5000.0,40.0]).to(device)
    batch_size = 10
    adv = torch.zeros(batch_size)     # COC
    input_rand = torch.rand(batch_size, *input_lb.shape[1:]).to(input_lb)
    input_rand = input_rand * (input_ub - input_lb) + input_lb
    h0_dot_rand = torch.ones(batch_size) * h0_dot_initial
    h_records = [input_rand[:,0,0,0].clone()]
    for step in range(10):
        print(f"Step {step}")
        adv_new = torch.zeros_like(adv).to(device)
        h0_dot_dot = torch.zeros_like(h0_dot_rand).to(device)
        for adv_idx in range(9):            
            lirpa_model = models[adv_idx]
            mask_idx = adv == adv_idx
            if not mask_idx.any():
                continue    # Igonre if there isn't any input with this adv
            input = input_rand[mask_idx]
            output = lirpa_model((input-scale_mean)/scale_range)
            adv_new[mask_idx] = torch.argmax(output, dim=1).to(torch.float32)
            h0_dot_dot[mask_idx] = get_acceleration_random(adv_new[mask_idx], h0_dot=input[:,0,0,1], strategy=strategy)
        adv = adv_new
        print(adv)
        dynamics_random(input_rand[:,0,0,:], h0_dot_dot)
        h_records.append(input_rand[:,0,0,0].clone())
        cex_list = torch.nonzero((input_rand[:,0,0,0] > -100) * (input_rand[:,0,0,0] < 100))
        if cex_list.numel() != 0:
            cex_idx = cex_list[0, 0]
            stacked_records = torch.stack(h_records)
            return True, stacked_records[:, cex_idx]
    return False, None

# if __name__ == '__main__':
#     benchmark_dir = "../../ARCH-COMP2024/benchmarks/VCAS"
#     model_dir = os.path.join(benchmark_dir, "onnx_networks")
#     models = load_models(model_dir)
#     scale_mean = torch.tensor([0.0,0.0,20.0]).to(device)
#     scale_range = torch.tensor([16000.0,5000.0,40.0]).to(device)
    
#     batch_size = 10
#     # h0_dot_initial = [-19.5, -22.5, -25.5, -28.5]
#     h0_dot_initial = -28.5
#     strategy = "middle"

#     adv = torch.zeros(batch_size)     # COC
#     # input_lb = torch.tensor([-133, h0_dot_initial[0], 25]).view(-1, 1, 1, 3).to(device)
#     # input_ub = torch.tensor([-129, h0_dot_initial[0], 25]).view(-1, 1, 1, 3).to(device)
#     h0_rand = torch.rand(batch_size) * 4 - 133
#     # h0_dot_rand = torch.tensor([random.choice(h0_dot_initial) for _ in range(batch_size)])
#     h0_dot_rand = torch.ones(batch_size) * h0_dot_initial
#     tau_rand = torch.ones(batch_size) * 25
#     input_rand = torch.stack((h0_rand, h0_dot_rand, tau_rand), dim=1).view(-1, 1, 1, 3).to(device)

#     h_records = [input_rand[:,0,0,0].clone()]
#     for step in range(10):
#         print(f"Step {step}")
#         adv_new = torch.zeros_like(adv).to(device)
#         h0_dot_dot = torch.zeros_like(h0_dot_rand).to(device)
#         for adv_idx in range(9):            
#             # ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
#             # bounded_tensor = BoundedTensor(input_lb, ptb)
#             # required_A = defaultdict(set)
#             lirpa_model = models[adv_idx]
#             mask_idx = adv == adv_idx
#             if not mask_idx.any():
#                 continue    # Igonre if there isn't any input with this adv
#             input = input_rand[mask_idx]
#             # required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
#             # lb, ub, A_dict = lirpa_model.compute_bounds(x=(bounded_tensor,), method='CROWN', return_A=True, needed_A_dict=required_A)
#             output = lirpa_model((input-scale_mean)/scale_range)
#             adv_new[mask_idx] = torch.argmax(output, dim=1).to(torch.float32)
#             h0_dot_dot[mask_idx] = get_acceleration(adv_new[mask_idx], h0_dot=input[:,0,0,1], strategy=strategy)
#         adv = adv_new
#         print(adv)
#         dynamics(input_rand[:,0,0,:], h0_dot_dot)
#         h_records.append(input_rand[:,0,0,0].clone())
    
#     # print(h_records)
#     visualization(h_records, f"VCAS_{h0_dot_initial}_{strategy}.png")
