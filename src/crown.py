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
import yaml
CROWN_DIR = "../../auto_LiRPA"
sys.path.append(CROWN_DIR)

import torch
import onnx
import onnx2pytorch
from collections import defaultdict, OrderedDict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import json

# For RPC
import gevent
import gevent.pywsgi
import gevent.queue
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.wsgi import WsgiServerTransport
from tinyrpc.server.gevent import RPCServerGreenlets
from tinyrpc.dispatch import RPCDispatcher

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)


dispatcher = RPCDispatcher()
transport = WsgiServerTransport(max_content_length=1000000, queue_class=gevent.queue.Queue)
# start wsgi server as a background-greenlet
wsgi_server = gevent.pywsgi.WSGIServer(('127.0.0.1', 5000), transport.handle)
gevent.spawn(wsgi_server.serve_forever)
rpc_server = RPCServerGreenlets(transport, JSONRPCProtocol(), dispatcher)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
torch.set_default_dtype(torch.float64)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model_path = config['model_dir']
input_shape = tuple(config['input_shape'])
output_T_shape = tuple(config['output_T_shape'])
output_c_shape = tuple(config['output_c_shape'])
output_scale = config['output_scale']
output_offset = config['output_offset']

onnx_model = onnx.load(model_path)
model_ori = onnx2pytorch.ConvertModel(onnx_model)
model_ori.to(torch.get_default_dtype())
model_ori.training = False
lirpa_model = BoundedModule(model_ori, torch.zeros(1, *input_shape[1:]), device=device, bound_opts={'activation_bound_option': 'same-slope'})

@dispatcher.public
def CROWN_reach(input_lb, input_ub):
    input_lb = torch.tensor(input_lb).view(*input_shape).to(dtype=torch.get_default_dtype(), device=device)
    input_ub = torch.tensor(input_ub).view(*input_shape).to(dtype=torch.get_default_dtype(), device=device)
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    bounded_tensor = BoundedTensor(input_lb, ptb)
    required_A = defaultdict(set)
    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
    lb, ub, A_dict = lirpa_model.compute_bounds(x=(bounded_tensor,), method='CROWN', return_A=True, needed_A_dict=required_A)
    A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]
    coefficients = OrderedDict()
    coefficients["T"] = (A['lA'] * output_scale).view(output_T_shape).cpu().tolist()
    coefficients["u_min"] = ((A['lbias'] - output_offset) * output_scale).view(output_c_shape).cpu().tolist()
    coefficients["u_max"] = ((A['ubias'] - output_offset) * output_scale).view(output_c_shape).cpu().tolist()
    return coefficients

print("Server started.")
rpc_server.serve_forever()