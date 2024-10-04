import onnxruntime as ort
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import yaml
from torchdiffeq import odeint
ort.set_default_logger_severity(4)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def compile_dynamics(dynamics_expressions):
    dynamics = [expr.replace('^', '**') for expr in dynamics_expressions]
    return [compile(expr, "<string>", "eval") for expr in dynamics]

def get_dtype_from_onnx_model(session):
    output_info = session.get_outputs()[0]
    if output_info.type == 'tensor(float)':
        return torch.float32
    elif output_info.type == 'tensor(double)':
        return torch.float64
    return torch.float32

def run_attack_simulation(config_path):
    config = load_config(config_path)
    if not config['run_attack']:
        return ''

    ort_session = ort.InferenceSession(config['model_dir'], providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    dtype = get_dtype_from_onnx_model(ort_session)
    
    compiled_expressions = compile_dynamics(config['dynamicsExpressions'])
    
    def odefunc(t, y):
        locals_dict = {config['initialSet'][i]['name']: y[i].item() for i in range(len(y))}
        input_vars = y[:config['num_nn_input']].detach().numpy().astype(np.float64 if dtype == torch.float64 else np.float32).reshape(1, -1)
        control_outputs = ort_session.run(None, {input_name: input_vars})[0].squeeze()
        adjusted_control_outputs = control_outputs * config['output_scale'] + config['output_offset']

        for i in range(config['num_nn_output']):
            locals_dict[f"u{i+1}"] = adjusted_control_outputs[i] if adjusted_control_outputs.ndim > 0 else adjusted_control_outputs.item()

        safe_builtins = {'abs': abs, 'min': min, 'max': max, 'sin': math.sin, 'cos': math.cos}
        dydt = [eval(expr, {'__builtins__': safe_builtins}, locals_dict) for expr in compiled_expressions]
        return torch.tensor(dydt, dtype=dtype)
    
    initial_set = config['initialSet']
    initial_conditions = [np.random.uniform(low=var['interval'][0], high=var['interval'][1]) for var in initial_set]

    total_time = config['steps'] * config['step_size']
    times = torch.linspace(0, total_time, steps=int(config['steps']))

    trajectory = odeint(odefunc, torch.tensor(initial_conditions, dtype=dtype), times)

    def evaluate_constraints(trajectory, constraints):
        results = []
        for constraint in constraints:
            check_expr = compile(constraint.replace('^', '**'), "<string>", "eval")
            for point in trajectory:
                locals_dict = {config['initialSet'][i]['name']: point[i].item() for i in range(len(point))}
                if eval(check_expr, {'__builtins__': {}}, locals_dict)>0:
                    # print(constraint, locals_dict, eval(check_expr, {'__builtins__': {}}, locals_dict))
                    results.append(False)
                    break
            else:
                results.append(True)
        return all(results)

    falsified = False

    if "constraints_target" in config:
        if not evaluate_constraints(trajectory[-1:], config["constraints_target"]):
            falsified = True

    if "constraints_unsafe" in config:
        if evaluate_constraints(trajectory, config["constraints_unsafe"]):
            falsified = True

    if "constraints_safe" in config:
        if not evaluate_constraints(trajectory, config["constraints_safe"]):
            falsified = True
    
    return falsified

try:
    run_attack_simulation('configs/attitude_control.yaml')
except Exception as e:
    print("results unknown, please use verifier.")
