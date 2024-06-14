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

import subprocess
import os
import sys
import csv

VERIFIER_ROOT = "/home/CROWN-Reach/"
sys.path.append(VERIFIER_ROOT)

def check_ready_signal(server_process):
    while True:
        output = server_process.stdout.readline()
        # print(output)
        if output == '' and server_process.poll() is not None:
            raise RuntimeError("Server process exited unexpectedly")
        if "Server started." in output:
            return True
        
def parse_verifier_output(output):
    time_str = ""
    result_str = "VERIFIED" 
    output_lower = output.lower()
    
    for line in output_lower.splitlines():
        if "time cost" in line:
            time_str = line.split(":")[1].strip().split()[0]
        if any(keyword in line for keyword in ["unsafe", "unreachable", "falsified"]):
            result_str = "FALSIFIED"
        elif any(keyword in line for keyword in ["flow* terminated", "killed", "unknown"]):
            result_str = "UNKNOWN"
    return result_str, time_str

def run_verification(work_dir, python_script_path, cpp_executable_path, instance_info):
    print(f"Begin verifying benchmark {instance_info['benchmark']}, instance {instance_info['instance']}.")
    #start server
    env = os.environ.copy()
    pythonpath = os.pathsep.join(sys.path)
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUNBUFFERED"] = "1"
    server_process = subprocess.Popen(
        ["python3", python_script_path], 
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    # make sure sever running
    check_ready_signal(server_process)

    #start cpp simuation
    cpp_process = subprocess.Popen([cpp_executable_path], cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # while True:
    #     output = cpp_process.stdout.readline()
    #     if output == '' and cpp_process.poll() is not None:
    #         break
    #     if output:
    #         print(output.strip())
    cpp_output, _ = cpp_process.communicate()
    result, exec_time = parse_verifier_output(cpp_output)
    print(f"Result: {result}")
    print(f"Time: {exec_time}")

    server_process.terminate()
    server_process.wait()

    #write result in csv
    with open('/home/CROWN-Reach/results/results.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([instance_info['benchmark'], instance_info['instance'], result, exec_time])

def run_verification_VCAS(work_dir, python_script_path, instance_info):
    print(f"Begin verifying benchmark VCAS, instance {instance_info['strategy']+str(instance_info['h0_dot'])}.")
    env = os.environ.copy()
    env = os.environ.copy()
    pythonpath = os.pathsep.join(sys.path)
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUNBUFFERED"] = "1"
    verifier_process = subprocess.Popen(
        ["python3", python_script_path, "--h0_dot", str(instance_info['h0_dot']), "--strategy", instance_info['strategy']], 
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    verifier_output, _ = verifier_process.communicate()
    result, exec_time = parse_verifier_output(verifier_output)
    print(f"Result: {result}")
    print(f"Time: {exec_time}")

    with open('/home/CROWN-Reach/results/results.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['VCAS', instance_info['strategy']+str(instance_info['h0_dot']), result, exec_time])

if __name__ == "__main__":
    acc_dir = "/home/CROWN-Reach/archcomp/ACC/"
    acc_info = {'benchmark': 'ACC', 'instance':'safe-distance'}
    run_verification(acc_dir, acc_dir+'crown.py', acc_dir+'acc', acc_info)

    airplane_dir = "/home/CROWN-Reach/archcomp/Airplane/"
    airplane_info = {'benchmark': 'Airplane', 'instance':'continues'}
    run_verification(airplane_dir, airplane_dir+'crown.py', airplane_dir+'airplane', airplane_info)

    AttitudeControl_dir = "/home/CROWN-Reach/archcomp/AttitudeControl/"
    AttitudeControl_info = {'benchmark': 'Attitude Control', 'instance':'avoid'}
    run_verification(AttitudeControl_dir, AttitudeControl_dir+'crown.py', AttitudeControl_dir+'attitude_control', AttitudeControl_info)
    
    SinglePendulum_dir = "/home/CROWN-Reach/archcomp/SinglePendulum/"
    SinglePendulum_info = {'benchmark': 'Single Pendulum', 'instance':'reach'}
    run_verification(SinglePendulum_dir, SinglePendulum_dir+'crown.py', SinglePendulum_dir+'single_pendulum', SinglePendulum_info)

    tora_dir = "/home/CROWN-Reach/archcomp/TORA/"
    tora_info = {'benchmark': 'TORA', 'instance':'remain'}
    run_verification(tora_dir, tora_dir+'crown.py', tora_dir+'Tora', tora_info)

    tora_sigmoid_dir = "/home/CROWN-Reach/archcomp/TORA/"
    tora_sigmoid_info = {'benchmark': 'TORA', 'instance':'reach-sigmoid'}
    run_verification(tora_sigmoid_dir, tora_sigmoid_dir+'crown_sigmoid.py', tora_sigmoid_dir+'tora_sigmoid', tora_sigmoid_info)

    tora_relu_tanh_dir = "/home/CROWN-Reach/archcomp/TORA/"
    tora_relu_tanh_info = {'benchmark': 'TORA', 'instance':'reach-tanh'}
    run_verification(tora_relu_tanh_dir, tora_relu_tanh_dir+'crown_relu_tanh.py', tora_relu_tanh_dir+'tora_relu_tanh', tora_relu_tanh_info)

    Unicycle_dir = "/home/CROWN-Reach/archcomp/Unicycle/"
    Unicycle_info = {'benchmark': 'Unicycle', 'instance':'reach'}
    run_verification(Unicycle_dir, Unicycle_dir+'crown.py', Unicycle_dir+'Unicycle', Unicycle_info)

    balancing_dir = "/home/CROWN-Reach/archcomp/CartPole/"
    balancing_info = {'benchmark': 'CartPole', 'instance':'reach'}
    run_verification(balancing_dir, balancing_dir+'crown.py', balancing_dir+'balancing', balancing_info)

    VCAS_dir = "/home/CROWN-Reach/archcomp/VCAS/"
    run_verification_VCAS(VCAS_dir, VCAS_dir+'crown.py', {'h0_dot': -19.5, 'strategy': 'worst'})
    run_verification_VCAS(VCAS_dir, VCAS_dir+'crown.py', {'h0_dot': -22.5, 'strategy': 'worst'})
    run_verification_VCAS(VCAS_dir, VCAS_dir+'crown.py', {'h0_dot': -25.5, 'strategy': 'worst'})
    run_verification_VCAS(VCAS_dir, VCAS_dir+'crown.py', {'h0_dot': -28.5, 'strategy': 'worst'})
    run_verification_VCAS(VCAS_dir, VCAS_dir+'crown.py', {'h0_dot': -19.5, 'strategy': 'middle'})
    run_verification_VCAS(VCAS_dir, VCAS_dir+'crown.py', {'h0_dot': -22.5, 'strategy': 'middle'})
    run_verification_VCAS(VCAS_dir, VCAS_dir+'crown.py', {'h0_dot': -25.5, 'strategy': 'middle'})
    run_verification_VCAS(VCAS_dir, VCAS_dir+'crown.py', {'h0_dot': -28.5, 'strategy': 'middle'})

    dp_more_robust_dir = "/home/CROWN-Reach/archcomp/DoublePendulum/"
    dp_more_robust_info = {'benchmark': 'Double Pendulum', 'instance':'more-robust'}
    run_verification(dp_more_robust_dir, dp_more_robust_dir+'crown_more_robust.py', dp_more_robust_dir+'double_pendulum_more_robust', dp_more_robust_info)

    dp_less_robust_dir = "/home/CROWN-Reach/archcomp/DoublePendulum/"
    dp_less_robust_info = {'benchmark': 'Double Pendulum', 'instance':'less-robust'}
    run_verification(dp_less_robust_dir, dp_less_robust_dir+'crown_less_robust.py', dp_less_robust_dir+'double_pendulum_less_robust', dp_less_robust_info)

    NAV_robust_dir = "/home/CROWN-Reach/archcomp/NAV/"
    NAV_robust_info = {'benchmark': 'NAV', 'instance':'robust'}
    run_verification(NAV_robust_dir, NAV_robust_dir+'crown_robust.py', NAV_robust_dir+'NAV_robust', NAV_robust_info)

    NAV_standard_dir = "/home/CROWN-Reach/archcomp/NAV/"
    NAV_standard_info = {'benchmark': 'NAV', 'instance':'standard'}
    run_verification(NAV_standard_dir, NAV_standard_dir+'crown_standard.py', NAV_standard_dir+'NAV_standard', NAV_standard_info)

    quad_dir = "/home/CROWN-Reach/archcomp/Quadrotor/"
    quad_info = {'benchmark': 'Quadrotor', 'instance':'reach'}
    run_verification(quad_dir, quad_dir+'crown.py', quad_dir+'quad', quad_info)
    