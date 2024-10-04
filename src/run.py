import subprocess
import os
import sys
import signal
from attack import run_attack_simulation

def check_ready_signal(server_process):
    while True:
        output = server_process.stdout.readline()
        if output == '' and server_process.poll() is not None:
            returncode = server_process.returncode
            error_output = server_process.stderr.read() if server_process.stderr else 'No stderr available'
            raise RuntimeError(f"Server process exited unexpectedly with return code {returncode}. Error output:\n{error_output}")
        if "Server started." in output:
            print("Server Started")
            return True
        
def run_crownreach(config_path, test_mode=False):
    cmd = ['./CrownSettings', config_path]
    if test_mode:
        cmd.append('--test') 
    with subprocess.Popen(cmd, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:
        try:
            # Real-time output display
            while True:
                output = proc.stdout.readline()
                if output == '':
                    if proc.poll() is not None:
                        break
                if output:
                    print(output.strip())  # Print each line of stdout
            # Once the loop exits, check if there were any errors
            stderr_output = proc.stderr.read()
            if stderr_output:
                print("Errors:\n", stderr_output.strip())
        except Exception as e:
            print("An error occurred:", e)
            proc.kill()
            proc.wait()

def kill_process_on_port(port):
    try:
        # Find the PID of the process using the port
        result = subprocess.check_output(["lsof", "-t", f"-i:{port}"])
        pids = result.decode().strip().split('\n')
        for pid in pids:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process {pid} on port {port}")
    except subprocess.CalledProcessError:
        pass

# ------------------- config_path -------------------- #
config_path = sys.argv[1]
port = 5000  # specify your server port

# ------------------- check if test mode -------------------- #
test_mode = '--test' in sys.argv

# ------------------- run attack -------------------- #
attack_result = run_attack_simulation(config_path)
if attack_result:
    print("Falsified.")
else:
    print("Attack results unknown, start verifying.")

    # ------------------- check and kill process on port -------------------- #
    kill_process_on_port(port)

    # ------------------- start server -------------------- #
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    server_process = subprocess.Popen(
        ["python3", './crown.py', config_path], 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    # make sure server is running
    check_ready_signal(server_process)

    # ------------------- run verification -------------------- #
    run_crownreach(config_path, test_mode=test_mode)

    # ------------------- close server -------------------- #
    try:
        # Try to terminate the server gracefully
        server_process.terminate()
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill the server if it doesn't terminate gracefully
        os.kill(server_process.pid, signal.SIGKILL)
        server_process.wait()

    print("Server Closed")
