import socket
import subprocess
import os
import platform
import signal
import atexit
import webbrowser
import atexit
_server_process = None

def is_server_running(host="localhost", port=8000, timeout=1): ## Function to check if server is running
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True             ## Return true and establish connection if server is listening
    except OSError:
        return False

def kill_process_using_port(port):
    """Kill any process using the specified port."""
    try:
        if platform.system() == "Windows":  
            result = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()  ## Creating subprocess to run in background and check if server is running
            for line in result.strip().split("\n"):
                if "LISTENING" in line:
                    pid = int(line.strip().split()[-1])
                    subprocess.run(f"taskkill /PID {pid} /F", shell=True)
        else:
            result = subprocess.check_output(["lsof", "-ti", f":{port}"]).decode().strip()
            if result:
                for pid in result.split("\n"):
                    os.kill(int(pid), signal.SIGKILL)   ## Kill the server/ process if it is running using port number
    except subprocess.CalledProcessError:
        pass

def _cleanup_server():
    global _server_process
    if _server_process and _server_process.poll() is None:
        try:
            if platform.system() == "Windows":
                subprocess.run(f"taskkill /PID {_server_process.pid} /F", shell=True)
            else:
                os.killpg(os.getpgid(_server_process.pid), signal.SIGTERM)
        except Exception as e:
            print(f"Cleanup warning: {e}")
    kill_process_using_port(8000)
    _server_process = None

atexit.register(_cleanup_server)

def ConnectAndOpen():
    """Start server only if not running."""
    global _server_process
    if is_server_running():
        print("Server already running. Skipping startup.")
        return

    environmentPath = "pokemon-showdown"
    command = ["node", "pokemon-showdown", "start", "--no-security"]

    if platform.system() == "Windows":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        _server_process = subprocess.Popen(
            command, cwd=environmentPath, creationflags=creationflags
        )
    else:
        _server_process = subprocess.Popen(
            command,
            cwd=environmentPath,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    print(f"Server started with PID {_server_process.pid}")
if __name__ == "__main__":
    ConnectAndOpen()
