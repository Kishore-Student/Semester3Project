import socket
import subprocess
import os
import webbrowser
import time
import platform
import signal

def kill_process_using_port(port):
    """Find and kill the process using the specified port."""
    system = platform.system()

    try:
        if system == "Windows":
            # Windows: use netstat + taskkill
            result = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
            for line in result.strip().split("\n"):
                if "LISTENING" in line:
                    pid = int(line.strip().split()[-1])
                    print(f"Killing PID {pid} using port {port}...")
                    subprocess.run(f"taskkill /PID {pid} /F", shell=True)
        else:
            # Linux/macOS: use lsof + kill
            result = subprocess.check_output(["lsof", "-ti", f":{port}"]).decode().strip()
            if result:
                for pid in result.split("\n"):
                    print(f"Killing PID {pid} using port {port}...")
                    os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        # No process was using the port
        pass

def StartServer(host_no, port_no, timeout=60): 
    wait_until = time.time() + timeout 
    attempt = 0
    os.system("cls" if os.name == "nt" else "clear")

    while time.time() < wait_until:
        try:
            with socket.create_connection((host_no, port_no), timeout=2):
                return True
        except OSError:
            if attempt % 10 == 0:
                print("Waiting for server to start...")
            time.sleep(0.1)
            attempt += 1
    return False

def ConnectAndOpen():
    environmentPath = "pokemon-showdown"
    command = ["node", "pokemon-showdown", "start", "--no-security"]

    # Kill any process currently using port 8000
    kill_process_using_port(8000)

    # Start the server
    Serverstart = subprocess.Popen(command, cwd=environmentPath)

    if StartServer("localhost", 8000):
        print(f"Server started with PID {Serverstart.pid} on localhost:8000")
        time.sleep(5) ## Give time for server to start properly
        webbrowser.open_new_tab("http://localhost:8000")
        print("Opening browser...")
        time.sleep(5) ## Let the browser be opened completely
    else:
        print("Server timed out. Terminating process...")
        Serverstart.terminate()
        Serverstart.wait()
        exit(1)

if __name__ == "__main__":
    ConnectAndOpen()
