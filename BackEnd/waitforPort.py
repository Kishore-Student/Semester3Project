import socket
import time

def wait_for_server(port=8000, host="127.0.0.1", interval=0.2, timeout=30):
    start = time.time()
    while True:
        sock = socket.socket()
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            sock.close()
            return
        except Exception:
            sock.close()
            if time.time() - start > timeout:
                raise TimeoutError(f"Server on port {port} did not become ready in time.")
            time.sleep(interval)
