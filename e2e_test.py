import os
import sys
import time
import requests
import subprocess
import tempfile
import pandas as pd

def main():
    print("Starting Coordinator Server...")
    db_path = "backend/coordinator/database.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    import shutil
    local_data_dir = "local_data"
    if os.path.exists(local_data_dir):
        shutil.rmtree(local_data_dir)
    server_process = subprocess.Popen(
        [sys.executable, "backend/coordinator/server.py"],
        env=dict(os.environ, PORT="5000", PYTHONIOENCODING="utf-8")
    )
    
    # Wait for server to start
    time.sleep(3)
    
    base_url = "http://localhost:5000"
    try:
        resp = requests.get(base_url)
        print("Server is up!")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        server_process.terminate()
        sys.exit(1)
        
    print("Creating dummy dataset...")
    test_data = {
        'review': [f'This is a great review {i}' for i in range(100)],
        'label': [i % 2 for i in range(100)]
    }
    df = pd.DataFrame(test_data)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)
        
    print("Uploading dataset...")
    with open(csv_path, 'rb') as f:
        files = {'file': f}
        resp = requests.post(f"{base_url}/upload_dataset", files=files)
        print(f"Upload response: {resp.json()}")

    print("Starting Client A...")
    client_process = subprocess.Popen(
        [sys.executable, "backend/client/client.py", "--client_id", "client_A", "--coordinator_url", base_url],
        env=dict(os.environ, PYTHONIOENCODING="utf-8")
    )
    
    # Wait for client to register (poll until it appears)
    print("Waiting for client to register...")
    for _ in range(30):
        try:
            r = requests.get(f"{base_url}/get_clients")
            clients = r.json().get("clients", [])
            if any(c.get("id") == "client_A" for c in clients):
                break
        except:
            pass
        time.sleep(1)
    
    print("Starting training...")
    resp = requests.post(f"{base_url}/start_training", json={"client_count": 1, "rounds": 2, "epochs": 2})
    print(f"Start training response: {resp.json()}")
    
    print("Waiting for client to complete training (timeout 120s)...")
    try:
        client_process.wait(timeout=120)
        print("Client finished.")
    except subprocess.TimeoutExpired:
        print("Client timed out!")
        client_process.terminate()
        
    print("Checking results...")
    resp = requests.get(f"{base_url}/results")
    print(f"Results: {resp.json()}")
    
    print("Cleaning up...")
    server_process.terminate()
    os.unlink(csv_path)

if __name__ == "__main__":
    main()
