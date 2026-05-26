import requests
import torch
import os

BASE_URL = "http://localhost:5000"

print("1. Registering 'mock_client'...")
res = requests.post(f"{BASE_URL}/register", json={"client_id": "mock_client"})
print(f"Response: {res.json()}\n")

print("2. Checking /status...")
res = requests.get(f"{BASE_URL}/status")
print(f"Response: {res.json()}\n")

print("3. Generating dummy trained weights...")
# Create a dummy PyTorch state_dict so fedavg.py has something valid to average
dummy_model = {"embedding.weight": torch.tensor([[0.5, -0.2], [0.1, 0.9]])}
torch.save(dummy_model, "mock_weights.pt")

print("4. Submitting weights to /submit (This triggers FedAvg & Credits!)...")
with open("mock_weights.pt", "rb") as f:
    # Multipart form-data payload
    files = {'weights': ('mock_weights.pt', f, 'application/octet-stream')}
    data = {
        'client_id': 'mock_client',
        'samples_trained': 5000,
        'time_seconds': 15.2
    }
    res = requests.post(f"{BASE_URL}/submit", files=files, data=data)
    
print(f"Response: {res.json()}\n")

print("5. Checking /status for next round...")
res = requests.get(f"{BASE_URL}/status")
print(f"Response: {res.json()}\n")

# Clean up local dummy file
os.remove("mock_weights.pt")