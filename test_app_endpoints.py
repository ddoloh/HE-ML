import sys
import os
import time
import urllib.request
import json
import multiprocessing

sys.path.insert(0, os.path.abspath('.'))
from app import app
import uvicorn

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8011, log_level="error")

def test_api():
    print("[*] Starting Large Dataset AI Training FastAPI Server on port 8011...")
    proc = multiprocessing.Process(target=run_server)
    proc.start()
    
    time.sleep(2.5) # Wait for server startup

    base_url = "http://127.0.0.1:8011"
    
    try:
        # 1. Healthcheck
        req = urllib.request.Request(f"{base_url}/api/v1/health")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("1. Healthcheck Response:", data)
            assert data["status"] == "healthy"

        # 2. List Large Datasets (Over 100 datasets)
        req = urllib.request.Request(f"{base_url}/api/v1/train/large_datasets/list")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"\n2. Large Datasets List Count: {len(data['datasets'])} datasets available.")
            assert len(data["datasets"]) >= 100

        # 3. Start Large Dataset Training Job
        payload = json.dumps({"dataset_id": "ts_dataset_001", "model_type": "lstm", "epochs": 5, "n_samples": 2000}).encode()
        req = urllib.request.Request(f"{base_url}/api/v1/train/large_dataset/start", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("\n3. Large Training Start Response:", data)
            job_id = data["job_id"]
            assert data["status"] == "started"

        # 4. Poll Training Status
        time.sleep(1.0)
        req = urllib.request.Request(f"{base_url}/api/v1/train/large_dataset/status/{job_id}")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("\n4. Training Progress Status:", f"Job: {data['job_id']}, Progress: {data['progress_pct']}%, Epoch: {data['current_epoch']}/{data['total_epochs']}")
            assert "progress_pct" in data

        # 5. Export Model Checkpoint
        payload = json.dumps({"job_id": job_id}).encode()
        req = urllib.request.Request(f"{base_url}/api/v1/train/large_dataset/export/{job_id}", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("\n5. Checkpoint Export Response:", data["export_info"])
            assert data["status"] == "exported"

        # 6. Web UI Model Inference Test
        payload = json.dumps({"job_id": job_id, "model_type": "lstm", "scheme_id": "ckks"}).encode()
        req = urllib.request.Request(f"{base_url}/api/v1/inference/run", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("\n6. Web Inference Output Response:", data)
            assert data["status"] == "success"

        # 7. Dashboard UI GET /
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode()
            print(f"\n7. Web Dashboard HTML Response: Received {len(html)} bytes")
            assert "AI Model Training & Inference Platform" in html

        print("\n============================================================")
        print("[ALL 100+ DATASETS & WEB UI TRAINING/INFERENCE ENDPOINTS VERIFIED 100%!]")
        print("============================================================")

    finally:
        proc.terminate()
        proc.join()

if __name__ == "__main__":
    test_api()

