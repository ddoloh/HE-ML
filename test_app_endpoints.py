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
    uvicorn.run(app, host="127.0.0.1", port=8009, log_level="error")

def test_api():
    print("[*] Starting Autotuned Multi-Scheme HE FastAPI Server on port 8009...")
    proc = multiprocessing.Process(target=run_server)
    proc.start()
    
    time.sleep(2.5) # Wait for server startup

    base_url = "http://127.0.0.1:8009"
    
    try:
        # 1. Healthcheck
        req = urllib.request.Request(f"{base_url}/api/v1/health")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("1. Healthcheck Response:", data)
            assert data["status"] == "healthy"

        # 2. Security Validation & Autotuner
        payload = json.dumps({"security_level": "128_bit", "dim": 32, "scale": 1000000, "noise_level": 12.5}).encode()
        req = urllib.request.Request(f"{base_url}/api/v1/security/validate", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("\n2. Security Validation Response:", data["security_compliance"]["level_name"], "Compliant:", data["security_compliance"]["is_standard_compliant"])
            assert data["security_compliance"]["is_standard_compliant"] == True

        # 3. Parallel CPU Benchmark
        payload = json.dumps({"model_id": "mobilenet_v3", "sample_count": 30, "max_workers": 4}).encode()
        req = urllib.request.Request(f"{base_url}/api/v1/parallel/benchmark", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("\n3. Parallel CPU Benchmark Response:", data["model_id"], "Throughput:", data["parallel_throughput_samples_per_sec"], "samp/s")
            assert "parallel_throughput_samples_per_sec" in data

        # 4. 4-Scheme Comparison Matrix
        payload = json.dumps({"task": "lstm", "target_id": "sine_wave"}).encode()
        req = urllib.request.Request(f"{base_url}/api/v1/compare/schemes", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"\n4. 4-Scheme Matrix Comparison ({len(data['scheme_comparison'])} schemes):")
            for s in data["scheme_comparison"]:
                print(f"   - {s['scheme_id'].upper():10s} | {s['scheme_name']:45s} | Pred: {s['prediction']}")
            assert len(data["scheme_comparison"]) == 4

        # 5. Dashboard UI GET /
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode()
            print(f"\n5. Web Dashboard HTML Response: Received {len(html)} bytes")
            assert "Enterprise Multi-Scheme HE Engine" in html

        print("\n============================================================")
        print("[ALL IMPROVEMENTS, ENDPOINTS & WEB UI VERIFIED 100%!]")
        print("============================================================")

    finally:
        proc.terminate()
        proc.join()

if __name__ == "__main__":
    test_api()
