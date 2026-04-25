import time
import os
import json
import re
from datetime import datetime
from ingest import ingest, VECTOR_STORE_DIR
from rag_chain import build_chain, ask
import config

def get_dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB

def run_test_suite(repo_url: str):
    print(f"🚀 Starting Comprehensive Test Suite for: {repo_url}")
    print("-" * 60)
    
    results = {"metadata": {"repo": repo_url, "timestamp": datetime.now().isoformat()}, "tests": []}
    
    # 1. Performance: Ingestion
    print("⏳ Running Ingestion Benchmark...")
    start_time = time.time()
    store, stats, cached = ingest(repo_url, force=True)
    ingest_time = time.time() - start_time
    
    results["ingestion"] = {
        "time_sec": round(ingest_time, 2),
        "files": stats.get("files", 0),
        "chunks": stats.get("chunks", 0),
        "store_size_mb": round(get_dir_size(VECTOR_STORE_DIR), 2)
    }
    
    chain = build_chain(store)
    
    # 2. SMOKE TESTS (Accuracy & Confidence)
    smoke_tests = [
        {
            "name": "FACT_CHECK",
            "q": "What is the name of the main Python file for the user interface?",
            "expected_keyword": "app.py"
        },
        {
            "name": "LOGIC_CHECK",
            "q": "How does the ingestion process handle lockfiles like package-lock.json?",
            "expected_keyword": "skip"
        },
        {
            "name": "NEGATIVE_CHECK",
            "q": "What is the capital of France?",
            "expected_keyword": "Not related to the codebase"
        }
    ]
    
    print("\n🕵️ Running Smoke Tests...")
    for test in smoke_tests:
        print(f"\nTEST [{test['name']}]")
        print(f"Q: {test['q']}")
        
        start_time = time.time()
        answer, sources, score = ask(chain, test['q'])
        latency = time.time() - start_time
        
        passed = test['expected_keyword'].lower() in answer.lower()
        status = "✅ PASSED" if passed else "❌ FAILED"
        
        print(f"A: {answer[:100]}...")
        print(f"Score: {score}% | Latency: {latency:.2f}s | {status}")
        
        results["tests"].append({
            "name": test["name"],
            "question": test["q"],
            "passed": passed,
            "confidence_score": score,
            "latency_sec": round(latency, 2),
            "sources": len(sources)
        })

    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Ingestion Time: {results['ingestion']['time_sec']}s")
    print(f"Total Tests Run: {len(results['tests'])}")
    print(f"Success Rate: {len([t for t in results['tests'] if t['passed']])}/{len(results['tests'])}")
    
    # Save report
    report_name = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_name, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Full report saved to: {report_name}")

if __name__ == "__main__":
    TEST_REPO = "https://github.com/abhi-7-7/COdebase_rag_Assistant"
    run_test_suite(TEST_REPO)
