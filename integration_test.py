"""
integration_test.py — Net-Neutral AI

Comprehensive integration test for automated federated learning workflow.
Tests all major components end-to-end.
"""

import os
import sys
import tempfile
import pandas as pd

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'backend'))
sys.path.insert(0, os.path.join(project_root, 'backend', 'coordinator'))
sys.path.insert(0, os.path.join(project_root, 'backend', 'client'))

# Import modules
from shared import config, ip_utils
from coordinator import data_distributor


def test_ip_discovery():
    """Test 1: IP address discovery"""
    print("\n" + "=" * 60)
    print("TEST 1: IP Discovery")
    print("=" * 60)

    ip = ip_utils.get_local_ip()
    hostname = ip_utils.get_local_hostname()

    print(f"[OK] Local IP: {ip}")
    print(f"[OK] Hostname: {hostname}")

    assert ip and isinstance(ip, str), "Invalid IP"
    assert hostname and isinstance(hostname, str), "Invalid hostname"
    print("[OK] IP discovery test PASSED")
    return True


def test_config_parameters():
    """Test 2: Configuration parameters"""
    print("\n" + "=" * 60)
    print("TEST 2: Configuration Parameters")
    print("=" * 60)

    assert config.WAIT_FOR_DATA_TIMEOUT_SECS == 300, "Wrong timeout"
    assert config.LOCAL_DATA_DIR == "local_data", "Wrong directory"
    assert "{client_id}" in config.DATA_SHARD_FILENAME, "Wrong filename format"

    print(f"[OK] WAIT_FOR_DATA_TIMEOUT_SECS: {config.WAIT_FOR_DATA_TIMEOUT_SECS}s")
    print(f"[OK] LOCAL_DATA_DIR: {config.LOCAL_DATA_DIR}")
    print(f"[OK] DATA_SHARD_FILENAME: {config.DATA_SHARD_FILENAME}")
    print("[OK] Configuration test PASSED")
    return True


def test_data_distribution():
    """Test 3: Data distribution logic"""
    print("\n" + "=" * 60)
    print("TEST 3: Data Distribution")
    print("=" * 60)

    # Create test dataset
    test_data = {
        'review': [f'Review {i}' for i in range(300)],
        'label': [i % 2 for i in range(300)]
    }
    test_df = pd.DataFrame(test_data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_csv = f.name
        test_df.to_csv(test_csv, index=False)

    try:
        # Test divide_dataset
        print("[TEST] Dividing 300 samples among 3 clients...")
        shards = data_distributor.divide_dataset(test_csv, 3)

        assert len(shards) == 3, "Wrong number of shards"
        print(f"[OK] 3 shards created")

        # Verify distribution
        for client_id, shard in shards.items():
            samples = len(shard)
            print(f"[OK] {client_id}: {samples} samples")
            assert samples > 0, f"{client_id} has no samples"

        total = sum(len(s) for s in shards.values())
        print(f"[OK] Total: {total} samples")
        assert total == 240, "Sample count mismatch"

        # Test validation
        print("[TEST] Validating shards...")
        data_distributor.validate_shards(shards, 240)
        print("[OK] All shards valid")

        print("[OK] Data distribution test PASSED")
        return True

    finally:
        os.unlink(test_csv)


def test_file_structure():
    """Test 4: Project file structure"""
    print("\n" + "=" * 60)
    print("TEST 4: Project File Structure")
    print("=" * 60)

    required_files = {
        'backend/coordinator/server.py': 'Coordinator server',
        'backend/coordinator/data_distributor.py': 'Data distributor',
        'backend/coordinator/fedavg.py': 'FedAvg aggregation',
        'backend/client/client.py': 'Client trainer',
        'backend/client/data.py': 'Data loader',
        'backend/client/model.py': 'Transformer model',
        'backend/shared/config.py': 'Configuration',
        'backend/shared/ip_utils.py': 'IP utilities',
        'frontend/index.html': 'Frontend HTML',
        'frontend/styles.css': 'Frontend CSS',
        'frontend/app.js': 'Frontend JavaScript',
    }

    for filepath, description in required_files.items():
        fullpath = os.path.join(project_root, filepath)
        if os.path.exists(fullpath):
            size_kb = os.path.getsize(fullpath) / 1024
            print(f"[OK] {filepath} ({size_kb:.1f} KB) - {description}")
        else:
            print(f"[ERROR] {filepath} NOT FOUND")
            return False

    print("[OK] File structure test PASSED")
    return True


def test_module_imports():
    """Test 5: Module imports"""
    print("\n" + "=" * 60)
    print("TEST 5: Module Imports")
    print("=" * 60)

    try:
        from shared import config as cfg
        print("[OK] shared.config imported")

        from shared import ip_utils
        print("[OK] shared.ip_utils imported")

        from coordinator import data_distributor
        print("[OK] coordinator.data_distributor imported")

        print("[OK] Module imports test PASSED")
        return True

    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False


def test_syntax_check():
    """Test 6: Python syntax validation"""
    print("\n" + "=" * 60)
    print("TEST 6: Python Syntax Check")
    print("=" * 60)

    python_files = [
        'backend/coordinator/server.py',
        'backend/coordinator/data_distributor.py',
        'backend/client/client.py',
        'backend/client/data.py',
        'backend/shared/config.py',
        'backend/shared/ip_utils.py',
    ]

    import py_compile

    for filepath in python_files:
        fullpath = os.path.join(project_root, filepath)
        try:
            py_compile.compile(fullpath, doraise=True)
            print(f"[OK] {filepath} - syntax valid")
        except py_compile.PyCompileError as e:
            print(f"[ERROR] {filepath} - {e}")
            return False

    print("[OK] Syntax check test PASSED")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("  NET-NEUTRAL AI: END-TO-END INTEGRATION TEST")
    print("=" * 70)

    tests = [
        ("IP Discovery", test_ip_discovery),
        ("Configuration", test_config_parameters),
        ("Data Distribution", test_data_distribution),
        ("File Structure", test_file_structure),
        ("Module Imports", test_module_imports),
        ("Syntax Check", test_syntax_check),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n  [SUCCESS] All integration tests PASSED!")
        print("  System is ready for deployment.")
        print("\n  NEXT STEPS:")
        print("  1. Start coordinator: python backend/coordinator/server.py")
        print("  2. Open frontend: http://localhost:5000/")
        print("  3. Upload dataset (CSV with 'review' and 'label' columns)")
        print("  4. Start 3+ clients: python client.py --client_id client_A/B/C")
        print("  5. Click 'Start Training' in frontend")
        return True
    else:
        print("\n  [FAILURE] Some tests failed. Please fix before deployment.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
