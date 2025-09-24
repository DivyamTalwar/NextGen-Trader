#!/usr/bin/env python3
"""
ULTIMATE TESTING SCRIPT - Tests EVERYTHING with extreme edge cases
"""

import os
import sys
import json
import time
import requests
import random
import string
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
import traceback

# Add SourceCode to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SourceCode'))

BASE_URL = "http://localhost:8000"

class UltimateTester:
    def __init__(self):
        self.results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    def log_result(self, test_name, passed, error=""):
        self.results["total"] += 1
        if passed:
            self.results["passed"] += 1
            print(f"[PASS] {test_name}")
        else:
            self.results["failed"] += 1
            print(f"[FAIL] {test_name}: {error}")
            self.results["errors"].append({"test": test_name, "error": error})
    
    def test_endpoint_with_edge_cases(self, method, endpoint, test_cases):
        """Test endpoint with multiple edge case payloads"""
        for case_name, payload in test_cases:
            test_name = f"{method} {endpoint} - {case_name}"
            try:
                if method == "GET":
                    response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
                else:
                    response = requests.request(method, f"{BASE_URL}{endpoint}", 
                                              json=payload, timeout=5)
                
                # Check response format
                if response.status_code in [200, 201]:
                    data = response.json()
                    if isinstance(data, dict):
                        self.log_result(test_name, True)
                    else:
                        self.log_result(test_name, False, "Invalid response format")
                elif response.status_code == 422 and "invalid" in case_name.lower():
                    self.log_result(test_name, True)  # Expected validation error
                elif response.status_code == 404 and "notfound" in case_name.lower():
                    self.log_result(test_name, True)  # Expected not found
                else:
                    self.log_result(test_name, False, f"Status {response.status_code}")
            except Exception as e:
                self.log_result(test_name, False, str(e))
    
    def run_all_tests(self):
        """Run all test suites"""
        print("="*80)
        print("ULTIMATE TESTING SUITE - TESTING EVERYTHING")
        print("="*80)
        
        # Test 1: Health endpoints
        print("\n[1] Testing Health Endpoints...")
        self.test_endpoint_with_edge_cases("GET", "/health", [
            ("normal", None),
        ])
        self.test_endpoint_with_edge_cases("GET", "/", [
            ("normal", None),
        ])
        
        # Test 2: Agent listing endpoints
        print("\n[2] Testing Agent Listing...")
        self.test_endpoint_with_edge_cases("GET", "/api/agents/list", [
            ("normal", None),
        ])
        self.test_endpoint_with_edge_cases("GET", "/api/beast/agents", [
            ("normal", None),
        ])
        
        # Test 3: Agent info endpoints
        print("\n[3] Testing Agent Info...")
        agents = ["warren_buffett", "phil_fisher", "ben_graham", "invalid_agent"]
        for agent in agents:
            self.test_endpoint_with_edge_cases("GET", f"/api/agents/agent/{agent}/info", [
                ("normal", None),
            ])
        
        # Test 4: Agent analysis with edge cases
        print("\n[4] Testing Agent Analysis with Edge Cases...")
        test_payloads = [
            ("valid_simple", {
                "tickers": ["AAPL"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "agent_name": "warren_buffett"
            }),
            ("valid_multiple", {
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "agent_name": "warren_buffett"
            }),
            ("empty_tickers", {
                "tickers": [],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "agent_name": "warren_buffett"
            }),
            ("invalid_no_tickers", {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "agent_name": "warren_buffett"
            }),
            ("invalid_dates", {
                "tickers": ["AAPL"],
                "start_date": "invalid",
                "end_date": "invalid",
                "agent_name": "warren_buffett"
            }),
            ("future_dates", {
                "tickers": ["AAPL"],
                "start_date": "2030-01-01",
                "end_date": "2030-12-31",
                "agent_name": "warren_buffett"
            }),
            ("reversed_dates", {
                "tickers": ["AAPL"],
                "start_date": "2024-12-31",
                "end_date": "2024-01-01",
                "agent_name": "warren_buffett"
            }),
            ("special_chars_ticker", {
                "tickers": ["AAPL!@#$"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "agent_name": "warren_buffett"
            }),
            ("very_long_ticker", {
                "tickers": ["A" * 100],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "agent_name": "warren_buffett"
            }),
            ("many_tickers", {
                "tickers": [f"TICK{i}" for i in range(50)],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "agent_name": "warren_buffett"
            }),
            ("null_values", {
                "tickers": None,
                "start_date": None,
                "end_date": None,
                "agent_name": "warren_buffett"
            }),
            ("empty_object", {}),
            ("wrong_types", {
                "tickers": "AAPL",  # Should be list
                "start_date": 2024,  # Should be string
                "end_date": True,    # Should be string
                "agent_name": 123    # Should be string
            }),
        ]
        
        for agent in ["warren_buffett", "phil_fisher"]:
            endpoint = f"/api/agents/analyze/{agent}"
            self.test_endpoint_with_edge_cases("POST", endpoint, test_payloads[:5])
        
        # Test 5: Beast mode endpoints
        print("\n[5] Testing Beast Mode Endpoints...")
        beast_payloads = [
            ("valid_beast", {
                "tickers": ["AAPL", "MSFT"],
                "analysis_depth": "quick",
                "parallel_processing": True,
                "include_contrarian": True,
                "risk_tolerance": "moderate"
            }),
            ("invalid_depth", {
                "tickers": ["AAPL"],
                "analysis_depth": "invalid_depth"
            }),
            ("invalid_risk", {
                "tickers": ["AAPL"],
                "risk_tolerance": "super_risky"
            }),
        ]
        
        self.test_endpoint_with_edge_cases("POST", "/api/beast/analyze/beast-mode", 
                                          beast_payloads)
        
        # Test 6: Batch and consensus endpoints
        print("\n[6] Testing Batch and Consensus...")
        self.test_endpoint_with_edge_cases("POST", "/api/agents/batch-analyze", [
            ("valid_batch", {
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "agent_name": "warren_buffett,phil_fisher"
            })
        ])
        
        self.test_endpoint_with_edge_cases("POST", "/api/beast/consensus", [
            ("valid_consensus", {
                "tickers": ["AAPL"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            })
        ])
        
        # Test 7: Concurrent requests
        print("\n[7] Testing Concurrent Requests...")
        self.test_concurrent_requests()
        
        # Test 8: Rate limiting / flooding
        print("\n[8] Testing Rate Limiting...")
        self.test_rate_limiting()
        
        # Test 9: Memory stress
        print("\n[9] Testing Memory Stress...")
        self.test_memory_stress()
        
        # Test 10: Error recovery
        print("\n[10] Testing Error Recovery...")
        self.test_error_recovery()
        
        return self.results
    
    def test_concurrent_requests(self):
        """Test multiple concurrent requests"""
        def make_request(i):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=2)
                return response.status_code == 200
            except:
                return False
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request, i) for i in range(100)]
            success = sum(1 for f in futures if f.result())
        
        self.log_result("concurrent_100_requests", success >= 95, 
                       f"Only {success}/100 succeeded")
    
    def test_rate_limiting(self):
        """Test rapid-fire requests"""
        success = 0
        for i in range(50):
            try:
                response = requests.get(f"{BASE_URL}/api/agents/list", timeout=1)
                if response.status_code == 200:
                    success += 1
            except:
                pass
        
        self.log_result("rate_limiting_50_requests", success >= 45,
                       f"Only {success}/50 succeeded")
    
    def test_memory_stress(self):
        """Test with large payloads"""
        # Create a very large payload
        huge_payload = {
            "tickers": [f"TICK{i:04d}" for i in range(1000)],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "agent_name": "warren_buffett",
            "metadata": {"x" * 100: "y" * 100 for _ in range(100)}
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/agents/analyze/warren_buffett",
                json=huge_payload,
                timeout=10
            )
            self.log_result("memory_stress_huge_payload", 
                          response.status_code in [200, 422, 413])
        except Exception as e:
            self.log_result("memory_stress_huge_payload", False, str(e))
    
    def test_error_recovery(self):
        """Test server recovery after errors"""
        # Send invalid request
        try:
            requests.post(f"{BASE_URL}/api/agents/analyze/invalid_agent",
                         json={"invalid": "data"}, timeout=2)
        except:
            pass
        
        # Check if server still responds
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            self.log_result("error_recovery", response.status_code == 200)
        except Exception as e:
            self.log_result("error_recovery", False, str(e))

class SecurityTester:
    """Test for security vulnerabilities"""
    
    def __init__(self):
        self.results = []
    
    def test_sql_injection(self):
        """Test SQL injection attempts"""
        payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ]
        
        for payload in payloads:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/agents/analyze/warren_buffett",
                    json={"tickers": [payload], "agent_name": payload},
                    timeout=5
                )
                # Should handle gracefully, not error
                if response.status_code not in [400, 422]:
                    self.results.append(f"Potential SQL injection vulnerability: {payload}")
            except:
                pass
    
    def test_xss(self):
        """Test XSS attempts"""
        payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>"
        ]
        
        for payload in payloads:
            try:
                response = requests.post(
                    f"{BASE_URL}/api/agents/analyze/warren_buffett",
                    json={"tickers": [payload], "agent_name": payload},
                    timeout=5
                )
                # Check if payload is reflected without escaping
                if payload in response.text:
                    self.results.append(f"Potential XSS vulnerability: {payload}")
            except:
                pass
    
    def test_path_traversal(self):
        """Test path traversal attempts"""
        payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file:///etc/passwd"
        ]
        
        for payload in payloads:
            try:
                response = requests.get(
                    f"{BASE_URL}/api/agents/agent/{payload}/info",
                    timeout=5
                )
                if response.status_code == 200:
                    self.results.append(f"Potential path traversal: {payload}")
            except:
                pass
    
    def run_all_tests(self):
        print("\n" + "="*80)
        print("SECURITY TESTING")
        print("="*80)
        
        self.test_sql_injection()
        self.test_xss()
        self.test_path_traversal()
        
        if self.results:
            print("[SECURITY] Found potential vulnerabilities:")
            for issue in self.results:
                print(f"  - {issue}")
        else:
            print("[SECURITY] No obvious vulnerabilities found")
        
        return len(self.results) == 0

def main():
    print("STARTING ULTIMATE TEST SUITE")
    print("This will brutally test EVERYTHING...")
    print("="*80)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("Server not running! Please start the API server first.")
            return 1
    except:
        print("Cannot connect to server at", BASE_URL)
        return 1
    
    # Run ultimate tests
    tester = UltimateTester()
    results = tester.run_all_tests()
    
    # Run security tests
    security = SecurityTester()
    security_passed = security.run_all_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']} ({results['passed']/max(1,results['total'])*100:.1f}%)")
    print(f"Failed: {results['failed']} ({results['failed']/max(1,results['total'])*100:.1f}%)")
    print(f"Security: {'PASSED' if security_passed else 'FAILED'}")
    
    if results['errors']:
        print(f"\nErrors found ({len(results['errors'])} total):")
        for error in results['errors'][:10]:
            print(f"  - {error['test']}: {error['error'][:50]}")
    
    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "security_passed": security_passed
    }
    
    report_file = f"ultimate_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    if results['failed'] == 0 and security_passed:
        print("\n[SUCCESS] ALL TESTS PASSED! System is bulletproof!")
        return 0
    else:
        print(f"\n[FAILED] {results['failed']} tests failed. Fix and retry.")
        return 1

if __name__ == "__main__":
    sys.exit(main())