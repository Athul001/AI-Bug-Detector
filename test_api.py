#!/usr/bin/env python3
"""
API Test Script for Bug Detection Tool

This script tests the main endpoints and ML models of the Bug Detection Tool.
It can be run from the command line with: python test_api.py
"""

import unittest
import requests
import json
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

class TestBugDetectionAPI(unittest.TestCase):    
    BASE_URL = "http://localhost:5000"
    
    def setUp(self):
        """Set up test data and initialize models"""
        self.sample_code = """
def example():
    x = 10
    if x == 10:
        print("x is 10")
    else:
        print("x is not 10")"""
        
        # Initialize models with same parameters as in app.py
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def test_home_page(self):
        """Test if the home page loads successfully"""
        response = requests.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200, "Home page should return 200")
        print("✓ Home page test passed")
    
    def test_models_initialization(self):
        """Test if models are initialized with correct parameters"""
        self.assertEqual(self.vectorizer.max_features, 5000)
        self.assertEqual(self.classifier.n_estimators, 100)
        self.assertEqual(self.anomaly_detector.contamination, 0.1)
        print("✓ Model initialization test passed")
        
    def test_vectorizer_fit_transform(self):
        """Test TF-IDF vectorizer fit and transform"""
        sample_docs = ["This is a test", "Another test document", "Python code analysis"]
        X = self.vectorizer.fit_transform(sample_docs)
        self.assertEqual(X.shape[0], 3)  # Number of documents
        self.assertLessEqual(X.shape[1], 5000)  # Max features
        print("✓ Vectorizer fit/transform test passed")
        
    def test_classifier_training(self):
        """Test RandomForest classifier training"""
        X = np.random.rand(10, 5)  # 10 samples, 5 features
        y = np.random.randint(0, 2, 10)  # Binary classification
        self.classifier.fit(X, y)
        self.assertEqual(len(self.classifier.estimators_), 100)
        print("✓ Classifier training test passed")
        
    def test_anomaly_detector(self):
        """Test Isolation Forest anomaly detection"""
        X = np.random.rand(100, 5)  # 100 samples, 5 features
        self.anomaly_detector.fit(X)
        predictions = self.anomaly_detector.predict(X)
        self.assertIn(-1, predictions)  # Should find some anomalies
        self.assertIn(1, predictions)   # And some normal points
        print("✓ Anomaly detector test passed")
        
    def test_code_analysis(self):
        """Test code analysis endpoint with model integration"""
        payload = {
            "code": self.sample_code,
            "language": "python"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/analyze",
            json=payload
        )
        
        self.assertEqual(response.status_code, 200, "API should return 200")
        data = response.json()
        
        # Check response structure
        self.assertIn("success", data, "Response should have 'success' field")
        self.assertIn("results", data, "Response should have 'results' field")
        
        # Check model outputs in response
        results = data["results"]
        self.assertIn("ml_prediction", results, "Results should have 'ml_prediction' field")
        self.assertIn("bug_probability", results["ml_prediction"], 
                     "ML prediction should have 'bug_probability' field")
        
        # Check if bug probability is within valid range
        bug_prob = results["ml_prediction"]["bug_probability"]
        self.assertGreaterEqual(bug_prob, 0, "Bug probability should be >= 0")
        self.assertLessEqual(bug_prob, 1, "Bug probability should be <= 1")
        
        print("✓ Code analysis with model integration test passed")
    
    def test_bugs_list(self):
        """Test bugs list endpoint"""
        response = requests.get(f"{self.BASE_URL}/api/bugs")
        self.assertEqual(response.status_code, 200, "Bugs endpoint should return 200")
        data = response.json()
        self.assertIn("bugs", data, "Response should have 'bugs' field")
        print("✓ Bugs list test passed")
    
    def test_dashboard(self):
        """Test dashboard endpoint"""
        response = requests.get(f"{self.BASE_URL}/api/dashboard")
        self.assertEqual(response.status_code, 200, "Dashboard endpoint should return 200")
        data = response.json()
        self.assertIn("dashboard", data, "Response should have 'dashboard' field")
        self.assertIn("statistics", data["dashboard"], "Dashboard should have 'statistics' field")
        print("✓ Dashboard test passed")

def run_tests():
    """Run all tests and print results"""
    print("Starting API tests...\n" + "="*50)
    
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBugDetectionAPI)
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print final result
    print("\n" + "="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return 0 if result.wasSuccessful() else 1

def check_server_running() -> bool:
    """Check if the Flask server is running"""
    try:
        response = requests.get(f"{TestBugDetectionAPI.BASE_URL}/", timeout=2)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        return False

if __name__ == "__main__":
    import sys
    import subprocess
    import time
    
    # Check if server is running
    if not check_server_running():
        print("\n⚠️  Flask server is not running. Starting the server...\n")
        
        # Start the Flask server in a new process
        flask_process = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        
        # Give the server time to start
        print("Waiting for server to start...")
        time.sleep(3)
        
        # Check if server started successfully
        if not check_server_running():
            print("\n❌ Failed to start the server. Please start it manually with:")
            print("   python app.py\n")
            print("Then run the tests again with:")
            print("   python test_api.py\n")
            sys.exit(1)
    
    # Run the tests
    try:
        sys.exit(run_tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        sys.exit(1)
