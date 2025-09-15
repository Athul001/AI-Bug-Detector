#!/bin/bash

# Test script for Bug Detection Tool API
# Make sure to run the Flask app first: `python app.py`

# Base URL of the API
BASE_URL="http://localhost:5000"

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "\e[32m✓ $2\e[0m"
    else
        echo -e "\e[31m✗ $2\e[0m"
    fi
}

# Test 1: Check if the home page loads
echo "Testing home page..."
curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/" | grep -q 200
print_result $? "Home page should return 200"

# Test 2: Test code analysis with sample Python code
SAMPLE_CODE='def example():
    x = 10
    if x == 10:
        print("x is 10")
    else:
        print("x is not 10")'

echo -e "\nTesting code analysis..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/analyze" \
    -H "Content-Type: application/json" \
    -d "{\"code\": \"$SAMPLE_CODE\", \"language\": \"python\"}")

# Check if response contains expected fields
echo "$RESPONSE" | jq -e '.success' > /dev/null
print_result $? "Analysis should return success status"

echo "$RESPONSE" | jq -e '.features' > /dev/null
print_result $? "Response should contain features"

# Test 3: Get bugs list
echo -e "\nTesting bugs list..."
RESPONSE=$(curl -s "$BASE_URL/api/bugs")

echo "$RESPONSE" | jq -e '.bugs' > /dev/null
print_result $? "Should return bugs list"

# Test 4: Get dashboard data
echo -e "\nTesting dashboard data..."
RESPONSE=$(curl -s "$BASE_URL/api/dashboard")

echo "$RESPONSE" | jq -e '.stats' > /dev/null
print_result $? "Should return dashboard stats"

echo -e "\nTesting complete!"
