from flask import Flask, render_template, request, jsonify, session
import ast
import re
import json
import hashlib
import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import subprocess
import tempfile

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

class CodeAnalyzer:
    def __init__(self):
        self.bug_patterns = {
            'null_pointer': [
                r'\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*\)\s*without\s+null\s+check',
                r'[^=!]=\s*null\s*[^=]',
                r'null\.[a-zA-Z_][a-zA-Z0-9_]*'
            ],
            'memory_leak': [
                r'malloc\s*\([^)]+\)(?!.*free)',
                r'new\s+[a-zA-Z_][a-zA-Z0-9_]*(?!.*delete)',
                r'open\s*\([^)]+\)(?!.*close)'
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*\+.*["\']',
                r'query\s*\(\s*["\'].*%.*["\']',
                r'SELECT.*\+.*FROM'
            ],
            'buffer_overflow': [
                r'strcpy\s*\(',
                r'sprintf\s*\(',
                r'gets\s*\('
            ],
            'race_condition': [
                r'threading\.Thread.*shared_variable',
                r'multiprocessing.*global\s+[a-zA-Z_][a-zA-Z0-9_]*'
            ],
            'logic_error': [
                r'if\s*\([^)]*==[^)]*\)\s*{[^}]*return[^}]*}\s*else\s*{[^}]*return[^}]*}',
                r'for\s*\([^)]*;\s*[^<>]=.*;\s*[^)]*\)',
                r'while\s*\(\s*true\s*\)(?!.*break)'
            ],
            'security_vulnerability': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'system\s*\(',
                r'subprocess\.call\s*\([^)]*shell\s*=\s*True'
            ],
            'performance_issue': [
                r'for\s+[a-zA-Z_][a-zA-Z0-9_]*\s+in\s+range\s*\(\s*len\s*\(',
                r'\.append\s*\([^)]+\)\s*in\s+loop',
                r'time\.sleep\s*\(\s*[0-9]+\s*\)'
            ]
        }
        
        self.complexity_thresholds = {
            'cyclomatic': 10,
            'cognitive': 15,
            'nesting': 4
        }
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False

    def extract_features(self, code: str) -> Dict[str, Any]:
        """Extract comprehensive features from code"""
        try:
            tree = ast.parse(code)
            features = {
                'lines_of_code': len(code.split('\n')),
                'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                'loops': len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]),
                'conditions': len([n for n in ast.walk(tree) if isinstance(n, ast.If)]),
                'try_except': len([n for n in ast.walk(tree) if isinstance(n, ast.Try)]),
                'complexity_score': self.calculate_complexity(tree),
                'nesting_depth': self.calculate_nesting_depth(tree),
                'comment_ratio': self.calculate_comment_ratio(code)
            }
        except SyntaxError:
            # Handle non-Python code or syntax errors
            features = {
                'lines_of_code': len(code.split('\n')),
                'functions': len(re.findall(r'def\s+\w+', code)),
                'classes': len(re.findall(r'class\s+\w+', code)),
                'imports': len(re.findall(r'import\s+\w+|from\s+\w+', code)),
                'loops': len(re.findall(r'for\s+\w+|while\s+\w+', code)),
                'conditions': len(re.findall(r'if\s+\w+', code)),
                'try_except': len(re.findall(r'try:', code)),
                'complexity_score': len(code.split('\n')) * 0.1,
                'nesting_depth': code.count('    ') / 4 if code.count('    ') > 0 else 0,
                'comment_ratio': self.calculate_comment_ratio(code)
            }
        
        return features

    def calculate_complexity(self, tree) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def calculate_nesting_depth(self, tree) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return get_depth(tree)

    def calculate_comment_ratio(self, code: str) -> float:
        """Calculate ratio of comments to code lines"""
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        total_lines = len([line for line in lines if line.strip()])
        return comment_lines / total_lines if total_lines > 0 else 0

    def detect_pattern_bugs(self, code: str) -> List[Dict[str, Any]]:
        """Detect bugs using pattern matching"""
        bugs = []
        lines = code.split('\n')
        
        for bug_type, patterns in self.bug_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    bugs.append({
                        'type': bug_type,
                        'line': line_num,
                        'code': lines[line_num - 1].strip() if line_num <= len(lines) else '',
                        'message': f'Potential {bug_type.replace("_", " ")} detected',
                        'severity': self.get_severity(bug_type),
                        'confidence': 0.7
                    })
        
        return bugs

    def get_severity(self, bug_type: str) -> str:
        """Get severity level for bug type"""
        high_severity = ['sql_injection', 'buffer_overflow', 'security_vulnerability']
        medium_severity = ['null_pointer', 'memory_leak', 'race_condition']
        
        if bug_type in high_severity:
            return 'high'
        elif bug_type in medium_severity:
            return 'medium'
        else:
            return 'low'

    def train_ml_model(self, code_samples: List[str], labels: List[int]):
        """Train machine learning model with code samples"""
        if len(code_samples) < 10:
            # Generate synthetic training data
            synthetic_data = self.generate_training_data()
            code_samples.extend(synthetic_data['code'])
            labels.extend(synthetic_data['labels'])
        
        # Extract features and convert to text for TF-IDF
        text_features = []
        numerical_features = []
        
        for code in code_samples:
            # Text features for TF-IDF
            text_features.append(code)
            
            # Numerical features
            features = self.extract_features(code)
            numerical_features.append([
                features['lines_of_code'],
                features['functions'],
                features['classes'],
                features['complexity_score'],
                features['nesting_depth'],
                features['comment_ratio']
            ])
        
        # Train TF-IDF vectorizer and classifier
        tfidf_features = self.vectorizer.fit_transform(text_features)
        
        # Combine TF-IDF and numerical features
        numerical_features = np.array(numerical_features)
        combined_features = np.hstack([tfidf_features.toarray(), numerical_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    def generate_training_data(self) -> Dict[str, List]:
        """Generate synthetic training data"""
        buggy_code_patterns = [
            "def unsafe_function(data):\n    result = data.process()\n    return result.value",  # Potential null pointer
            "def sql_query(user_input):\n    query = 'SELECT * FROM users WHERE name = ' + user_input\n    execute(query)",  # SQL injection
            "def memory_allocator():\n    ptr = malloc(1024)\n    return ptr",  # Memory leak
            "def infinite_loop():\n    while True:\n        print('running')",  # Logic error
            "def eval_input(user_code):\n    return eval(user_code)",  # Security vulnerability
        ]
        
        clean_code_patterns = [
            "def safe_function(data):\n    if data is not None:\n        result = data.process()\n        return result.value if result else None\n    return None",
            "def safe_sql_query(user_input):\n    query = 'SELECT * FROM users WHERE name = %s'\n    execute(query, (user_input,))",
            "def safe_memory_allocator():\n    ptr = malloc(1024)\n    if ptr:\n        # use ptr\n        free(ptr)\n    return ptr",
            "def controlled_loop(max_iterations):\n    count = 0\n    while count < max_iterations:\n        print('running')\n        count += 1",
            "def safe_eval(user_code):\n    # Parse and validate code safely\n    if validate_code(user_code):\n        return ast.literal_eval(user_code)\n    return None",
        ]
        
        return {
            'code': buggy_code_patterns + clean_code_patterns,
            'labels': [1] * len(buggy_code_patterns) + [0] * len(clean_code_patterns)
        }

    def predict_bugs(self, code: str) -> Dict[str, Any]:
        """Predict bugs using ML model"""
        if not self.is_trained:
            # Train with synthetic data if not trained
            synthetic_data = self.generate_training_data()
            self.train_ml_model(synthetic_data['code'], synthetic_data['labels'])
        
        # Extract features
        text_features = self.vectorizer.transform([code])
        numerical_features = self.extract_features(code)
        numerical_array = np.array([[
            numerical_features['lines_of_code'],
            numerical_features['functions'],
            numerical_features['classes'],
            numerical_features['complexity_score'],
            numerical_features['nesting_depth'],
            numerical_features['comment_ratio']
        ]])
        
        # Combine features
        combined_features = np.hstack([text_features.toarray(), numerical_array])
        
        # Predict
        bug_probability = self.classifier.predict_proba(combined_features)[0]
        is_anomaly = self.anomaly_detector.predict(combined_features)[0] == -1
        
        return {
            'bug_probability': float(bug_probability[1]) if len(bug_probability) > 1 else 0.5,
            'is_anomaly': bool(is_anomaly),
            'features': numerical_features
        }

class BugTracker:
    def __init__(self):
        self.bugs = []
        self.bug_id_counter = 1

    def add_bug(self, bug_data: Dict[str, Any]) -> int:
        bug_id = self.bug_id_counter
        bug = {
            'id': bug_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'open',
            **bug_data
        }
        self.bugs.append(bug)
        self.bug_id_counter += 1
        return bug_id

    def get_bugs(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not filters:
            return self.bugs
        
        filtered_bugs = self.bugs
        if 'severity' in filters:
            filtered_bugs = [b for b in filtered_bugs if b.get('severity') == filters['severity']]
        if 'type' in filters:
            filtered_bugs = [b for b in filtered_bugs if b.get('type') == filters['type']]
        if 'status' in filters:
            filtered_bugs = [b for b in filtered_bugs if b.get('status') == filters['status']]
        
        return filtered_bugs

    def update_bug_status(self, bug_id: int, status: str) -> bool:
        for bug in self.bugs:
            if bug['id'] == bug_id:
                bug['status'] = status
                return True
        return False

    def get_bug_statistics(self) -> Dict[str, Any]:
        if not self.bugs:
            return {'total': 0, 'by_severity': {}, 'by_type': {}, 'by_status': {}}
        
        stats = {
            'total': len(self.bugs),
            'by_severity': {},
            'by_type': {},
            'by_status': {}
        }
        
        for bug in self.bugs:
            # Count by severity
            severity = bug.get('severity', 'unknown')
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
            
            # Count by type
            bug_type = bug.get('type', 'unknown')
            stats['by_type'][bug_type] = stats['by_type'].get(bug_type, 0) + 1
            
            # Count by status
            status = bug.get('status', 'unknown')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
        
        return stats

# Initialize global objects
analyzer = CodeAnalyzer()
bug_tracker = BugTracker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        # Pattern-based bug detection
        pattern_bugs = analyzer.detect_pattern_bugs(code)
        
        # ML-based prediction
        ml_prediction = analyzer.predict_bugs(code)
        
        # Extract features for analysis
        features = analyzer.extract_features(code)
        
        # Generate quality score
        quality_score = calculate_quality_score(features, ml_prediction['bug_probability'])
        
        # Add bugs to tracker
        for bug in pattern_bugs:
            bug_tracker.add_bug(bug)
        
        if ml_prediction['bug_probability'] > 0.7:
            bug_tracker.add_bug({
                'type': 'ml_detected',
                'line': 1,
                'code': code.split('\n')[0][:50] + '...',
                'message': f'ML model detected potential bug (probability: {ml_prediction["bug_probability"]:.2f})',
                'severity': 'medium' if ml_prediction['bug_probability'] > 0.8 else 'low',
                'confidence': ml_prediction['bug_probability']
            })
        
        return jsonify({
            'success': True,
            'results': {
                'pattern_bugs': pattern_bugs,
                'ml_prediction': ml_prediction,
                'features': features,
                'quality_score': quality_score,
                'recommendations': generate_recommendations(features, pattern_bugs, ml_prediction)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        code_samples = data.get('code_samples', [])
        labels = data.get('labels', [])
        
        if len(code_samples) != len(labels):
            return jsonify({'error': 'Code samples and labels must have the same length'}), 400
        
        results = analyzer.train_ml_model(code_samples, labels)
        
        return jsonify({
            'success': True,
            'training_results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bugs', methods=['GET'])
def get_bugs():
    try:
        filters = {
            'severity': request.args.get('severity'),
            'type': request.args.get('type'),
            'status': request.args.get('status')
        }
        filters = {k: v for k, v in filters.items() if v}
        
        bugs = bug_tracker.get_bugs(filters)
        return jsonify({
            'success': True,
            'bugs': bugs,
            'statistics': bug_tracker.get_bug_statistics()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bugs/<int:bug_id>/status', methods=['PUT'])
def update_bug_status(bug_id):
    try:
        data = request.json
        status = data.get('status')
        
        if not status:
            return jsonify({'error': 'Status is required'}), 400
        
        success = bug_tracker.update_bug_status(bug_id, status)
        
        if success:
            return jsonify({'success': True, 'message': 'Bug status updated'})
        else:
            return jsonify({'error': 'Bug not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard')
def dashboard():
    try:
        stats = bug_tracker.get_bug_statistics()
        recent_bugs = sorted(bug_tracker.get_bugs(), 
                           key=lambda x: x['timestamp'], reverse=True)[:10]
        
        # Calculate trend data (mock implementation)
        trend_data = []
        for i in range(7):
            date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
            trend_data.append({
                'date': date,
                'bugs_found': max(0, len(bug_tracker.bugs) - i * 2),
                'bugs_fixed': max(0, len([b for b in bug_tracker.bugs if b['status'] == 'fixed']) - i)
            })
        
        return jsonify({
            'success': True,
            'dashboard': {
                'statistics': stats,
                'recent_bugs': recent_bugs,
                'trend_data': list(reversed(trend_data)),
                'model_status': 'trained' if analyzer.is_trained else 'not_trained'
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def export_bugs():
    try:
        bugs = bug_tracker.get_bugs()
        export_format = request.args.get('format', 'json')
        
        if export_format == 'json':
            return jsonify({'bugs': bugs})
        elif export_format == 'csv':
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['id', 'type', 'severity', 'line', 'message', 'status', 'timestamp'])
            writer.writeheader()
            writer.writerows(bugs)
            
            response = app.response_class(
                response=output.getvalue(),
                status=200,
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=bugs.csv'}
            )
            return response
        
        return jsonify({'error': 'Unsupported format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_quality_score(features: Dict[str, Any], bug_probability: float) -> Dict[str, Any]:
    """Calculate overall code quality score"""
    scores = {
        'complexity': max(0, 100 - features['complexity_score'] * 5),
        'maintainability': max(0, 100 - features['nesting_depth'] * 15),
        'documentation': min(100, features['comment_ratio'] * 500),
        'bug_risk': max(0, 100 - bug_probability * 100)
    }
    
    overall_score = sum(scores.values()) / len(scores)
    
    return {
        'overall': round(overall_score, 2),
        'breakdown': scores,
        'grade': get_quality_grade(overall_score)
    }

def get_quality_grade(score: float) -> str:
    """Convert quality score to letter grade"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

def generate_recommendations(features: Dict[str, Any], pattern_bugs: List[Dict[str, Any]], ml_prediction: Dict[str, Any]) -> List[str]:
    """Generate code improvement recommendations"""
    recommendations = []
    
    if features['complexity_score'] > 10:
        recommendations.append("Consider breaking down complex functions into smaller, more manageable pieces")
    
    if features['nesting_depth'] > 3:
        recommendations.append("Reduce nesting depth by using early returns or extracting nested logic into functions")
    
    if features['comment_ratio'] < 0.1:
        recommendations.append("Add more comments to improve code documentation and maintainability")
    
    if len(pattern_bugs) > 0:
        recommendations.append(f"Address {len(pattern_bugs)} pattern-based potential bugs found in the code")
    
    if ml_prediction['bug_probability'] > 0.7:
        recommendations.append("The ML model indicates high probability of bugs - consider thorough testing")
    
    if ml_prediction['is_anomaly']:
        recommendations.append("Code structure appears unusual - review for potential issues")
    
    if not recommendations:
        recommendations.append("Code quality looks good! Continue following best practices")
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)