from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
from datetime import datetime
import statistics

app = Flask(__name__)
CORS(app)

import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
submissions = []

DEFAULT_WEIGHTS = {
    "gwp": 0.40,    
    "circularity": 0.35,  
    "cost": 0.25      
}

def calculate_sustainability_score(gwp, circularity, cost, weights=None):
    """
    Calculate sustainability score based on GWP, Circularity, and Cost.
    
    Args:
        gwp: Global Warming Potential in kg CO2e (lower is better, range 0-100)
        circularity: Circularity score 0-100 (higher is better)
        cost: Product cost in USD (lower is better, range 0-1000)
        weights: Dict with keys 'gwp', 'circularity', 'cost' (must sum to 1.0)
    
    Returns:
        Float score between 0-100
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    weight_sum = weights['gwp'] + weights['circularity'] + weights['cost']
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError("Weights must sum to 1.0")
    
    gwp_normalized = max(0, min(100, float(gwp)))
    gwp_score = 100 - gwp_normalized
    
    circularity_score = max(0, min(100, float(circularity)))
    
    cost_normalized = max(0, min(1000, float(cost)))
    cost_score = 100 - (cost_normalized / 1000 * 100)
    
    final_score = (
        gwp_score * weights['gwp'] +
        circularity_score * weights['circularity'] +
        cost_score * weights['cost']
    )
    
    return round(final_score, 2)

def get_rating(score):
    """Convert score to letter rating."""
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"

def extract_issues(materials, transport, packaging):
    """Extract sustainability issues from product attributes."""
    issues = []
    
    problem_materials = ['plastic', 'pvc', 'polystyrene', 'styrofoam']
    for mat in materials:
        if any(pm in mat.lower() for pm in problem_materials):
            issues.append(f"{mat} material used")
    
    if transport.lower() in ['air', 'air freight', 'airplane']:
        issues.append("Air transport (high emissions)")
    
    non_recyclable = ['plastic', 'mixed materials', 'non-recyclable']
    if any(nr in packaging.lower() for nr in non_recyclable):
        issues.append("Non-recyclable packaging")
    
    return issues

def get_ai_suggestions(product_data, score, rating):
    """Get AI-powered suggestions from Claude."""
    try:
        prompt = f"""Analyze this product's sustainability and provide 3-5 specific, actionable suggestions to improve its environmental impact:

Product: {product_data['product_name']}
Materials: {', '.join(product_data['materials'])}
Weight: {product_data.get('weight_grams', 'N/A')} grams
Transport: {product_data['transport']}
Packaging: {product_data['packaging']}
GWP: {product_data['gwp']} kg CO2e
Cost: ${product_data['cost']}
Circularity Score: {product_data['circularity']}/100
Current Sustainability Score: {score}/100 (Rating: {rating})

Provide ONLY a bulleted list of 3-5 practical suggestions. Be concise and specific. Format as:
- Suggestion 1
- Suggestion 2
etc."""

        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        suggestions_text = message.content[0].text.strip()
        suggestions = [s.strip('- ').strip() for s in suggestions_text.split('\n') if s.strip().startswith('-')]
        return suggestions[:5]  
        
    except Exception as e:
        return [
            "Consider using more sustainable materials",
            "Optimize transport method to reduce emissions",
            "Improve packaging recyclability"
        ]

@app.route('/score', methods=['POST'])
def calculate_score():
    """Calculate sustainability score for a product."""
    try:
        data = request.get_json()
        
        required = ['product_name', 'materials', 'transport', 'packaging', 'gwp', 'cost', 'circularity']
        missing = [field for field in required if field not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing)}"
            }), 400
        
        weights = data.get('weights', DEFAULT_WEIGHTS)
        
        score = calculate_sustainability_score(
            data['gwp'],
            data['circularity'],
            data['cost'],
            weights
        )
        
        rating = get_rating(score)
        
        suggestions = get_ai_suggestions(data, score, rating)
        
        issues = extract_issues(data['materials'], data['transport'], data['packaging'])
        
        result = {
            "product_name": data['product_name'],
            "sustainability_score": score,
            "rating": rating,
            "suggestions": suggestions,
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
        
        submission_record = {
            **data,
            "score": score,
            "rating": rating,
            "suggestions": suggestions,
            "issues": issues,
            "timestamp": result['timestamp'],
            "weights_used": weights
        }
        submissions.append(submission_record)
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Internal error: {str(e)}"}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get all recent submissions."""
    try:
        history = sorted(submissions, key=lambda x: x['timestamp'], reverse=True)
        return jsonify({
            "success": True,
            "count": len(history),
            "submissions": history
        }), 200
    except Exception as e:
        
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/score-summary', methods=['GET'])
def get_summary():
    """Get statistics across all submitted products."""
    try:
        if not submissions:
            return jsonify({
                "success": True,
                "total_products": 0,
                "average_score": 0,
                "ratings": {},
                "top_issues": []
            }), 200
        
        scores = [s['score'] for s in submissions]
        ratings = [s['rating'] for s in submissions]
        
        rating_counts = {}
        for rating in ['A', 'B', 'C', 'D', 'F']:
            rating_counts[rating] = ratings.count(rating)
        
        all_issues = []
        for s in submissions:
            all_issues.extend(s.get('issues', []))
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_issues = [{"issue": issue, "count": count} for issue, count in top_issues]
        
        distribution = {
            "min_score": min(scores),
            "max_score": max(scores),
            "median_score": statistics.median(scores),
            "std_dev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0
        }
        
        return jsonify({
            "success": True,
            "total_products": len(submissions),
            "average_score": round(statistics.mean(scores), 2),
            "ratings": rating_counts,
            "top_issues": top_issues,
            "distribution": distribution,
            "score_range": {
                "excellent": sum(1 for s in scores if s >= 85),
                "good": sum(1 for s in scores if 70 <= s < 85),
                "fair": sum(1 for s in scores if 55 <= s < 70),
                "poor": sum(1 for s in scores if 40 <= s < 55),
                "failing": sum(1 for s in scores if s < 40)
            }
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_data():
    """Clear all submission data (for testing)."""
    global submissions
    submissions = []
    return jsonify({
        "success": True,
        "message": "All data cleared"
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)