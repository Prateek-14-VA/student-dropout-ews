"""
Main routes - landing page + health check.
"""
from flask import Blueprint, jsonify


main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    return '''
    <html>
    <head><title>Student Dropout EWS</title></head>
    <body style="font-family: sans-serif; max-width: 700px; margin: 50px auto;">
        <h1>Student Dropout Early Warning System</h1>
        <p>Flask app is running! Week 5 setup complete.</p>
        <h3>Available endpoints:</h3>
        <ul>
            <li><a href="/health">/health</a> - JSON health check</li>
        </ul>
    </body>
    </html>
    '''


@main_bp.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Backend is healthy'})