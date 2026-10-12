"""
Flask entry point.
Usage:
    python run.py
Then visit: http://127.0.0.1:5000
"""
from app import create_app
from app.models import db


app = create_app()


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='127.0.0.1', port=5000, debug=True)