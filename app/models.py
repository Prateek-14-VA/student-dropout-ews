"""
Database models for the Student Dropout EWS.

Tables:
    - User: authentication + role-based access (admin/hod/mentor)
    - Student: anonymized student record
    - Prediction: risk score + SHAP factors history
    - Intervention: mentor-logged action
"""
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='mentor')
    full_name = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    interventions = db.relationship('Intervention', backref='mentor', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Student(db.Model):
    __tablename__ = 'students'

    id = db.Column(db.Integer, primary_key=True)
    anon_id = db.Column(db.String(32), unique=True, nullable=False, index=True)
    department = db.Column(db.String(80))
    semester = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    admission_year = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship('Prediction', backref='student',
                                  lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Student {self.anon_id}>'


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    stage = db.Column(db.String(20), nullable=False)
    predicted_class = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    probabilities_json = db.Column(db.Text)
    top_factors_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    interventions = db.relationship('Intervention', backref='prediction',
                                    lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Prediction {self.id} {self.predicted_class} risk={self.risk_score:.3f}>'


class Intervention(db.Model):
    __tablename__ = 'interventions'

    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    mentor_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    category = db.Column(db.String(30))
    action_taken = db.Column(db.Text)
    status = db.Column(db.String(20), default='suggested')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Intervention {self.id} {self.status}>'