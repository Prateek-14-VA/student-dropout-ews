"""
Application configuration.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')

    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    # Use absolute path with forward slashes for SQLite on Windows
    _default_db = 'sqlite:///' + os.path.join(
        BASE_DIR, 'instance', 'app.db'
    ).replace('\\', '/')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', _default_db)
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'production')
    ENCODER_DIR = os.path.join(BASE_DIR, 'models', 'encoders')
    SHAP_DIR = os.path.join(BASE_DIR, 'models', 'shap')

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


def get_config():
    env = os.getenv('FLASK_ENV', 'development').lower()
    if env == 'production':
        return ProductionConfig
    return DevelopmentConfig