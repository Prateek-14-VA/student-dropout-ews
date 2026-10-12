"""
Flask application factory.
"""
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

from app.config import get_config


db = SQLAlchemy()
login_manager = LoginManager()


def create_app(config_name=None):
    """Application factory."""
    app = Flask(__name__)

    if config_name:
        from app.config import DevelopmentConfig, ProductionConfig
        app.config.from_object(
            DevelopmentConfig if config_name == 'development' else ProductionConfig
        )
    else:
        app.config.from_object(get_config())

    os.makedirs(os.path.join(app.root_path, '..', 'instance'), exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'

    from app.routes.main import main_bp
    app.register_blueprint(main_bp)

    return app