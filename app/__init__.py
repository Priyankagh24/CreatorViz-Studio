import os
from flask import Flask
from flask_caching import Cache
from flask_cors import CORS

# Root project directory: C:\DataViz-AI
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths OUTSIDE the app folder
TEMPLATES_FOLDER = os.path.join(BASE_DIR, "templates")
OUTPUTS_FOLDER = os.path.join(BASE_DIR, "outputs")


def create_app():
    # Create Flask app with external template + static folders
    app = Flask(
        __name__,
        template_folder="../templates",   # Load templates from C:\DataViz-AI\templates
        static_folder="../outputs",      # Load CSS/JS from C:\DataViz-AI\outputs
        static_url_path="/outputs"         # Browser will access via /outputs/...
    )

    # -----------------------------
    # Ensure "outputs" folder exists
    # -----------------------------
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    # -----------------------------
    # Flask configuration
    # -----------------------------
    CORS(app)

    app.secret_key = "blackshadow"
    app.config["CACHE_TYPE"] = "filesystem"
    app.config["CACHE_DIR"] = os.path.join(BASE_DIR, "cache")
    app.config["CACHE_DEFAULT_TIMEOUT"] = 300
    app.config["SESSION_COOKIE_SECURE"] = False
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Initialize cache
    cache = Cache(app)

    # -----------------------------
    # Register routes
    # -----------------------------
    with app.app_context():
        from .routes import main
        app.register_blueprint(main)

    return app
