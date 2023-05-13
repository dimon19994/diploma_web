from app import app
from .display import display


app.register_blueprint(display)
