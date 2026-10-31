"""Optional RoboBench ASGI entry point, run from its existing environment."""

from server.app import app
from .robobench_bridge import install

install(app)
