"""Serialize Matplotlib work across background renderers, never on the GUI thread."""
from threading import RLock

RENDER_LOCK = RLock()
