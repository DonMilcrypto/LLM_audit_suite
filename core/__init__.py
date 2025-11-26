"""
Core Module Initialization
--------------------------
Exposes the primary subsystems (Engine, Loader, IO) to the package namespace.
"""

from .engine import AuditEngine
from .data_loader import AuditDataLoader
from .io_manager import IOManager

# Define the public interface for 'from core import *'
__all__ = [
    "AuditEngine", 
    "AuditDataLoader", 
    "IOManager"
]
