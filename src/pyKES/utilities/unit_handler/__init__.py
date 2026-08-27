"""
Custom lightweight unit handler for pyKES.
"""

from .config import DIMENSIONS, ABSOLUTE_TEMPERATURE
from .quantity import Quantity

__all__ = ['Quantity', 'DIMENSIONS', 'ABSOLUTE_TEMPERATURE']
