"""Centralized safety validation shared by simulation, training, and the API."""

from .shield import SafetyShield

__all__ = ["SafetyShield"]
