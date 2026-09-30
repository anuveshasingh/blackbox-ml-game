"""
conftest.py — Shared pytest configuration.
"""
import sys
import os

# Make sure the src layout is importable in tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
