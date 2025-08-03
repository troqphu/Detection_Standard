# Vercel entry point
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api import app

# This is the entry point for Vercel
app = app
