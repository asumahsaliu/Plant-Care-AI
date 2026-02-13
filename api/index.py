import sys
import os

# Add the Flask Deployed App directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Flask Deployed App'))

from app import app

# Vercel expects the app to be named 'app'
# This is the entry point for Vercel
