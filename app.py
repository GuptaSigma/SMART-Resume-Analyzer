# Vercel Entry Point - Root directory mein rakho

import sys
import os

# Add your nested project directory to path
nested_path = os.path.join(os.path.dirname(__file__), 'smart-hire-main (2)', 'smart-hire-main', 'smart-hire-main')
sys.path.insert(0, nested_path)

# Import your main Flask app
try:
    from app import app  # Ya 'app.py' hai
except ImportError:
    try:
        from main import app  # Agar tumhara main file 'main.py' hai
    except ImportError:
        from application import app  # Ya koi aur naam hai

# Vercel needs this
if __name__ == "__main__":
    app.run()
