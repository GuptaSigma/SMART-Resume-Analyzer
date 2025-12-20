from app import app
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))  # Gradio uses port 7860
    app.run(host='0.0.0.0', port=port, debug=False)  # DEBUG FALSE! 
