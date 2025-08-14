# Simple API redirect
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main API
if __name__ == "__main__":
    from api import *
    import uvicorn
    import socket
    
    def find_free_port():
        """Find a free port starting from 8000"""
        for port in range(8000, 8010):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return 8000
    
    port = find_free_port()
    print(f"[START] Enhanced Fake Detection API on port {port}")
    print(f"[ACCESS] Web interface at: http://127.0.0.1:{port}")
    print(f"[STATUS] API status at: http://127.0.0.1:{port}/status")
    
    try:
        uvicorn.run("api:app", host="127.0.0.1", port=port, reload=False)
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        input("Press Enter to exit...")
