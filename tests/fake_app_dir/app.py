"""The 'app.py' the launcher test starts: a fake family engine on GRADIO_SERVER_PORT."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fake_engine
if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT") or 7899)
    print(f"fake app booting on {port}", flush=True)
    fake_engine.serve(port).serve_forever()
