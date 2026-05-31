#!/bin/bash
echo "=== FLORENCE2 DIAGNOSTIC ==="
python3 -c "
import sys, traceback
sys.path.insert(0, '/comfyui')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('f2init', '/comfyui/custom_nodes/ComfyUI-Florence2/__init__.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print('FLORENCE2 SUCCESS:', list(getattr(mod, 'NODE_CLASS_MAPPINGS', {}).keys()))
except Exception as e:
    print('FLORENCE2 ERROR:', e)
    traceback.print_exc()
"
echo "=== END DIAGNOSTIC ==="
cd /comfyui
python main.py --listen 127.0.0.1 --port 8188 &
until curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; do
    sleep 1
done
echo "ComfyUI ready!"
python -u /handler.py
