#!/bin/bash
# Night session automated test script
# Usage: ./night_test.sh

set -e

echo "=== Night Session Test ==="
echo ""

# 1. Static check
echo "[1/4] Syntax check..."
python -c "import ast; ast.parse(open('app_v3.py').read())" && echo "  PASS" || { echo "  FAIL"; exit 1; }

# 2. Import check
echo "[2/4] Import check..."
python -c "import ast; compile(open('app_v3.py').read(), 'app_v3.py', 'exec')" && echo "  PASS" || { echo "  FAIL"; exit 1; }

# 3. Headless start
echo "[3/4] Headless start..."
streamlit run app_v3.py --server.headless true --server.port 8501 &
APP_PID=$!
sleep 8

# 4. Health check
echo "[4/4] Health check..."
if curl -sf http://localhost:8501/_stcore/health; then
    echo ""
    echo "  PASS - Streamlit running"
else
    echo "  FAIL - Streamlit not healthy"
    kill $APP_PID 2>/dev/null
    exit 1
fi

# Cleanup
kill $APP_PID 2>/dev/null
sleep 2

echo ""
echo "=== All tests passed ==="
