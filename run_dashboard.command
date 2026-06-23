#!/bin/bash
# ============================================================================
#  DOUBLE-CLICK THIS FILE on macOS to open the Ads Detection dashboard.
#  (First run sets everything up; later runs just open the dashboard.)
#  No command-line knowledge needed.
# ============================================================================
cd "$(dirname "$0")" || exit 1

echo "📰  Newspaper Ads Detection — starting up..."
echo

# 1) find Python 3
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "❌ Python 3 is not installed."
  echo "   Install it from https://www.python.org/downloads/ then double-click again."
  read -r -p "Press Enter to close."
  exit 1
fi

# 2) create a private virtual environment (one-time)
if [ ! -d ".venv" ]; then
  echo "🛠  First-time setup: creating a private environment..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) install the required libraries (one-time; skipped if already present)
if ! python -c "import streamlit, streamlit_drawable_canvas, cv2" >/dev/null 2>&1; then
  echo "📦  Installing required libraries (one-time, a few minutes)..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements.txt
fi

# 4) launch the dashboard (opens in your web browser) on a unique port
PORT=8765
echo
echo "✅  Opening the dashboard in your browser at:  http://localhost:$PORT"
echo "    (Leave this window open while you work. Close it to stop.)"
echo "    If port $PORT is busy, change the PORT number on this line and re-open."
echo
exec python -m streamlit run app/dashboard.py \
  --server.port "$PORT" \
  --server.address localhost \
  --browser.gatherUsageStats false
