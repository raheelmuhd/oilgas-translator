#!/bin/bash
# Oil & Gas Document Translator - Unix/Linux/macOS Start Script
# Run: chmod +x scripts/start.sh && ./scripts/start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "🛢️  Starting Oil & Gas Document Translator..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend server..."
cd "$PROJECT_DIR/backend"
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 3

# Start frontend
echo "Starting frontend server..."
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

sleep 5

echo ""
echo "✅ Application started!"
echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Open browser (macOS or Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:3000
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:3000 2>/dev/null || true
fi

# Wait for processes
wait

