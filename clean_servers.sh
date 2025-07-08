#!/bin/bash

echo "🧹 Cleaning up Flask/Dash/Python servers..."

# Kill any processes listening on common dev ports
for port in 5000 8050 8080; do
  PIDS=$(lsof -ti tcp:$port)
  if [ -n "$PIDS" ]; then
    echo "🔪 Killing processes on port $port: $PIDS"
    kill -9 $PIDS
  else
    echo "✅ No process on port $port"
  fi
done

# Kill leftover Flask/Dash/Python processes by name
echo "🔍 Killing stray flask/dash/python processes..."
pkill -f "flask run"
pkill -f "app.py"
pkill -f "sc_explorer.py"
pkill -f "cbBuild"
pkill -f "cbServe"
pkill -f "python.*run_server"

echo "✅ Cleanup complete. You can now restart your servers cleanly."

