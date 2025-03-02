#!/bin/bash

# Get the PID of the first running python process
PID=$(pgrep -o python)

# Check if Python process exists
if [ -z "$PID" ]; then
    echo "No Python process found."
    exit 1
fi

echo "Found Python process with PID: $PID"

# Set CPU affinity to core 0
taskset -cp 0 $PID
if [ $? -eq 0 ]; then
    echo "Set CPU affinity of process $PID to core 0."
else
    echo "Failed to set CPU affinity."
    exit 1
fi

# Set process priority to -20 (highest priority)
sudo renice -n -20 -p $PID
if [ $? -eq 0 ]; then
    echo "Set priority of process $PID to -20."
else
    echo "Failed to set priority. Try running as sudo."
    exit 1
fi

