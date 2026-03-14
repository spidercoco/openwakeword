#!/bin/bash

# 后端工作目录
# 脚本现在在根目录，需要跳转到后端子目录
BACKEND_DIR=$(cd "$(dirname "$0")"/wakeword-web/backend; pwd)
cd "$BACKEND_DIR"

# 配置信息
PORT=8000
APP_NAME="main:app"
LOG_FILE="backend.log"
CONDA_ENV="oww_train"

echo "=== 正在重启后端服务 ==="

# 1. 查找并清理占用端口的旧进程
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
    echo "发现正在运行的进程 (PID: $PID)，正在终止..."
    kill -9 $PID
    sleep 1
else
    echo "未发现运行中的后端进程。"
fi

# 2. 启动后端服务
echo "正在启动后端服务 (端口: $PORT)..."
# 使用 nohup 后台运行，并将日志重定向到 backend.log
nohup python3 main.py > $LOG_FILE 2>&1 &

# 3. 检查是否启动成功
sleep 2
NEW_PID=$(lsof -t -i:$PORT)
if [ -n "$NEW_PID" ]; then
    echo "后端服务已成功启动！"
    echo "PID: $NEW_PID"
    echo "日志文件: $BACKEND_DIR/$LOG_FILE"
else
    echo "启动失败，请检查 $LOG_FILE"
fi
