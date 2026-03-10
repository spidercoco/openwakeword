#!/bin/bash

# 获取当前日期时间作为默认提交信息
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
COMMIT_MSG="Auto sync: $TIMESTAMP"

# 如果传入了参数，则使用参数作为提交信息
if [ -n "$1" ]; then
    COMMIT_MSG="$1"
fi

echo "--- Starting Git Sync ---"

# 1. 添加所有更改
echo "Adding changes..."
git add .

# 2. 提交
echo "Committing with message: '$COMMIT_MSG'"
git commit -m "$COMMIT_MSG"

# 3. 推送
echo "Pushing to remote..."
git push

if [ $? -eq 0 ]; then
    echo "--- Sync Successful ---"
else
    echo "--- Sync Failed ---"
    exit 1
fi
