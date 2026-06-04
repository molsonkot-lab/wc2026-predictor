#!/bin/bash
# 一键启动世界杯预测工具 + 生成公开分享链接
# 用法: ./share.sh

set -e
cd "$(dirname "$0")"

PORT=8501

# 确保 Streamlit 没有在运行
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 1

echo ""
echo "⚽ 启动 2026 世界杯预测工具..."
echo ""

# 后台启动 Streamlit
python3 -m streamlit run app.py \
  --server.port $PORT \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  > /tmp/streamlit_wc2026.log 2>&1 &
SL_PID=$!
echo "Streamlit PID: $SL_PID"

# 等待 Streamlit 就绪
echo "等待服务启动..."
for i in $(seq 1 15); do
  if curl -s http://localhost:$PORT/_stcore/health > /dev/null 2>&1; then
    echo "✅ 服务已启动"
    break
  fi
  sleep 1
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 生成公开分享链接（localtunnel）..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 启动隧道
node_modules/.bin/lt --port $PORT 2>&1 &
LT_PID=$!

sleep 3

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📱 本地访问:  http://localhost:$PORT"
echo "🔗 上方 'your url is: https://xxx' 即为分享链接"
echo "   将该链接发给朋友，手机电脑均可访问"
echo ""
echo "⚠️  首次访问公开链接时，朋友需要在弹出页面"
echo "   点击 'Click to Continue' 才能进入"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待
wait $LT_PID
