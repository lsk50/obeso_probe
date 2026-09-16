#!/bin/bash
# =============================================================
# 00_clone_obeso_repo.sh
# 克隆 Obeso 官方仓库到 third_party/ 目录（仅供代码参考）
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PROJECT_DIR/third_party/obeso_repo"

echo "=========================================="
echo "  Step 0: Clone Obeso Reference Repo"
echo "=========================================="

if [ -d "$TARGET_DIR/.git" ]; then
    echo "[INFO] 仓库已存在，执行 git pull 更新..."
    cd "$TARGET_DIR" && git pull
else
    echo "[INFO] 克隆 Obeso 仓库..."
    mkdir -p "$(dirname "$TARGET_DIR")"
    git clone https://github.com/obalcells/hallucination_probes.git "$TARGET_DIR"
fi

echo "[DONE] 仓库已保存到: $TARGET_DIR"
echo ""
echo "关键文件参考:"
echo "  - $TARGET_DIR/README.md          (项目说明)"
echo "  - $TARGET_DIR/probes/             (探针训练代码参考)"
echo "  - $TARGET_DIR/evaluation/         (评估代码参考)"
echo ""
echo "注意: 数据不在 GitHub 仓库中，需要从 HuggingFace 下载。"
echo "      数据下载请看 01_download_obeso_data.py"
