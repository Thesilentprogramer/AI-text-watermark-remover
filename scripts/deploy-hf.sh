#!/usr/bin/env bash
# Sync this repo to the Hugging Face Space and push.
# Usage: ./scripts/deploy-hf.sh
# Requires: git, HF access token with write access when prompted for password.

set -euo pipefail

SPACE_REPO="https://huggingface.co/spaces/ultimatememer/synthid-watermark-remover"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/.hf-space-deploy"

echo "Project:  $PROJECT_ROOT"
echo "HF clone: $DEPLOY_DIR"

if [[ ! -d "$DEPLOY_DIR/.git" ]]; then
  echo "Cloning Space repo..."
  git clone "$SPACE_REPO" "$DEPLOY_DIR"
fi

echo "Syncing files..."
rsync -av --delete \
  --exclude '.git' \
  --exclude '.env' \
  --exclude '.hf-space-deploy' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude 'synthid-watermark-remover-hf' \
  "$PROJECT_ROOT/" "$DEPLOY_DIR/"

cp "$PROJECT_ROOT/README_HF_SPACE.md" "$DEPLOY_DIR/README.md"

cd "$DEPLOY_DIR"
git add -A
if git diff --cached --quiet; then
  echo "No changes to deploy."
  exit 0
fi

git commit -m "Deploy SynthID Watermark Remover FastAPI app"
echo ""
echo "Pushing to Hugging Face (use your HF token as password if prompted)..."
git push

echo ""
echo "Done. Watch build: https://huggingface.co/spaces/ultimatememer/synthid-watermark-remover"
