---
title: SynthID Watermark Remover
emoji: 🛡️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SynthID Watermark Remover

Adversarial ML pipeline for reversing SynthID text watermarks — landing page, workspace UI, and API in one Space.

- `/` — landing page
- `/app` — live workspace
- `/remove-watermark` — API
- `/health` — health check

## Space secrets (Settings → Repository secrets)

| Secret | Required |
|---|---|
| `GOOGLE_API_KEY` | **Yes** — free Gemma 4 paraphrase via [Google AI Studio](https://aistudio.google.com/apikey) |

## Space variables (Settings → Variables)

| Variable | Value |
|---|---|
| `GEMMA_API_MODEL` | `gemma-4-26b-a4b-it` |
| `PARAPHRASE_BACKEND` | `api` |
| `GEMMA_API_CHUNKING` | `auto` |
| `ENABLE_LOCAL_GEMMA` | `false` |
| `ENABLE_PERPLEXITY` | `true` |

`HF_TOKEN` is only needed if you enable local Gemma weights (`ENABLE_LOCAL_GEMMA=true` on a GPU Space).
