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
- `/health` — health check (`google_api_configured` must be `true` for Gemma paraphrase)

## Required: Space secrets

Go to **Settings → Secrets and variables → Secrets** (not Variables):

| Secret name | Value |
|---|---|
| `GOOGLE_API_KEY` | Your key from [Google AI Studio](https://aistudio.google.com/apikey) |

After adding the secret, **Factory rebuild** the Space (Settings → Restart this Space).

## Space variables

**Settings → Secrets and variables → Variables:**

| Variable | Value |
|---|---|
| `GEMMA_API_MODEL` | `gemma-4-26b-a4b-it` |
| `PARAPHRASE_BACKEND` | `api` |
| `GEMMA_API_CHUNKING` | `auto` |
| `ENABLE_LOCAL_GEMMA` | `false` |
| `ENABLE_PERPLEXITY` | `true` |

## Verify

Open `/health` — expect:

```json
{"google_api_configured": true, "gemma_api_model": "gemma-4-26b-a4b-it", ...}
```

If `google_api_configured` is `false`, paraphrase falls back to heuristic (weak rewrites).
