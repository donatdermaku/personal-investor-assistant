---
description: Manual guide - Test with Render-like environment locally before deploying
---

# Local Render Testing Guide (Manual)

**NOTE:** This is a reference guide for YOU to run manually. It does NOT contain
any credentials and should NOT be run by agents due to security concerns.

## Setup

1. Create a `.env` file with your Supabase credentials (never commit this file)
2. Ensure `.env` is in your `.gitignore`

## Manual Steps

1. **Load environment:** `source .venv/bin/activate && export $(grep -v '^#' .env | xargs)`
2. **Start server:** `uvicorn src.api.server:app --reload --port 8000`
3. **Test uploads:** Use curl with your own credentials
4. **Run tests:** `python -m pytest tests/unit/ -q`

## Security Notes

- Never share credentials with AI agents
- Use a staging Supabase project for testing when possible
- Keep `.env` out of git
