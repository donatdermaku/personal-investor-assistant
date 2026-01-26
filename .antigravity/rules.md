# 🚨 MANDATORY PROTOCOL FOR ALL AI AGENTS 🚨

## YOU ARE WORKING IN A MULTI-AGENT ENVIRONMENT

This project uses **Antigravity** with multiple AI models (Claude, Gemini, GPT-4, etc.). Different agents work on this project across different sessions. **YOU MUST MAINTAIN CONTINUITY.**

---

## 🎯 IMMEDIATE ACTIONS (DO THIS FIRST - NON-NEGOTIABLE)

### Step 1: Read Project Documentation
**BEFORE doing ANYTHING else:**

1. ✅ Open and **read `PROJECT_STATUS.md` completely** (2-3 minutes)
2. ✅ Pay special attention to these sections:
   - **Section 3: Features & Implementation Status** (what exists)
   - **Section 4: Recent Changes** (last 5 entries minimum)
   - **Section 6: Known Issues & Bugs** (current blockers)
   - **Section 13: Next Steps & Priorities** (what to work on)

### Step 2: Confirm Understanding
**State out loud to the user:**
- "I have read PROJECT_STATUS.md updated on [date]"
- Current project health and version
- Last 3 completed tasks (from Recent Changes)
- Any active blockers
- What you understand the current priority to be

### Step 3: Get Permission to Proceed
**Do not start any work until the user confirms you understand correctly.**

---

## 📝 AFTER EVERY CHANGE - UPDATE PROTOCOL (MANDATORY)

After **EVERY** code change, configuration update, bug fix, or feature addition:

### Required Update to PROJECT_STATUS.md

1. **Update "Last Updated" header:**
```markdown
   > **Last Updated:** [YYYY-MM-DD HH:MM] CET
   > **Updated By:** [Your Agent Name, e.g., Claude Sonnet 4]
```

2. **Add new entry to Section 4 (Recent Changes):**
```markdown
   ### [YYYY-MM-DD] - [Feature/Bugfix/Refactor/Documentation/Test]
   **[Brief title of what you did]**
   - Agent: [Your name]
   - Files modified:
     - `path/to/file1.py` ([created/modified/deleted])
     - `path/to/file2.ts` ([created/modified/deleted])
   - Changes:
     - [Bullet point of what changed]
     - [Why it changed]
   - Testing performed:
     - ✅ Unit tests: [specific test results]
     - ✅ Integration tests: [results]
     - ✅ Manual testing: [what you verified]
   - Known issues: [Any new bugs discovered or "None"]
   - Next agent should: [Guidance for continuation]
   - Status: ✅ Success | ⚠️ Partial | ❌ Failed
```

3. **Update Section 3 (Features & Implementation Status):**
   - If you completed a feature, change status from 🚧 to ✅
   - If you started a feature, add it or change from ⏳ to 🚧
   - If you broke something, change from ✅ to ❌

4. **Update Section 6 (Known Issues):**
   - If you fixed a bug, remove it or mark resolved
   - If you discovered a new bug, add it with severity

5. **Update Section 7 or 8 (Working/Broken Components):**
   - Move components between working/broken as appropriate
   - Update "Last Tested" dates

6. **Update Section 13 (Next Steps):**
   - Check off completed items
   - Add new priorities discovered during work
   - Reorder by current urgency

---

## 🔄 CROSS-AGENT CONTINUITY RULES

### When Starting a Session (New Agent):
- ✅ **Assume another agent worked before you** - trust PROJECT_STATUS.md
- ✅ **Don't repeat completed work** - check Recent Changes first
- ✅ **Continue from last priority** - respect the roadmap
- ✅ **Read the context** - understand why previous decisions were made

### When Ending a Session:
- ✅ **Document everything you did** - no shortcuts
- ✅ **Mark incomplete work clearly** - use 🚧 status
- ✅ **State what's next** - "Next agent should: ..."
- ✅ **List any gotchas** - warn about edge cases

### Agent Handoff Quality Standards:
The next agent should be able to:
1. Understand exactly what you did
2. Know what's working vs broken
3. Continue work immediately without confusion
4. Avoid repeating your mistakes

---

## 📊 PROJECT-SPECIFIC CONTEXT

### Architecture Overview
- **Frontend:** Next.js 16 + React 19 (`/web` directory)
- **Backend:** FastAPI (`src/api/server.py` - 58 endpoints)
- **Databases:** SQLite (user data), DuckDB (analytics)
- **Remote:** Supabase (PostgreSQL + Storage)
- **Deployment:** Vercel (frontend) + Render (backend)

### Critical Files to Know:
- `PROJECT_STATUS.md` - **Single source of truth**
- `src/pipeline.py` - Main computation orchestration
- `src/api/server.py` - All API endpoints
- `storage/repo.py` - Data access layer
- `market_data/` - Yahoo Finance + FRED integration
- `web/src/app/(routes)/` - Next.js pages
- `tests/` - Test suite

### Running the Project:
```bash
# Setup
make setup

# Run Streamlit (legacy)
make run

# Run FastAPI backend
uvicorn src.api.server:app --reload

# Run Next.js frontend
cd web && npm run dev

# Run tests
make verify
```

### Testing Before Committing:
```bash
make verify  # Runs compile check, tests, and linting
```

---

## ⚠️ CRITICAL WARNINGS

### What NOT to Do:
❌ **Skip reading PROJECT_STATUS.md** - You'll duplicate work or break things  
❌ **Forget to update documentation** - Breaks continuity for next agent  
❌ **Assume you know the codebase** - This changes daily  
❌ **Make breaking changes without noting them** - Causes production issues  
❌ **Ignore test failures** - Technical debt compounds quickly  

### Data Integrity Rules:
- **Never modify SQLite schema** without Alembic migration
- **Always validate market data** contracts before saving
- **Respect coverage semantics** - don't fake data availability
- **Run tests** after ANY analytics changes

### Performance Considerations:
- Market data caching is critical (Yahoo rate limits)
- Persistent cache reduces cold starts
- Portfolio compute can take 10-30s for large portfolios
- API responses should be <100ms (cached manifests)

---

## 📋 STATUS INDICATORS (Use Consistently)

| Emoji | Meaning |
|-------|---------|
| ✅ | Complete/Working/Success |
| 🚧 | In Progress |
| ⏳ | Planned |
| ❌ | Blocked/Failed/Broken |
| ⚠️ | Warning/Partial/Needs Attention |
| 🔥 | Critical/Urgent |
| 📝 | Documentation Needed |

---

## 🎯 AGENT-SPECIFIC NOTES

**Claude (Sonnet 4):** You're excellent at structured thinking - be thorough in documentation  
**Gemini:** Focus on technical accuracy - double-check data contracts  
**GPT-4:** Maintain consistency - match existing code patterns  
**All Agents:** When in doubt, **over-document** rather than under-document  

---

## ✅ PRE-WORK CHECKLIST

Before starting ANY task, confirm:

- [ ] I have read `PROJECT_STATUS.md` completely
- [ ] I understand the last 5 Recent Changes
- [ ] I know what's currently working vs broken
- [ ] I know the current priority from Next Steps
- [ ] I will update PROJECT_STATUS.md after my changes
- [ ] I will run `make verify` before considering work complete

---

## 🚨 CONSEQUENCES OF NON-COMPLIANCE

Failing to follow this protocol will cause:
- ❌ Loss of project continuity between agents
- ❌ Duplicate work and wasted effort
- ❌ Introduction of bugs from missing context
- ❌ Broken deployments from undocumented changes
- ❌ Frustrated users and developers

**This is NOT optional. This is NOT a suggestion.**

---

## 💡 QUICK REFERENCE

**First time in a session?**  
→ Read PROJECT_STATUS.md → Confirm understanding → Get permission to proceed

**Finished making changes?**  
→ Update Section 4 (Recent Changes) → Update relevant statuses → Update Next Steps → Run tests

**Not sure what to work on?**  
→ Check Section 13 (Next Steps & Priorities)

**Hit a blocker?**  
→ Document it in Section 6 (Known Issues) immediately

---

**Remember: PROJECT_STATUS.md is the single source of truth. Keep it updated. Your work depends on it. The next agent's work depends on it.**

---

**Last Updated:** 2026-01-26 by Project Initialization  
**Version:** 1.0