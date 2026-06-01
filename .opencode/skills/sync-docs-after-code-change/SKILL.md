---
name: sync-docs-after-code-change
description: >-
  TRIGGER: After ANY edit/write/change to Python files (.py), YAML prompts, frontend HTML, or API endpoints.
  Automatically sync README.md, PocketMemory-教学文档.md, and docs/开发问题与优化方案.md with the code changes.
  Use after modifying agent/, backend/, api/, config/, frontend/, data_ingestion/, or rag/ directories.
  Triggers on: code change, edit file, modify code, implement feature, fix bug, optimize, refactor.
---

# Sync Docs After Code Change

This project maintains three tiers of documentation. After ANY code change, the corresponding docs must be updated.

## Documentation Tiers

| Tier | File | When to Update |
|------|------|---------------|
| **User-facing** | `README.md` | Frontend UI changes, new features, API changes |
| **Teaching/Interview** | `PocketMemory-教学文档.md` | Agent logic changes, architecture changes, prompt changes, new features |
| **Dev Log** | `docs/开发问题与优化方案.md` | Bug fixes, parameter tuning, new problems solved |

## Workflow

### Step 1: Identify Affected Docs

After making a code change, determine which tier(s) are affected:

- **`backend/local_simulate.py`** changed → Dev Log (parameter tuning) + Teaching Doc (if generation behavior changes)
- **`agent/generator.py`** changed → Teaching Doc (agent logic) + Dev Log (if fixing a bug)
- **`agent/router.py`** changed → Teaching Doc
- **`frontend/index.html`** changed → README (UI description) + Teaching Doc (frontend layout section)
- **`config/prompts/*.yaml`** changed → Teaching Doc (prompt design section) + Dev Log (if fixing a prompt issue)
- **`api/server.py`** changed → README (API endpoints) + Teaching Doc
- **New file/module** added → README (project structure) + Teaching Doc (architecture section)

### Step 2: Update Teaching Doc (`PocketMemory-教学文档.md`)

Key sections to check:
- **Architecture diagram** (around line 289-343): Update if module layout changed
- **Technical selection table** (around line 480-495): Update if tech stack changed
- **Prompt design section** (§2.6): Update if any prompt was modified
- **Module sections** (§4-10): Update if module behavior changed
- **Resume summary** (around line 1876-1884): Update if new features added
- **Interview Q&A** (§12): Update Q16/Q18 if relevant
- **Frontend description** (around line 1480-1505): Update if UI changed

### Step 3: Update README

Key sections to check:
- **Architecture diagram**: Reflect current layout
- **Feature list**: Add/remove features
- **Project structure**: Reflect new/deleted files
- **API endpoints**: Add/remove endpoints
- **Tech stack table**: Update if dependencies changed

### Step 4: Update Dev Log (`docs/开发问题与优化方案.md`)

When fixing a bug or tuning parameters:
- Add a new "问题 N" entry under the appropriate 阶段 (phase)
- Include: 现象 (symptoms), 诊断过程 (diagnosis), 根因 (root cause), 解决方案 (solution), 面试要点 (interview talking point)
- Update the 整体评估 table at the end if module status changed

### Step 5: Verify

- Check that all doc references to changed code are accurate
- Verify line numbers and section references still point to correct locations
- Ensure no stale descriptions of the old behavior remain

## Example: Generation Parameter Fix

When `backend/local_simulate.py` was updated with official Qwen parameters:

1. **Dev Log**: Added "问题 22" entry documenting: symptom (repetitive answers), diagnosis (missing repetition_penalty + eos_token_id), root cause (greedy decoding without stop tokens), solution (official Qwen generation_config.json parameters), interview talking point
2. **Teaching Doc**: Updated resume summary to reflect improved generation quality
3. **README**: No change needed (parameter tuning doesn't affect user-facing docs)
