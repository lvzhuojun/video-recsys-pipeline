# CLAUDE.md — Project Standards & Compliance Guide

This file is the authoritative reference for Claude when working on this project.
Read it at the start of every session before taking any action.

---

## 1. Project Identity

**Repo:** `lvzhuojun/video-recsys-pipeline`
**Description:** A self-built, end-to-end two-stage video recommendation system.
Covers the full recall → ranking → serving pipeline trained on real KuaiRec 2.0 data.
**Language:** Python 3.10, PyTorch 2.11+cu128, CUDA 12.8, Windows 11 (bash shell via Git Bash).
**Python executable:** `/d/Anaconda3/envs/recsys/python.exe`
**All commands must be run from the project root:** `D:/Lyuzhuojun/Project/recsys/recsys-project`

---

## 2. Document Inventory & Ownership

### 2.1 Public (tracked in git, visible on GitHub)

| File | Purpose | Update trigger |
|------|---------|----------------|
| `README.md` | English project overview, results, quick-start | Any new model, new results, API change |
| `README_zh.md` | Chinese mirror of README.md | Must stay in sync with README.md |
| `CLAUDE.md` | This file — standards & compliance | Any process or structural change |
| `HANDOVER.md` | Project state snapshot for continuity | After each significant iteration |
| `configs/*.yaml` | Hyperparameters (single source of truth) | When tuning or adding models |
| `experiments/results/ablation_report.md` | Ablation analysis narrative | After running new ablation experiments |

### 2.2 Private (local only, listed in .gitignore)

| File | Purpose |
|------|---------|
| `docs/LEARNING_GUIDE.md` | Personal deep-dive study notes — **never commit, never push** |
| `data/raw/` | Raw data files (large, reproducible) |
| `data/processed/` | Processed pickles (reproducible) |
| `experiments/checkpoints/` | Model weights (large binaries) |
| `experiments/logs/` | TensorBoard event files |
| `experiments/results/*.npy` | Large numpy arrays (embeddings) |

### 2.3 Sensitive content policy

- Do **not** add any language targeting specific companies, job titles, or interview formats to any public file.
- Do **not** commit `docs/LEARNING_GUIDE.md` under any name or path.
- Do **not** add the `.claude/` directory to git.

---

## 3. Code Standards

### 3.1 File organisation
- All source code lives under `src/`. No logic in `main.py` beyond argument parsing and orchestration.
- Models: one file per architecture in `src/models/`. Constructor signature: `(meta: dict, cfg: dict)`.
- Training scripts: `src/training/train_<stage>.py`. Each must support CLI args and save results JSON to `experiments/results/`.
- Tests: `tests/test_<module>.py`. Use `MOCK_META` fixtures — never depend on real data files.

### 3.2 Configuration
- **All hyperparameters live in `configs/*.yaml`** — never hardcode in model or training files.
- `base_config.yaml` is the shared root (paths, seed, data dims).
- Model-specific configs extend it without duplicating shared keys.

### 3.3 Style rules
- `encoding="utf-8"` on every `open()` call (Windows default is GBK — causes errors).
- Sequence padding: **right-pad** (valid tokens first, zeros at end). Left-padding breaks causal attention in SASRec.
- All `nn.Embedding` for history sequences use `padding_idx=0`; item IDs are 1-indexed.
- `grad_clip=1.0` is the default in `Trainer` — do not disable without justification.
- Single-worker `uvicorn` for serving on Windows (multiprocessing + CUDA = crash).

### 3.4 Testing
- Run `pytest tests/ -q` after every non-trivial code change before committing.
- 61 tests must pass. Any regression blocks the commit.
- Tests use synthetic tensors — no file I/O, no GPU required.

---

## 4. Document Standards

### 4.1 README.md / README_zh.md

**Structure (must be maintained in this order):**
1. Header (badges, one-line description)
2. Overview + pipeline diagram
3. System Architecture (Mermaid)
4. Model Zoo table
5. Tech Stack table
6. Quick Start (env → data → train → serve)
7. Project Structure (file tree)
8. Experiment Results (retrieval → ranking → MTL → ablation)
9. API Reference
10. Development Guide
11. References (paper links)
12. License

**Rules:**
- README.md and README_zh.md must stay in sync after every change.
- Experiment result tables must reflect the **most recent actual training run**, not placeholders.
- Never use `—` (em-dash placeholder) in result tables — run the experiment first.
- Model parameter counts in the zoo table should be approximate (`~1.3M`).
- Quick-start commands must be tested and runnable.

### 4.2 HANDOVER.md

Captures the current project state for session continuity. Must contain:
- Current iteration number and what was completed
- All latest experiment metrics (copy from results JSON)
- Known issues and workarounds
- Immediate next steps (ordered by priority)
- Environment and path notes

**Update rule:** Update HANDOVER.md at the end of every working session or after any significant change (new model, new results, bug fix with architectural impact).

### 4.3 CLAUDE.md (this file)

Update when:
- A new document is added or removed from the project
- A code standard changes (e.g., padding direction, config structure)
- A new model architecture pattern is established
- A sensitive content policy is clarified

### 4.4 ablation_report.md

Narrative analysis in `experiments/results/ablation_report.md`. Update after running `experiments/run_ablation.py`. Include: what was ablated, numerical results, and a 1–2 sentence interpretation per variant.

---

## 5. Git & Commit Standards

### 5.1 Commit message format

```
<type>: <short description> (<25–60 chars)

Optional body: what changed and why (not how — the diff shows how).
```

Types: `feat` · `fix` · `docs` · `refactor` · `test` · `chore`

### 5.2 What to stage

- Stage specific files by name. Never `git add .` or `git add -A`.
- Never stage: `.claude/`, `data/`, `experiments/checkpoints/`, `experiments/logs/`, `*.log`, `docs/LEARNING_GUIDE.md`.
- Always verify `git status` before committing.

### 5.3 Push policy

- Push only to `main` (single branch project).
- Confirm with user before pushing if the session involved structural changes.
- Never force-push.

---

## 6. Training & Experiment Workflow

### 6.1 Standard training sequence

```bash
# Step 1: data (pick one)
python src/data/download_data.py                   # synthetic mock data
python src/data/prepare_kuairec_real.py            # real KuaiRec 2.0 (preferred)

# Step 2: ranking models
python src/training/train_ranking.py --model all   # deepfm + din + dien

# Step 3: multi-task
python src/training/train_multitask.py

# Step 4: retrieval
python src/training/train_retrieval.py --seq_model mean_pool
python src/training/train_retrieval.py --seq_model sasrec

# Step 5: ablation
python experiments/run_ablation.py

# Step 6: generate charts
python experiments/plot_results.py

# Step 7: tests
python -m pytest tests/ -q
```

### 6.2 After training

- Update result tables in README.md **and** README_zh.md with actual numbers.
- Update HANDOVER.md with new metrics.
- Regenerate figures with `plot_results.py`.
- Commit: `feat: retrain on <data-description> — AUC X.XXX`.

### 6.3 Results file locations

| File | Contents |
|------|----------|
| `experiments/results/ranking_results.json` | DeepFM / DIN / DIEN test metrics |
| `experiments/results/multitask_results.json` | MMoE test metrics |
| `experiments/results/ablation_results.json` | All ablation variant metrics |
| `experiments/results/*_history.json` | Per-epoch training history |
| `experiments/results/figures/` | PNG charts (not git-tracked) |

---

## 7. Known Issues & Workarounds

| Issue | Root cause | Workaround |
|-------|-----------|------------|
| SASRec NaN loss | Left-padded sequences → all-masked rows in causal attn | Right-pad in `feature_engineering.py` (already fixed) |
| `UnicodeDecodeError: gbk` | Windows default encoding | `encoding="utf-8"` on all `open()` calls |
| CUDA not detected (RTX 50xx) | cu121 build doesn't support sm_120 | Use `cu128` PyTorch build |
| FastAPI crash on Windows | Multiprocessing + CUDA | `uvicorn` with `workers=1` |
| MHA type mismatch warning | `attn_mask` float + `key_padding_mask` bool | Known PyTorch deprecation, functionally correct, low priority to fix |

---

## 8. Reading & Updating Private Notes (LEARNING_GUIDE.md)

`docs/LEARNING_GUIDE.md` is a **local-only** personal study reference. It is excluded from git via `.gitignore` and must never appear in any commit or push.

**When Claude should read it:** Only if the user explicitly asks ("check the learning guide", "update LEARNING_GUIDE"). Do not read it proactively.

**When Claude should update it:** Only on explicit user instruction. Changes to the learning guide do not require a git commit — save the file locally and confirm to the user.

**Content rules for the learning guide:**
- Technically accurate and consistent with the current codebase.
- When a model's architecture changes (e.g., SASRec padding fix), update the corresponding section.
- Section structure mirrors the project iteration sequence.

---

## 9. Session Start Checklist

At the start of a new session, before taking any action:

1. Read `HANDOVER.md` for current project state and pending tasks.
2. Read `CLAUDE.md` (this file) for standards.
3. Run `git status` to understand uncommitted changes.
4. Check `experiments/results/*.json` for latest metrics if results are needed.
5. Do **not** read `docs/LEARNING_GUIDE.md` unless explicitly asked.
