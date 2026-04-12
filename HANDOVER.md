# HANDOVER.md — Project State Snapshot

> **Last updated:** 2026-04-11
> **Purpose:** Session continuity. Read this at the start of every new session alongside `CLAUDE.md`.

---

## 1. Current Iteration

**Iteration 10 + A+ upgrade — COMPLETE**

All planned features are implemented, trained on real data, and documented.

| # | Iteration | Status |
|---|-----------|--------|
| 1 | Data layer (feature engineering, datasets) | ✅ |
| 2 | Two-Tower recall + Faiss retrieval | ✅ |
| 3 | DeepFM and DIN ranking models | ✅ |
| 4 | Ablation study, tests, experimental analysis | ✅ |
| 5 | Complete pipeline, Gradio demo, full documentation | ✅ |
| 6 | SASRec sequential encoder for Two-Tower | ✅ |
| 7 | DIEN (GRU interest extractor + AUGRU) | ✅ |
| 8 | MMoE multi-task (watch_ratio + like) | ✅ |
| 9 | FastAPI serving layer | ✅ |
| 10 | Bilingual README (EN + ZH) | ✅ |
| A+ | Real KuaiRec 2.0 data, correlated synthetic data, SASRec NaN fix, matplotlib charts | ✅ |

---

## 2. Latest Experiment Results

All results from real KuaiRec 2.0 data (small_matrix, 300K interactions, 1,411 users, 3,013 items, 30 categories, 49.8% positive rate).

### Retrieval Stage

| Model | Hit@10 | Hit@50 | Recall@10 | Recall@50 |
|-------|--------|--------|-----------|-----------|
| Two-Tower (MeanPool) | 0.271 | 0.629 | 0.0080 | 0.0359 |
| Two-Tower + SASRec | **0.284** | **0.651** | 0.0071 | 0.0336 |

> Note: Recall@K is low because KuaiRec small_matrix is 93% dense. Hit@K is the meaningful metric.

### Ranking Stage (CTR)

| Model | AUC | GAUC | LogLoss |
|-------|-----|------|---------|
| DeepFM | 0.8769 | 0.8734 | 0.687 |
| DIN | **0.8776** | 0.8733 | 0.688 |
| DIEN | 0.8769 | **0.8736** | **0.674** |

### Multi-Task (MMoE)

| Model | Watch AUC | Like AUC | Watch GAUC | Like GAUC |
|-------|-----------|----------|------------|-----------|
| MMoE (4 experts) | **0.8792** | **0.8835** | **0.8729** | **0.8752** |

### Tests

61 tests passing (`pytest tests/ -q`).

---

## 3. Architecture Overview

```
User Request → Two-Tower / SASRec (recall) → Faiss IndexFlatIP (ANN, top-100)
             → DeepFM / DIN / DIEN (ranking) → top-10 results
             → FastAPI /recommend endpoint (session-aware, GPU-safe)
```

Key design decisions:
- **Right-padding** for history sequences (causal attention safe; left-padding causes SASRec NaN)
- **In-batch InfoNCE** for retrieval (better negative diversity than random negatives)
- **Personalized item sampling** for synthetic data (correlated features → learnable AUC signal)
- **Single-worker uvicorn** on Windows (multiprocessing + CUDA = crash)

---

## 4. Known Issues & Workarounds

| Issue | Status | Workaround |
|-------|--------|------------|
| SASRec NaN loss | **Fixed** | Right-pad in `feature_engineering.py`; guard in `sasrec.py` |
| GBK encoding errors (Windows) | **Fixed** | `encoding="utf-8"` on all `open()` calls |
| CUDA not detected (RTX 50xx) | **Fixed** | Use `cu128` PyTorch build |
| FastAPI crash on Windows | **Fixed** | `workers=1` in uvicorn |
| MHA float/bool type warning | Low priority | PyTorch deprecation warning; functionally correct |
| KuaiRec Recall@K appears low | Not a bug | 93% dense matrix; use Hit@K instead |

---

## 5. Environment

| Item | Value |
|------|-------|
| Python | `/d/Anaconda3/envs/recsys/python.exe` |
| PyTorch | 2.11+cu128 |
| CUDA | 12.8, RTX 5060 Laptop GPU (8 GB VRAM, Blackwell sm_120) |
| OS | Windows 11 Pro, Git Bash shell |
| Working dir | `D:/Lyuzhuojun/Project/recsys/recsys-project` |

---

## 6. File Locations (Key)

| Path | Contents |
|------|----------|
| `src/data/prepare_kuairec_real.py` | Real KuaiRec 2.0 preprocessor (primary data pipeline) |
| `src/data/download_data.py` | Synthetic mock data generator (quick testing) |
| `src/data/feature_engineering.py` | Feature engineering + temporal split; **right-pad sequences** |
| `src/models/sasrec.py` | SASRecBlock + all-padding guard |
| `src/training/train_retrieval.py` | Two-Tower / SASRec training |
| `src/training/train_ranking.py` | DeepFM / DIN / DIEN training (`--model all` trains all three) |
| `src/training/train_multitask.py` | MMoE training |
| `src/serving/serve.py` | FastAPI app (recall + rank + feedback loop) |
| `experiments/run_ablation.py` | 6 ablation experiments |
| `experiments/plot_results.py` | Matplotlib chart generation |
| `experiments/results/` | JSON metrics files (not git-tracked for large files) |
| `docs/LEARNING_GUIDE.md` | **Local only** — personal notes, never commit |

---

## 7. Git State

- **Branch:** `main`
- **Remote:** `lvzhuojun/video-recsys-pipeline`
- **LEARNING_GUIDE.md:** Removed from remote in commit `2531a75` (kept locally via `.gitignore`)
- **Untracked at last session:** `CLAUDE.md`, `HANDOVER.md` (this file) — stage and commit

---

## 8. Immediate Next Steps (Priority Order)

1. **No urgent tasks** — project is feature-complete.
2. Optional: Run `experiments/run_ablation.py` on real KuaiRec data and update ablation tables in README.
3. Optional: Add GAUC to the ablation report for ranking variants.
4. Optional: Simulate online A/B testing with the FastAPI feedback loop.
5. Optional: Try `big_matrix` for larger-scale training (7,176 users × 10,728 items).

---

## 9. Standard Reproduction Commands

```bash
# Full pipeline from scratch (real data)
curl -L https://zenodo.org/records/18164998/files/KuaiRec.zip \
     -o data/raw/kuairec_real/KuaiRec.zip
cd data/raw/kuairec_real && unzip KuaiRec.zip && cd ../../..
python src/data/prepare_kuairec_real.py
python src/training/train_ranking.py --model deepfm
python src/training/train_ranking.py --model din
python src/training/train_ranking.py --model dien
python src/training/train_multitask.py
python src/training/train_retrieval.py
python src/training/train_retrieval.py --seq_model sasrec
python experiments/run_ablation.py
python experiments/plot_results.py
python -m pytest tests/ -q

# Serving
python src/serving/serve.py   # FastAPI at http://localhost:8000/docs
python demo/app.py            # Gradio demo
```
