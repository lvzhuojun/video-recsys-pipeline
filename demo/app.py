"""Gradio demo for the two-stage video recommendation pipeline.

Shows the full recall → ranking funnel with per-stage scores.

Usage:
    python demo/app.py
    python demo/app.py --share          # public link
    python demo/app.py --port 7861
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np

from main import recommend
from src.data.dataset import load_meta, load_split
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Load metadata once at startup ───────────────────────────────────────────

import yaml

def _read_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

base_cfg = _read_cfg(ROOT / "configs/base_config.yaml")
proc_dir  = ROOT / base_cfg["data"]["processed_dir"]

meta      = load_meta(proc_dir)
test_data = load_split(proc_dir, "test")

N_USERS = int(test_data["user_ids"].max()) + 1
DUR_LABELS = ["< 15s", "15–30s", "30–60s", "1–3min", "3min+"]

# ── Inference function ───────────────────────────────────────────────────────

def run_pipeline(user_id: int, recall_k: int, top_k: int, ranker: str):
    """Called by Gradio on every submit."""
    try:
        results = recommend(
            user_id=int(user_id),
            recall_k=int(recall_k),
            top_k=int(top_k),
            ranker_name=ranker,
            no_train=True,
        )
    except FileNotFoundError as e:
        return (
            f"ERROR: {e}",
            None,
            _user_profile_html(int(user_id)),
        )
    except ValueError as e:
        return (
            f"ERROR: {e}",
            None,
            _user_profile_html(int(user_id)),
        )

    # Build funnel summary
    funnel_html = _funnel_html(int(recall_k), int(top_k), ranker)

    # Build results table
    table_data = []
    for r in results:
        dur_label = DUR_LABELS[r["item_dur_bkt"]] if r["item_dur_bkt"] < len(DUR_LABELS) else str(r["item_dur_bkt"])
        table_data.append([
            r["rank"],
            r["item_id"],
            f"{r['recall_score']:.4f}",
            f"{r['rank_score']:.4f}",
            r["item_category"],
            dur_label,
        ])

    return (
        funnel_html,
        table_data,
        _user_profile_html(int(user_id)),
    )


def _funnel_html(recall_k: int, top_k: int, ranker: str) -> str:
    n_items = meta["n_items"]
    return f"""
<div style="font-family:monospace; background:#1e1e2e; color:#cdd6f4; padding:16px; border-radius:8px; line-height:1.8">
  <b style="color:#89dceb">Recommendation Funnel</b><br><br>
  <span style="color:#a6e3a1">All Items</span>:  {n_items:,} candidates<br>
  <span style="color:#fab387">&#x2193; Two-Tower + Faiss (IP search)</span><br>
  <span style="color:#a6e3a1">Recall Set</span>:  {recall_k} candidates<br>
  <span style="color:#fab387">&#x2193; {ranker.upper()} Ranking</span><br>
  <span style="color:#f38ba8">Final List</span>:  {top_k} recommendations<br><br>
  <span style="color:#6c7086">Compression ratio: {n_items}/{recall_k}/{top_k}
  = {n_items//recall_k}x → {recall_k//top_k}x reduction</span>
</div>
"""


def _user_profile_html(user_id: int) -> str:
    # Find last row for this user in test data
    user_indices = [i for i, uid in enumerate(test_data["user_ids"]) if int(uid) == user_id]
    if not user_indices:
        return f"<p>User {user_id} not in test split.</p>"

    idx = user_indices[-1]
    hist = test_data["history_seqs"][idx]
    hlen = int(test_data["history_lens"][idx])
    actual_hist = hist[:hlen]
    hist_str = ", ".join(str(v) for v in actual_hist[:10])
    if hlen > 10:
        hist_str += f" ... (+{hlen-10} more)"

    positives = [
        int(test_data["item_ids"][i])
        for i in user_indices
        if test_data["labels"][i] == 1.0
    ]

    return f"""
<div style="font-family:monospace; background:#1e1e2e; color:#cdd6f4; padding:16px; border-radius:8px; line-height:1.8">
  <b style="color:#89dceb">User Profile — ID {user_id}</b><br><br>
  <span style="color:#a6e3a1">Watch history (recent {hlen} items):</span><br>
  &nbsp;&nbsp;{hist_str}<br><br>
  <span style="color:#a6e3a1">Positive interactions in test:</span>
  &nbsp;{len(positives)} items<br>
  &nbsp;&nbsp;{", ".join(str(i) for i in positives[:10])}{"..." if len(positives) > 10 else ""}
</div>
"""


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Video RecSys Pipeline Demo",
    ) as app:
        gr.Markdown("""
# Video Recommendation System Demo
**Two-stage pipeline**: Two-Tower Recall (Faiss) → DeepFM / DIN Ranking

*Based on KuaiRec schema | PyTorch + Faiss | RTX 5060 GPU*
""")

        with gr.Row():
            with gr.Column(scale=1):
                user_id_in = gr.Number(
                    label="User ID",
                    value=42,
                    minimum=0,
                    maximum=N_USERS - 1,
                    precision=0,
                    info=f"Valid range: 0 – {N_USERS-1}",
                )
                recall_k_in = gr.Slider(
                    label="Recall K (Two-Tower candidates)",
                    minimum=10, maximum=200, step=10, value=100,
                )
                top_k_in = gr.Slider(
                    label="Top K (final recommendations)",
                    minimum=5, maximum=20, step=1, value=10,
                )
                ranker_in = gr.Radio(
                    label="Ranking Model",
                    choices=["deepfm", "din"],
                    value="deepfm",
                )
                submit_btn = gr.Button("Get Recommendations", variant="primary")

            with gr.Column(scale=2):
                funnel_out = gr.HTML(label="Funnel Summary")
                table_out  = gr.Dataframe(
                    headers=["Rank", "Item ID", "Recall Score", "Rank Score", "Category", "Duration"],
                    label="Top Recommendations",
                    interactive=False,
                )

        user_profile_out = gr.HTML(label="User Profile")

        submit_btn.click(
            fn=run_pipeline,
            inputs=[user_id_in, recall_k_in, top_k_in, ranker_in],
            outputs=[funnel_out, table_out, user_profile_out],
        )

        # Auto-run on launch with default values
        app.load(
            fn=run_pipeline,
            inputs=[user_id_in, recall_k_in, top_k_in, ranker_in],
            outputs=[funnel_out, table_out, user_profile_out],
        )

        gr.Markdown("""
---
**Architecture notes:**
- **Two-Tower**: User tower (ID + dense + sequence GRU pooling) & Item tower (ID + category + duration + dense), L2-normalized embeddings, inner product similarity
- **DeepFM**: 6 sparse fields → shared FM embeddings → FM second-order + Linear + Deep MLP
- **DIN**: Target-aware attention on watch history → weighted pooling → MLP

See [docs/LEARNING_GUIDE.md](../docs/LEARNING_GUIDE.md) for full technical details.
""")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_ui()
    app.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
