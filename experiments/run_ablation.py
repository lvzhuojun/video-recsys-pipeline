"""Ablation study: compare variants of retrieval and ranking models."""
import sys, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import RetrievalDataset, RankingDataset, load_meta, load_split
from src.models.two_tower import TwoTowerModel
from src.models.deepfm import DeepFM
from src.models.din import DIN
from src.retrieval.faiss_index import FaissIndex
from src.evaluation.metrics import compute_retrieval_metrics, compute_auc, compute_gauc
from src.training.trainer import Trainer
from src.utils.gpu_utils import get_device, set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


base_cfg  = read_cfg(ROOT / "configs/base_config.yaml")
ret_cfg   = read_cfg(ROOT / "configs/retrieval_config.yaml")
rank_cfg  = read_cfg(ROOT / "configs/ranking_config.yaml")

proc_dir   = ROOT / base_cfg["data"]["processed_dir"]
meta       = load_meta(proc_dir)
train_data = load_split(proc_dir, "train")
val_data   = load_split(proc_dir, "val")
test_data  = load_split(proc_dir, "test")
DEVICE     = get_device()
set_seed(42)


# ── shared helpers ──────────────────────────────────────────────────────────

def build_item_lookup(*dicts):
    lk = {}
    for d in dicts:
        for i, vid in enumerate(d["item_ids"]):
            v = int(vid)
            if v not in lk:
                lk[v] = {"item_dense":    d["item_dense"][i],
                          "item_category": int(d["item_category"][i]),
                          "item_dur_bkt":  int(d["item_dur_bkt"][i])}
    return lk


@torch.no_grad()
def encode_items(model, lk, bs=512):
    model.eval()
    ids = sorted(lk.keys())
    embs = []
    for s in range(0, len(ids), bs):
        b = ids[s:s+bs]
        id_t  = torch.tensor(b, dtype=torch.long, device=DEVICE)
        den_t = torch.tensor(np.array([lk[i]["item_dense"] for i in b], dtype=np.float32), device=DEVICE)
        cat_t = torch.tensor([lk[i]["item_category"] for i in b], dtype=torch.long, device=DEVICE)
        dur_t = torch.tensor([lk[i]["item_dur_bkt"]  for i in b], dtype=torch.long, device=DEVICE)
        embs.append(model.encode_item(id_t, den_t, cat_t, dur_t).cpu().numpy())
    return np.array(ids, dtype=np.int32), np.vstack(embs)


@torch.no_grad()
def encode_users(model, data, bs=512):
    model.eval()
    last = {}
    for i, uid in enumerate(data["user_ids"]):
        last[int(uid)] = i
    uids = sorted(last.keys())
    idx  = [last[u] for u in uids]
    embs = []
    for s in range(0, len(uids), bs):
        bi = idx[s:s+bs]
        b = {"user_id":    torch.tensor(data["user_ids"][bi],    dtype=torch.long,    device=DEVICE),
             "user_dense": torch.tensor(data["user_dense"][bi],  dtype=torch.float32, device=DEVICE),
             "history_seq":torch.tensor(data["history_seqs"][bi],dtype=torch.long,    device=DEVICE),
             "history_len":torch.tensor(data["history_lens"][bi],dtype=torch.long,    device=DEVICE)}
        embs.append(model.encode_user(b).cpu().numpy())
    return uids, np.vstack(embs)


def eval_retrieval(model, data, lk, k_list=(10, 50)):
    item_ids, item_embs = encode_items(model, lk)
    idx = FaissIndex(dim=item_embs.shape[1], index_type="flat")
    idx.build(item_embs, item_ids)
    uids, user_embs = encode_users(model, data)
    _, retrieved = idx.search(user_embs, top_k=max(k_list))
    user_retrieved = {uid: retrieved[i] for i, uid in enumerate(uids)}
    user_relevant  = defaultdict(set)
    for uid, iid, lbl in zip(data["user_ids"], data["item_ids"], data["labels"]):
        if lbl == 1.0:
            user_relevant[int(uid)].add(int(iid))
    return compute_retrieval_metrics(user_retrieved, user_relevant, list(k_list))


def eval_ranking_auc(model, test_loader):
    model.eval()
    labs, scrs, uids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            scrs.append(model(batch).cpu().numpy())
            labs.append(batch["label"].cpu().numpy())
            uids.append(batch["user_id"].cpu().numpy())
    labs  = np.concatenate(labs)
    scrs  = np.concatenate(scrs)
    uids  = np.concatenate(uids)
    return compute_auc(labs, scrs), compute_gauc(uids, labs, scrs)


def ranking_loss_fn(model, batch):
    pw = torch.tensor(6.0, device=DEVICE)
    preds = model(batch)
    labels = batch["label"]
    weights = torch.where(labels == 1.0, pw, torch.ones_like(pw))
    return (weights * nn.functional.binary_cross_entropy(preds, labels, reduction="none")).mean()


# ── Retrieval ablations ─────────────────────────────────────────────────────

def ablate_retrieval(neg_mode, disable_seq=False, n_epochs=8, seq_model="mean_pool"):
    lk = build_item_lookup(train_data, val_data, test_data)
    ds = RetrievalDataset(train_data, meta, neg_mode=neg_mode, seed=42)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)

    # Build config with seq_model override
    cfg = {**ret_cfg, "model": {**ret_cfg["model"], "seq_model": seq_model}}
    model = TwoTowerModel(meta, cfg).to(DEVICE)

    if disable_seq:
        import types, torch.nn.functional as F2
        def _fwd(self_m, uid, user_dense, history_seq, history_len):
            u = self_m.user_embed(uid)
            z = torch.zeros(uid.size(0), self_m.seq_embed.embedding_dim, device=uid.device)
            d = F2.relu(self_m.dense_proj(user_dense))
            return F2.normalize(self_m.mlp(torch.cat([u, z, d], dim=-1)), p=2, dim=-1)
        model.user_tower.forward = types.MethodType(_fwd, model.user_tower)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    def loss_fn(m, b):
        ue, ie = m(b)
        return m.in_batch_loss(ue, ie)
    trainer = Trainer(model, opt, DEVICE)
    for _ in range(n_epochs):
        trainer.train_one_epoch(loader, loss_fn)
    metrics = eval_retrieval(model, test_data, lk)
    return metrics["recall@10"], metrics["recall@50"]


# ── Ranking ablations ───────────────────────────────────────────────────────

class _MLPBaseline(nn.Module):
    def __init__(self, meta):
        super().__init__()
        din = meta["user_dense_dim"] + meta["item_dense_dim"] + 4
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 1))
    def forward(self, batch):
        x = torch.cat([batch["user_dense"], batch["item_dense"],
                        batch["user_id"].float().unsqueeze(-1) / 500,
                        batch["item_id"].float().unsqueeze(-1) / 1000,
                        batch["item_category"].float().unsqueeze(-1) / 20,
                        batch["item_dur_bkt"].float().unsqueeze(-1) / 5], dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))


def ablate_ranking(model_name, no_fm=False, n_epochs=10):
    ds_tr = RankingDataset(train_data, meta)
    ds_te = RankingDataset(test_data,  meta)
    tr_ld = DataLoader(ds_tr, batch_size=512, shuffle=True,  num_workers=0)
    te_ld = DataLoader(ds_te, batch_size=1024, shuffle=False, num_workers=0)

    if model_name == "deepfm":
        model = DeepFM(meta, rank_cfg).to(DEVICE)
        if no_fm:
            import types
            def _no_fm_fwd(self_m, batch):
                embs = self_m._get_field_embeddings(batch)
                lin  = self_m.linear_proj(embs).squeeze(-1).sum(dim=1)
                deep_out = self_m.deep(embs.view(embs.size(0), -1)).squeeze(-1)
                return torch.sigmoid(lin + deep_out + self_m.bias)
            model.forward = types.MethodType(_no_fm_fwd, model)
    elif model_name == "din":
        model = DIN(meta, rank_cfg).to(DEVICE)
    else:
        model = _MLPBaseline(meta).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, opt, DEVICE)
    for _ in range(n_epochs):
        trainer.train_one_epoch(tr_ld, ranking_loss_fn)
    return eval_ranking_auc(model, te_ld)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    results = {}

    logger.info("Ablation 1/5: Two-Tower in-batch negatives")
    r10, r50 = ablate_retrieval("in_batch")
    results["retrieval_inbatch_recall10"] = r10
    results["retrieval_inbatch_recall50"] = r50
    logger.info(f"  Recall@10={r10:.4f}  Recall@50={r50:.4f}")

    logger.info("Ablation 2/5: Two-Tower random negatives")
    r10, r50 = ablate_retrieval("random")
    results["retrieval_random_recall10"] = r10
    results["retrieval_random_recall50"] = r50
    logger.info(f"  Recall@10={r10:.4f}  Recall@50={r50:.4f}")

    logger.info("Ablation 3/5: Two-Tower without sequence features")
    r10, r50 = ablate_retrieval("in_batch", disable_seq=True)
    results["retrieval_noseq_recall10"] = r10
    results["retrieval_noseq_recall50"] = r50
    logger.info(f"  Recall@10={r10:.4f}  Recall@50={r50:.4f}")

    logger.info("Ablation 4/5: DeepFM vs DeepFM-noFM vs MLP baseline")
    auc, gauc = ablate_ranking("deepfm")
    results["deepfm_auc"] = auc; results["deepfm_gauc"] = gauc
    auc_nofm, gauc_nofm = ablate_ranking("deepfm", no_fm=True)
    results["deepfm_nofm_auc"] = auc_nofm; results["deepfm_nofm_gauc"] = gauc_nofm
    auc_mlp, gauc_mlp = ablate_ranking("mlp")
    results["mlp_auc"] = auc_mlp; results["mlp_gauc"] = gauc_mlp
    logger.info(f"  DeepFM AUC={auc:.4f} | noFM AUC={auc_nofm:.4f} | MLP AUC={auc_mlp:.4f}")

    logger.info("Ablation 5/5: DIN")
    din_auc, din_gauc = ablate_ranking("din")
    results["din_auc"] = din_auc; results["din_gauc"] = din_gauc
    logger.info(f"  DIN AUC={din_auc:.4f}")

    logger.info("Ablation 6/6: SASRec vs Mean-Pool (in-batch, 8 epochs)")
    r10_mp, r50_mp = ablate_retrieval("in_batch", seq_model="mean_pool")
    results["meanpool_recall10"] = r10_mp
    results["meanpool_recall50"] = r50_mp
    logger.info(f"  MeanPool  Recall@10={r10_mp:.4f}  Recall@50={r50_mp:.4f}")

    r10_sa, r50_sa = ablate_retrieval("in_batch", seq_model="sasrec")
    results["sasrec_recall10"] = r10_sa
    results["sasrec_recall50"] = r50_sa
    logger.info(f"  SASRec    Recall@10={r10_sa:.4f}  Recall@50={r50_sa:.4f}")

    delta10 = r10_sa - r10_mp
    delta50 = r50_sa - r50_mp
    logger.info(f"  SASRec delta: Recall@10 {delta10:+.4f} | Recall@50 {delta50:+.4f}")

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out / 'ablation_results.json'}")
    return results


if __name__ == "__main__":
    results = main()
    print("\n=== Ablation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
