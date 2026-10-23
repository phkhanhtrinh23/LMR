"""Simulated retraining worlds (paper §3.1, §6.1, App. B.3-B.4): RQ2/RQ3/RQ5 experiments.

A mixed-channel marketplace: users arrive directly (H) or through a shopping agent (A); a shared
ranker serves 10-item slates within a subcategory; shoppers choose by Eq. 1 with the App. B.3
utility v_c = β_cj(z) + p_u·q_j + κ_c log(1/log2(m+2)) (κ_A = 0). Sellers revise items; attacked
items receive agent-persuasive revisions at round 1. The ranker is retrained every round on Eq. 3
under a training policy (Pooled, Drop, LMR, ...). Every policy runs in the *actual* world and in the
*human-matched* world (agents' revision response replaced by the human one, §3.1) with common random
numbers; excess exposure = human exposure(actual) / human exposure(human-matched) − 1.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .choice import lift_to_logodds
from .datasets import load_market
from .lmr import COUNT_KEYS, LMRConfig, LMRState, gap_estimate, holdout_assign, run_round
from .rankers import A, H, LightGCN, event_nll, make_ranker, score_items, train_ranker

POLICIES = ["pooled", "drop", "global_weight", "channel_heads", "clipping", "lorec",
            "canonicalized", "facts_only", "holdout_only", "lmr", "oracle"]


@dataclass
class WorldConfig:
    # Marketplace scale (§6.1)
    n_users: int = 2000
    n_cats: int = 5
    items_per_cat: int = 60
    requests_per_round: int = 8000
    slate_size: int = 10
    rounds: int = 6              # training rounds after round 0
    s: float = 0.10              # mean agent share of requests
    delegation_conc: float = 0.5  # Beta(a, a(1-s)/s) concentration of user delegation propensities (A1)
    # Choice model (App. B.3)
    latent_dim: int = 8
    quality_sd: float = 1.0
    pref_sd: float = 1.0
    kappa_H: float = 1.0
    purchase_rate: float = 0.5   # leave utility calibrated so ~50% of requests end in a purchase
    # Population: "synthetic" or a prepared public-data market (datasets.py: amz_e, amz_h, ml_1m)
    dataset: str = "synthetic"
    agent_model: str = "llm"     # data worlds: agent base utility from the LLM harness fit if present, else "same"
    # Revisions (§6.1). scenario "paper": every effect drawn per channel around lifts_pp;
    # "gpa_success": attacks have δ_H = 0 exactly, benign revisions δ_A = δ_H + N(0, ben_gap_sd²).
    scenario: str = "paper"
    ben_gap_sd: float = 0.05
    attack_frac: float = 0.05
    attack_rank: tuple = (11, 40)
    attack_round: int = 1
    benign_frac: float = 0.20
    het: float = 0.10
    base_share: float = 0.125
    lifts_pp: tuple = (("att_A", 8.77), ("att_H", 0.9), ("ben_A", 2.23), ("ben_H", 1.4))
    text_dim: int = 16
    # Serving / evaluation
    serve_temperature: float = 0.5  # Gumbel noise on scores when serving slates (A2)
    eval_requests: int = 6000
    history_len: int = 20
    # Interface views (A5)
    canon_text_scale: float = 0.5
    canon_agent_precision: float = 0.9
    facts_agent_precision: float = 0.7
    # LMR / holdout
    holdout_mult: int = 200      # 0 = own traffic only
    lmr: LMRConfig = field(default_factory=LMRConfig)
    # Baselines (App. B.4)
    global_weight_grid: tuple = (0.0, 0.25, 0.5, 1.0)
    clip_cap: int = 3
    lorec_strength: float = 0.25
    lorec_llm: bool = True       # add the local-LLM item-embedding coherence score when available
    # Ranker training
    ranker: str = "lightgcn"
    dim: int = 32
    epochs: int = 2
    lr: float = 0.01
    batch_size: int = 1024
    weight_decay: float = 1e-6
    device: str = "cpu"
    seed: int = 0


# ---------------------------------------------------------------- ground truth (world-independent)

class Catalog:
    """Users, items, revision schedule and effects, all drawn from the seed only, so the actual,
    human-matched and attack-free worlds share them (common random numbers)."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        rng = np.random.default_rng([cfg.seed, 11])
        n_items = cfg.n_cats * cfg.items_per_cat
        self.n_items = n_items
        self.item_cat = np.repeat(np.arange(cfg.n_cats), cfg.items_per_cat)
        self.market = None
        self.llm_emb = None
        if cfg.dataset == "synthetic":
            self.quality = cfg.quality_sd * rng.standard_normal(n_items)
            self.Q = rng.standard_normal((n_items, cfg.latent_dim)) / np.sqrt(cfg.latent_dim)
            self.P = cfg.pref_sd * rng.standard_normal((cfg.n_users, cfg.latent_dim))
            self.agent_quality = self.quality
        else:   # baseline human preferences fitted to public data (App. B.3)
            m = self.market = load_market(cfg.dataset)
            if (m.n_users, m.n_cats, m.items_per_cat) != (cfg.n_users, cfg.n_cats, cfg.items_per_cat) \
                    or not np.array_equal(m.item_cat, self.item_cat):
                raise ValueError(f"WorldConfig does not match market {cfg.dataset}; use config_for()")
            self.quality, self.Q, self.P = m.quality, m.Q, m.P
            use_llm = cfg.agent_model == "llm" and m.agent_quality is not None
            self.agent_quality = m.agent_quality if use_llm else self.quality
            if m.llm_emb is not None:
                e = m.llm_emb - m.llm_emb.mean(0)
                self.llm_emb = e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-8)
        a = cfg.delegation_conc
        self.propensity = rng.beta(a, a * (1 - cfg.s) / cfg.s, cfg.n_users)
        self.delegating = self.propensity >= np.quantile(self.propensity, 0.9)
        D = {k: lift_to_logodds(cfg.base_share, v / 100) for k, v in dict(cfg.lifts_pp).items()}
        self.effects = D

        # Attacked items: 5% of items, drawn from ranks 11-40 of true attack-free popularity (A3).
        pop = self.quality + (self.P.mean(0) @ self.Q.T)
        lo, hi = cfg.attack_rank
        per_cat = int(round(cfg.attack_frac * n_items / cfg.n_cats))
        attacked = []
        for c in range(cfg.n_cats):
            idx = np.where(self.item_cat == c)[0]
            ranked = idx[np.argsort(-pop[idx])][lo - 1:hi]
            attacked += list(rng.choice(ranked, per_cat, replace=False))
        self.attacked = np.array(sorted(attacked))

        # Revision schedule: benign revisions on 20% of non-attacked items every round (round 0 is the
        # attack-free reference period); attack revisions at `attack_round`.
        total_rounds = cfg.rounds + 1
        self.schedule = {r: [] for r in range(total_rounds)}
        pool = np.setdiff1d(np.arange(n_items), self.attacked)
        n_ben = int(round(cfg.benign_frac * n_items))
        for r in range(total_rounds):
            for j in rng.choice(pool, min(n_ben, len(pool)), replace=False):
                self.schedule[r].append((int(j), "benign"))
        for j in self.attacked:
            self.schedule[cfg.attack_round].append((int(j), "attack"))
        max_v = total_rounds + 1
        self.max_versions = max_v
        self.incA = np.zeros((n_items, max_v))
        self.incH = np.zeros((n_items, max_v))
        self.kind = {}
        version = np.zeros(n_items, int)
        for r in range(total_rounds):
            for j, kind in self.schedule[r]:
                version[j] += 1
                v = version[j]
                key = "att" if kind == "attack" else "ben"
                eA, eH = rng.standard_normal(2)
                if cfg.scenario == "gpa_success":
                    if kind == "attack":
                        self.incA[j, v], self.incH[j, v] = D["att_A"] + cfg.het * eA, 0.0
                    else:
                        self.incH[j, v] = D["ben_H"] + cfg.het * eH
                        self.incA[j, v] = self.incH[j, v] + (D["ben_A"] - D["ben_H"]) + cfg.ben_gap_sd * eA
                else:
                    self.incA[j, v] = D[f"{key}_A"] + cfg.het * eA
                    self.incH[j, v] = D[f"{key}_H"] + cfg.het * eH
                self.kind[(j, v)] = kind
        if self.market is not None and self.market.text_emb is not None:
            t0 = self.market.text_emb
            t0 = (t0 - t0.mean(0)) / t0.std(0).clip(1e-8)
            self.text = rng.standard_normal((n_items, max_v, t0.shape[1])) * 0.3
            self.text[:, 0] = t0
        else:
            self.text = rng.standard_normal((n_items, max_v, cfg.text_dim)) * 0.3
            self.text[:, 0] = rng.standard_normal((n_items, cfg.text_dim))
        self.text = np.cumsum(self.text, axis=1)   # each revision perturbs the previous text features
        self.leave = None

    def base_utility(self, users, items):
        """Fact- and preference-based utility β_j + p_u·q_j (no text, no position)."""
        return self.quality[items] + np.einsum("bd,bkd->bk", self.P[users], self.Q[items])


# ---------------------------------------------------------------- one world

class World:
    """kind: 'actual' | 'hm' (human-matched) | 'attack_free'."""

    def __init__(self, catalog: Catalog, cfg: WorldConfig, kind="actual", policy="pooled"):
        assert kind in ("actual", "hm", "attack_free") and policy in POLICIES
        self.cat, self.cfg, self.kind, self.policy = catalog, cfg, kind, policy
        n_items = catalog.n_items
        incA, incH = catalog.incA.copy(), catalog.incH.copy()
        self.schedule = {r: list(v) for r, v in catalog.schedule.items()}
        if kind == "attack_free":
            for r in self.schedule:
                self.schedule[r] = [(j, k) for j, k in self.schedule[r] if k != "attack"]
            incA[catalog.attacked], incH[catalog.attacked] = 0.0, 0.0
        if kind == "hm":
            incA = incH.copy()
        view = {"canonicalized": (cfg.canon_text_scale, cfg.canon_agent_precision),
                "facts_only": (0.0, cfg.facts_agent_precision)}.get(policy, (1.0, 1.0))
        self.text_scale, self.agent_precision = view
        self.cumA = np.cumsum(incA, axis=1) * self.text_scale
        self.cumH = np.cumsum(incH, axis=1) * self.text_scale
        self.true_cumH = np.cumsum(incH, axis=1)
        self.version = np.zeros(n_items, int)
        self.hist = [[] for _ in range(cfg.n_users)]
        self.events = {k: [] for k in ("user", "channel", "slate", "version", "choice", "hist", "round")}
        self.uses_holdout = policy in ("lmr", "holdout_only")
        self.lmr = LMRState(cfg.lmr)
        self.open_comparisons = []
        self.reference = {}
        self.pending_retries = set()
        torch.manual_seed(cfg.seed)   # identical initialization in every world (common random numbers)
        self.ranker = make_ranker(cfg.ranker, cfg.n_users, n_items, cfg.dim,
                                  channel_heads=(policy == "channel_heads"), text_feats=catalog.text,
                                  max_len=cfg.history_len)
        self.trained = False
        self.log = []
        self.estimates = []

    # -- choice model

    def utilities(self, users, channels, slates, versions):
        """(R, K) channel utilities of the shown versions with the human position term."""
        c = self.cat
        pref = np.einsum("bd,bkd->bk", c.P[users], c.Q[slates])
        is_a = (channels == A)[:, None]
        prec = np.where(is_a, self.agent_precision, 1.0)
        text = np.where(is_a, self.cumA[slates, versions], self.cumH[slates, versions])
        pos = np.log(1.0 / np.log2(np.arange(slates.shape[1]) + 2.0))[None, :]
        base = np.where(is_a, c.agent_quality[slates], c.quality[slates])
        return base + prec * pref + text + np.where(is_a, 0.0, self.cfg.kappa_H * pos)

    def calibrate_leave(self):
        """Leave utilities per channel so that ~purchase_rate of requests buy, on random slates."""
        cfg, c = self.cfg, self.cat
        rng = np.random.default_rng([cfg.seed, 5])
        R = 4000
        users = rng.integers(cfg.n_users, size=R)
        cats = rng.integers(cfg.n_cats, size=R)
        slates = np.stack([rng.choice(cfg.items_per_cat, cfg.slate_size, replace=False) for _ in range(R)])
        slates = slates + cats[:, None] * cfg.items_per_cat
        out = {}
        for ch in (H, A):
            u = self.utilities(users, np.full(R, ch), slates, np.zeros_like(slates))
            lse = np.log(np.exp(u).sum(1))
            lo, hi = -20.0, 20.0
            for _ in range(60):
                mid = (lo + hi) / 2
                buy = np.mean(1 / (1 + np.exp(mid - lse)))
                lo, hi = (mid, hi) if buy > cfg.purchase_rate else (lo, mid)
            out[ch] = (lo + hi) / 2
        c.leave = out

    # -- serving

    def _hist_array(self, users):
        L = self.cfg.history_len
        out = np.full((len(users), L), -1, np.int64)
        for i, u in enumerate(users):
            h = self.hist[u][-L:]
            if h:
                out[i, L - len(h):] = h
        return out

    def _cand(self, cats):
        return cats[:, None] * self.cfg.items_per_cat + np.arange(self.cfg.items_per_cat)[None, :]

    def serve(self, users, cats, channels, noise, temperature=None):
        """Top-K slates by ranker score + temperature·Gumbel; returns global item ids (R, K)."""
        cfg = self.cfg
        T = cfg.serve_temperature if temperature is None else temperature
        cand = self._cand(cats)
        if self.trained:
            vers = self.version[cand]
            s = score_items(self.ranker, users, cand, channels, vers, self._hist_array(users), cfg.device)
        else:
            s = np.zeros(cand.shape)
            T = 1.0
        z = s + T * noise
        top = np.argsort(-z, axis=1)[:, :cfg.slate_size]
        return np.take_along_axis(cand, top, 1)

    # -- one round of traffic

    def requests(self, r):
        cfg, c = self.cfg, self.cat
        rng = np.random.default_rng([cfg.seed, r, 1])
        R = cfg.requests_per_round
        users = rng.integers(cfg.n_users, size=R)
        cats = rng.integers(cfg.n_cats, size=R)
        channels = np.where(rng.random(R) < c.propensity[users], A, H)
        serve_noise = rng.gumbel(size=(R, cfg.items_per_cat))
        choice_noise = rng.gumbel(size=(R, cfg.items_per_cat + 1))
        return users, cats, channels, serve_noise, choice_noise

    def publish(self, r):
        """Apply round r's revisions; open holdouts (C1) for revisions and anchor retries."""
        cats = self.cat.item_cat
        self.open_comparisons = []
        revised = []
        for j, _ in self.schedule[r]:
            self.version[j] += 1
            revised.append(j)
            if self.uses_holdout:
                self.open_comparisons.append(self.lmr.open_comparison(j, int(cats[j]), new_version=self.version[j]))
        if self.uses_holdout:
            for j in sorted(self.pending_retries - set(revised)):
                self.open_comparisons.append(self.lmr.open_comparison(j, int(cats[j])))
        self.pending_retries = set()
        return revised

    def run_round(self, r):
        cfg = self.cfg
        self.publish(r)
        users, cats, channels, serve_noise, choice_noise = self.requests(r)
        slates = self.serve(users, cats, channels, serve_noise)
        versions = self.version[slates].copy()
        cmp_of = {}
        if self.open_comparisons:
            for cm in self.open_comparisons:
                cmp_of[cm.item] = cm
            for k in range(slates.shape[1]):
                for j in np.unique(slates[:, k]):
                    cm = cmp_of.get(int(j))
                    if cm is None:
                        continue
                    rows = np.where(slates[:, k] == j)[0]
                    ctrl = holdout_assign(users[rows], j, cm.comparison_id, channels[rows] == A, cfg.lmr, cfg.seed)
                    versions[rows, k] = np.where(ctrl, cm.control_version, cm.current_version)
        util = self.utilities(users, channels, slates, versions)
        local = slates - cats[:, None] * cfg.items_per_cat
        noisy = util + np.take_along_axis(choice_noise, local, 1)
        leave = np.where(channels == A, self.cat.leave[A], self.cat.leave[H]) + choice_noise[:, -1]
        best = np.argmax(noisy, 1)
        choice = np.where(noisy[np.arange(len(users)), best] > leave, best, cfg.slate_size)
        hist = self._hist_array(users)
        ev = self.events
        ev["user"].append(users); ev["channel"].append(channels); ev["slate"].append(slates)
        ev["version"].append(versions); ev["choice"].append(choice); ev["hist"].append(hist)
        ev["round"].append(np.full(len(users), r))
        bought = choice < cfg.slate_size
        for u, j in zip(users[bought], slates[bought, choice[bought]]):
            self.hist[u].append(int(j))
        if self.open_comparisons:
            self.close_comparisons(r, users, channels, slates, versions, util, choice)

    def close_comparisons(self, r, users, channels, slates, versions, util, choice):
        """Holdout counts n_cz, b_cz (own traffic, or holdout_mult× draws from the choice model), then
        one LMR estimation round (Alg. 2 lines 1-6)."""
        cfg = self.cfg
        comps = self.open_comparisons
        counts = {k: np.zeros(len(comps), np.int64) for k in COUNT_KEYS}
        rng = np.random.default_rng([cfg.seed, r, 2, {"actual": 0, "hm": 1, "attack_free": 2}[self.kind]])
        leave = np.where(channels == A, self.cat.leave[A], self.cat.leave[H])
        expu = np.exp(util)
        tot = expu.sum(1) + np.exp(leave)
        for ci, cm in enumerate(comps):
            rows, ks = np.nonzero(slates == cm.item)
            if len(rows) == 0:
                continue
            ch = channels[rows]
            if cfg.holdout_mult <= 0:   # own traffic
                z1 = versions[rows, ks] == cm.current_version
                buy = choice[rows] == ks
                for chn, name in ((A, "A"), (H, "H")):
                    m = ch == chn
                    counts[f"n{name}0"][ci] += np.sum(m & ~z1); counts[f"b{name}0"][ci] += np.sum(m & ~z1 & buy)
                    counts[f"n{name}1"][ci] += np.sum(m & z1); counts[f"b{name}1"][ci] += np.sum(m & z1 & buy)
                continue
            # Independent draws from the choice model at the logged slates (paper §6.1, B.3).
            others = tot[rows] - expu[rows, ks]
            p = {}
            for z, ver in ((0, cm.control_version), (1, cm.current_version)):
                vz = util[rows, ks] - np.where(ch == A, self.cumA[cm.item, versions[rows, ks]], self.cumH[cm.item, versions[rows, ks]]) \
                    + np.where(ch == A, self.cumA[cm.item, ver], self.cumH[cm.item, ver])
                p[z] = np.exp(vz) / (np.exp(vz) + others)
            for chn, name in ((A, "A"), (H, "H")):
                m = ch == chn
                if not m.any():
                    continue
                eps = cfg.lmr.eps_A if chn == A else cfg.lmr.eps_H
                n0 = rng.binomial(cfg.holdout_mult, eps, m.sum())
                n1 = cfg.holdout_mult - n0
                counts[f"n{name}0"][ci] += n0.sum(); counts[f"b{name}0"][ci] += rng.binomial(n0, p[0][m]).sum()
                counts[f"n{name}1"][ci] += n1.sum(); counts[f"b{name}1"][ci] += rng.binomial(n1, p[1][m]).sum()
        if not self.reference:   # round 0: the attack-free reference period R^ref
            g, v = gap_estimate(counts, cfg.lmr.human_term)
            cats_ = np.array([cm.category for cm in comps])
            self.reference = {int(c): (g[cats_ == c], v[cats_ == c]) for c in np.unique(cats_)}
        if self.policy == "lmr":
            out = run_round(comps, counts, self.reference, self.lmr)
            self.pending_retries = set(out["retries"])
            for k, cm in enumerate(comps):   # RQ4: estimates against the simulator's true gaps
                tg = lambda v: self.cumA[cm.item, v] - self.cumH[cm.item, v]
                self.estimates.append((r, cm.item, cm.control_version, cm.current_version,
                                       self.cat.kind.get((cm.item, cm.current_version), "retry"),
                                       float(tg(cm.current_version) - tg(cm.control_version)),
                                       float(out["g_hat"][k]), float(np.sqrt(out["var"][k])),
                                       float(out["g_tilde"][k]), bool(out["usable"][k]),
                                       float(self.lmr.weight(cm.item, cm.current_version)),
                                       float(np.clip(np.exp(-tg(cm.current_version)), cfg.lmr.w_min, 1.0))))
            self.log.append(dict(round=r, n_comparisons=len(comps), usable=float(out["usable"].mean()),
                                 median_impressions=float(np.median(counts["nA0"] + counts["nA1"] + counts["nH0"] + counts["nH1"]))))

    # -- training

    def stacked(self):
        return {k: np.concatenate(v) for k, v in self.events.items()}

    def event_weights(self, ev, r):
        cfg, K, c = self.cfg, self.cfg.slate_size, self.cat
        n = len(ev["user"])
        w = np.ones(n, np.float32)
        is_a = ev["channel"] == A
        bought = ev["choice"] < K
        rows = np.arange(n)
        item = np.where(bought, ev["slate"][rows, np.minimum(ev["choice"], K - 1)], -1)
        ver = np.where(bought, ev["version"][rows, np.minimum(ev["choice"], K - 1)], 0)
        p = self.policy
        if p == "drop":
            w[is_a] = 0.0
        elif p == "clipping":   # at most clip_cap agent purchases per item, user and round
            m = np.where(is_a & bought)[0]
            seen = {}
            for i in m:
                key = (item[i], ev["user"][i], ev["round"][i])
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > cfg.clip_cap:
                    w[i] = 0.0
        elif p == "lmr":
            for i in np.where(is_a & bought)[0]:
                w[i] = self.lmr.weight(int(item[i]), int(ver[i]))
        elif p == "oracle":
            m = is_a & bought
            g = self.cumA[item[m], ver[m]] - self.cumH[item[m], ver[m]]
            w[m] = np.clip(np.exp(-g), cfg.lmr.w_min, 1.0)
        elif p == "lorec" and self.trained:   # user-wise calibration from the latest round's fit (A6)
            last = np.where(ev["round"] == r)[0]
            nll = event_nll(self.ranker, ev, last, cfg.device)
            users = ev["user"][last]
            per_user = np.bincount(users, nll, cfg.n_users) / np.maximum(np.bincount(users, None, cfg.n_users), 1)
            seen_users = np.bincount(users, None, cfg.n_users) > 0
            med = np.median(per_user[seen_users])
            mad = 1.4826 * np.median(np.abs(per_user[seen_users] - med)) + 1e-8
            z = (per_user - med) / mad
            if cfg.lorec_llm and c.llm_emb is not None:   # LLM-knowledge calibrator (A6)
                z = (z + self._llm_incoherence(ev, last, seen_users)) / np.sqrt(2.0)
            uw = np.clip(1.0 - cfg.lorec_strength * np.maximum(0.0, z - 1.0), 0.2, 1.0)
            w = w * uw[ev["user"]].astype(np.float32)
        return w

    def _llm_incoherence(self, ev, idx, seen_users):
        """Robust z-score per user of 1 − cos(LLM embedding of the item bought, mean embedding of the
        user's earlier purchases) over the round's purchases (0 for users without history)."""
        cfg, K, E = self.cfg, self.cfg.slate_size, self.cat.llm_emb
        ch = ev["choice"][idx]
        b = ch < K
        rows = idx[b]
        item = ev["slate"][rows, ch[b]]
        hist = ev["hist"][rows]
        valid = hist >= 0
        hm = (E[hist.clip(0)] * valid[..., None]).sum(1)
        norm = np.linalg.norm(hm, axis=1)
        has = norm > 0
        inc = np.where(has, 1.0 - (E[item] * hm).sum(1) / norm.clip(1e-8), 0.0)
        users = ev["user"][rows]
        tot = np.bincount(users, inc, cfg.n_users)
        cnt = np.bincount(users, has.astype(float), cfg.n_users)
        per_user = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
        ref = per_user[seen_users & (cnt > 0)]
        if len(ref) < 10:
            return np.zeros(cfg.n_users)
        med = np.median(ref)
        mad = 1.4826 * np.median(np.abs(ref - med)) + 1e-8
        return np.where(cnt > 0, (per_user - med) / mad, 0.0)

    def graph(self, ev, w):
        K = self.cfg.slate_size
        bought = ev["choice"] < K
        idx = np.where(bought)[0]
        return ev["user"][idx], ev["slate"][idx, ev["choice"][idx]], w[idx]

    def train(self, r):
        cfg = self.cfg
        ev = self.stacked()
        tseed = cfg.seed * 1000 + r
        if self.policy == "global_weight":
            rng = np.random.default_rng([cfg.seed, r, 3])
            val = (ev["round"] == r) & (ev["channel"] == H) & (rng.random(len(ev["user"])) < 0.1)
            best = None
            for gw in cfg.global_weight_grid:
                m = copy.deepcopy(self.ranker)
                w = np.where(ev["channel"] == A, gw, 1.0).astype(np.float32)
                w[val] = 0.0
                train_ranker(m, ev, w, cfg.epochs, cfg.lr, cfg.batch_size, cfg.weight_decay, tseed, cfg.device,
                             self.graph(ev, w))
                loss = float(event_nll(m, ev, np.where(val)[0], cfg.device).mean())
                if best is None or loss < best[0]:
                    best = (loss, gw, m)
            self.ranker = best[2]
            self.log.append(dict(round=r, global_weight=best[1]))
        else:
            w = self.event_weights(ev, r)
            train_ranker(self.ranker, ev, w, cfg.epochs, cfg.lr, cfg.batch_size, cfg.weight_decay, tseed,
                         cfg.device, self.graph(ev, w))
        self.trained = True

    # -- evaluation (fixed requests, common random numbers)

    def evaluate(self):
        cfg, c = self.cfg, self.cat
        rng = np.random.default_rng([cfg.seed, 999])
        R = cfg.eval_requests
        users = rng.integers(cfg.n_users, size=R)
        cats = rng.integers(cfg.n_cats, size=R)
        noise = rng.gumbel(size=(R, cfg.items_per_cat))
        del_pool = np.where(c.delegating)[0]
        d_users = rng.choice(del_pool, R)
        d_cats = rng.integers(cfg.n_cats, size=R)
        d_noise = rng.gumbel(size=(R, cfg.items_per_cat))

        out = {}
        hs = self.serve(users, cats, np.full(R, H), noise)
        attacked = np.isin(hs, c.attacked)
        out["exposure_attacked"] = 1000.0 * attacked.sum() / R
        benign_items = np.array(sorted({j for r_ in self.schedule.values() for j, k in r_ if k == "benign"}))
        out["exposure_benign"] = 1000.0 * np.isin(hs, benign_items).sum() / R
        # HR@10 of attacked items: share of (human request, attacked item of the requested category) pairs shown.
        att_per_cat = np.bincount(c.item_cat[c.attacked], minlength=cfg.n_cats)
        out["hr10_attacked"] = float(attacked.sum() / max(1, att_per_cat[cats].sum()))
        out["ndcg_human"] = self._ndcg(users, cats, H)
        out["ndcg_delegating"] = self._ndcg(d_users, d_cats, A)
        ds = self.serve(d_users, d_cats, np.full(R, A), d_noise)
        util = self.utilities(d_users, np.full(R, A), ds, self.version[ds])
        z = np.concatenate([util, np.full((R, 1), c.leave[A])], 1)
        pa = np.exp(z - z.max(1, keepdims=True)); pa /= pa.sum(1, keepdims=True)
        goal = np.argmax(c.base_utility(d_users, ds), 1)
        out["agent_task_success"] = float(np.mean(pa[np.arange(R), goal]))
        return out

    def _ndcg(self, users, cats, channel, k=10):
        """NDCG@10 of the noiseless ranking against the evaluation preference model (true human
        utility of the current versions, A8); gains are human choice probabilities in the category."""
        cfg, c = self.cfg, self.cat
        cand = self._cand(cats)
        rel_u = c.base_utility(users, cand) + self.true_cumH[cand, self.version[cand]]
        gain = np.exp(rel_u - rel_u.max(1, keepdims=True)); gain /= gain.sum(1, keepdims=True)
        if self.trained:
            s = score_items(self.ranker, users, cand, np.full(len(users), channel), self.version[cand],
                            self._hist_array(users), cfg.device)
        else:
            s = np.zeros(cand.shape)
        disc = 1.0 / np.log2(np.arange(k) + 2.0)
        top = np.argsort(-s, 1)[:, :k]
        dcg = (np.take_along_axis(gain, top, 1) * disc).sum(1)
        idcg = (np.sort(gain, 1)[:, ::-1][:, :k] * disc).sum(1)
        return float(np.mean(dcg / idcg))


def run_world(cfg: WorldConfig, kind="actual", policy="pooled", catalog: Catalog | None = None, eval_rounds=None):
    """Round 0 (attack-free reference, random slates) followed by cfg.rounds retraining rounds.

    Returns per-round evaluation after training f_k (k = 0..rounds) and LMR logs.
    """
    torch.set_num_threads(1)
    catalog = catalog or Catalog(cfg)
    if catalog.leave is None:   # calibrated once per catalog under the full-text view
        World(catalog, cfg, "actual", "pooled").calibrate_leave()
    w = World(catalog, cfg, kind, policy)
    results = []
    for r in range(cfg.rounds + 1):
        w.run_round(r)
        w.train(r)
        if eval_rounds is None or r in eval_rounds:
            results.append(dict(round=r, **w.evaluate()))
    est_keys = ("round", "item", "control", "current", "kind", "true_gap", "g_hat", "se", "g_tilde", "usable",
                "weight", "oracle_weight")
    return dict(kind=kind, policy=policy, ranker=cfg.ranker, seed=cfg.seed, s=cfg.s, rounds=results, log=w.log,
                estimates=[dict(zip(est_keys, e)) for e in w.estimates])


def excess(actual, hm, key="exposure_attacked"):
    """§3.1: human exposure in the actual world / human-matched world − 1, per evaluated round."""
    return {a["round"]: a[key] / h[key] - 1.0 for a, h in zip(actual["rounds"], hm["rounds"]) if h[key] > 0}


GPA_SUCCESS = dict(scenario="gpa_success",
                   lifts_pp=(("att_A", 8.77), ("att_H", 0.0), ("ben_A", 3.0), ("ben_H", 3.0)))


def config_for(dataset="synthetic", scenario="paper", **kw):
    """WorldConfig sized to a prepared market (users, subcategories, items per subcategory)."""
    base = dict(GPA_SUCCESS) if scenario == "gpa_success" else {}
    if dataset != "synthetic":
        m = load_market(dataset)
        base.update(n_users=m.n_users, n_cats=m.n_cats, items_per_cat=m.items_per_cat, dataset=dataset)
    base.update(kw)
    return WorldConfig(**base)


def config_dict(cfg: WorldConfig):
    d = asdict(cfg)
    d["lmr"] = asdict(cfg.lmr)
    return d
