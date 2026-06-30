"""
Span-level NER metrics: exact, partial, and type variants.

Each variant returns an MLStatisticsModel so the frontend can render NER
metrics with the same table as classification.

Span format used everywhere: a list of dicts with keys
    {"start": int, "end": int, "tag": str}
where start/end are character offsets (inclusive start, exclusive end),
mirroring the annotation JSON stored in the DB.

Matching strategy (greedy 1-to-1):
- For each example, gold and predicted spans are matched at most once. A
  predicted span is consumed as soon as it matches a gold span, so a single
  prediction can never inflate two true-positive counts.
- exact: same (start, end, tag).
- partial: same tag, any character-range overlap.
- type: any character-range overlap, ignoring tag (only the overlap
  matters; the predicted tag does not need to equal the gold tag).
"""

from collections import defaultdict
from typing import Iterable

from activetigger.datamodels import MLStatisticsModel

Span = dict  # {"start": int, "end": int, "tag": str}


def _overlaps(a: Span, b: Span) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _match_exact(g: Span, p: Span) -> bool:
    return g["start"] == p["start"] and g["end"] == p["end"] and g["tag"] == p["tag"]


def _match_partial(g: Span, p: Span) -> bool:
    return g["tag"] == p["tag"] and _overlaps(g, p)


def _match_type(g: Span, p: Span) -> bool:
    return _overlaps(g, p)


def _greedy_match(
    gold: list[Span], pred: list[Span], match_fn
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Return (matched pairs as (gold_idx, pred_idx), unmatched gold idx,
    unmatched pred idx). Each gold/pred can match at most one counterpart.
    """
    matched: list[tuple[int, int]] = []
    used_pred: set[int] = set()
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if pi in used_pred:
                continue
            if match_fn(g, p):
                matched.append((gi, pi))
                used_pred.add(pi)
                break
    matched_gold = {gi for gi, _ in matched}
    unmatched_gold = set(range(len(gold))) - matched_gold
    unmatched_pred = set(range(len(pred))) - used_pred
    return matched, unmatched_gold, unmatched_pred


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _round(x: float, decimals: int = 3) -> float:
    return round(float(x), decimals)


def _build_confusion_table(
    confusion: dict[tuple[str, str], int],
    labels: list[str],
    unmatched_gold_per_label: dict[str, int],
    unmatched_pred_per_label: dict[str, int],
) -> dict:
    """Build a `to_dict(orient='split')`-shaped confusion matrix.

    Rows = gold tags (+ "(none)" for spurious pred spans with no gold match).
    Cols = predicted tags (+ "(none)" for missed gold spans).
    For exact/partial flavors all matches sit on the diagonal because the
    match function requires same-tag; for the type flavor off-diagonal
    cells reveal label-confusion patterns.
    """
    NONE = "(none)"
    index = labels + [NONE]
    columns = labels + [NONE]
    data: list[list[int]] = []
    for gold_tag in labels:
        row: list[int] = []
        for pred_tag in labels:
            row.append(confusion.get((gold_tag, pred_tag), 0))
        row.append(unmatched_gold_per_label.get(gold_tag, 0))
        data.append(row)
    # spurious-prediction row (no matching gold)
    spurious_row: list[int] = []
    for pred_tag in labels:
        spurious_row.append(unmatched_pred_per_label.get(pred_tag, 0))
    spurious_row.append(0)
    data.append(spurious_row)
    return {"index": index, "columns": columns, "data": data}


def _compute_flavor(
    gold_per_doc: list[list[Span]],
    pred_per_doc: list[list[Span]],
    labels: list[str],
    match_fn,
    label_for_tp: str = "gold",
) -> MLStatisticsModel:
    """Aggregate TP/FP/FN per label across all documents, then compute
    per-label P/R/F1, aggregate P/R/F1 (micro, macro, weighted), and a
    confusion matrix table consumed by the frontend per-label view.

    label_for_tp:
      "gold": count TP under the gold span's label (used for exact / partial).
      Cross-label confusions appear only as FN (gold) + FP (pred), giving
      type confusions credit only in the "type" flavor.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    confusion: dict[tuple[str, str], int] = defaultdict(int)

    for gold, pred in zip(gold_per_doc, pred_per_doc):
        matched, unmatched_gold, unmatched_pred = _greedy_match(gold, pred, match_fn)
        for gi, pi in matched:
            gold_tag = gold[gi]["tag"]
            pred_tag = pred[pi]["tag"]
            tp_tag = gold_tag if label_for_tp == "gold" else pred_tag
            tp[tp_tag] += 1
            confusion[(gold_tag, pred_tag)] += 1
        for gi in unmatched_gold:
            fn[gold[gi]["tag"]] += 1
        for pi in unmatched_pred:
            fp[pred[pi]["tag"]] += 1

    precision_label: dict[str, float | None] = {}
    recall_label: dict[str, float | None] = {}
    f1_label: dict[str, float | None] = {}
    support_label: dict[str, int] = {}
    f_values: list[float] = []
    total_tp = total_fp = total_fn = 0
    for lab in labels:
        p, r, f = _prf(tp[lab], fp[lab], fn[lab])
        precision_label[lab] = _round(p)
        recall_label[lab] = _round(r)
        f1_label[lab] = _round(f)
        support_label[lab] = tp[lab] + fn[lab]
        f_values.append(f)
        total_tp += tp[lab]
        total_fp += fp[lab]
        total_fn += fn[lab]

    precision_micro, _, f1_micro = _prf(total_tp, total_fp, total_fn)
    f1_macro = sum(f_values) / len(f_values) if f_values else 0.0
    total_support = sum(support_label.values()) or 1
    # Walk f_values + labels in lockstep so the type checker doesn't have to
    # narrow `f1_label[lab]` (a dict[str, float | None] lookup, which it
    # can't prove non-None even under an `is not None` filter).
    f1_weighted = (
        sum(f * support_label[lab] for f, lab in zip(f_values, labels)) / total_support
    )

    table = _build_confusion_table(confusion, labels, fn, fp)

    return MLStatisticsModel(
        training_kind="ner",
        precision_label=precision_label,
        recall_label=recall_label,
        f1_label=f1_label,
        f1_micro=_round(f1_micro),
        f1_macro=_round(f1_macro),
        f1_weighted=_round(f1_weighted),
        # `precision` is the legacy aggregate slot (micro value) preserved
        # for parity with classification metrics.
        precision=_round(precision_micro),
        table=table,
    )


def classify_disagreements(gold_spans: list[Span], pred_spans: list[Span]) -> list[dict]:
    """Classify each gold/pred span pair into one of:
    - "correct": same boundaries AND same tag (not surfaced; returned for parity)
    - "wrong_boundary": same tag, boundary mismatch
    - "wrong_tag": any overlap with gold, different tag
    - "missing": gold span never matched against any pred
    - "spurious": pred span never matched against any gold

    Pairing is greedy and runs strictest-first (exact → boundary →
    tag-overlap → leftover) so the same gold can't be billed as both
    "wrong tag" and "missing".
    """
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    disagreements: list[dict] = []

    def _record(kind: str, gold: Span | None, pred: Span | None):
        disagreements.append(
            {
                "kind": kind,
                "gold": gold,
                "pred": pred,
            }
        )

    # 1) Exact matches — silently consumed.
    for gi, g in enumerate(gold_spans):
        for pi, p in enumerate(pred_spans):
            if pi in used_pred:
                continue
            if _match_exact(g, p):
                used_gold.add(gi)
                used_pred.add(pi)
                break

    # 2) Same tag, overlap but not exact → wrong_boundary.
    for gi, g in enumerate(gold_spans):
        if gi in used_gold:
            continue
        for pi, p in enumerate(pred_spans):
            if pi in used_pred:
                continue
            if g["tag"] == p["tag"] and _overlaps(g, p):
                _record("wrong_boundary", g, p)
                used_gold.add(gi)
                used_pred.add(pi)
                break

    # 3) Any overlap, different tag → wrong_tag.
    for gi, g in enumerate(gold_spans):
        if gi in used_gold:
            continue
        for pi, p in enumerate(pred_spans):
            if pi in used_pred:
                continue
            if _overlaps(g, p):
                _record("wrong_tag", g, p)
                used_gold.add(gi)
                used_pred.add(pi)
                break

    # 4) Anything left is missing/spurious.
    for gi, g in enumerate(gold_spans):
        if gi not in used_gold:
            _record("missing", g, None)
    for pi, p in enumerate(pred_spans):
        if pi not in used_pred:
            _record("spurious", None, p)

    return disagreements


def _build_false_predictions(
    gold_per_doc: list[list[Span]],
    pred_per_doc: list[list[Span]],
    texts: list[str] | None,
    ids: list[str] | None,
) -> list[dict] | None:
    """One entry per document that has at least one disagreement.

    Returns None when texts isn't provided — the frontend modal needs the
    document text to render the inline highlights.
    """
    if texts is None:
        return None
    out: list[dict] = []
    for i, (gold, pred) in enumerate(zip(gold_per_doc, pred_per_doc)):
        disagreements = classify_disagreements(gold, pred)
        if not disagreements:
            continue
        out.append(
            {
                "id": ids[i] if ids is not None and i < len(ids) else str(i),
                "text": texts[i] if i < len(texts) else "",
                "gold_spans": gold,
                "pred_spans": pred,
                "disagreements": disagreements,
            }
        )
    return out


def compute_ner_metrics(
    gold_per_doc: Iterable[list[Span]],
    pred_per_doc: Iterable[list[Span]],
    labels: list[str],
    texts: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
) -> dict[str, MLStatisticsModel]:
    """Return {"exact": ..., "partial": ..., "type": ...}.

    gold_per_doc / pred_per_doc must be aligned (same length, same order).
    When ``texts`` is provided, every flavor's MLStatisticsModel carries
    the same ``false_predictions`` payload (classification of errors is
    independent of which metric flavor is being looked at).
    """
    gold_list = list(gold_per_doc)
    pred_list = list(pred_per_doc)
    if len(gold_list) != len(pred_list):
        raise ValueError(f"gold/pred length mismatch: {len(gold_list)} vs {len(pred_list)}")
    texts_list = list(texts) if texts is not None else None
    ids_list = list(ids) if ids is not None else None
    false_predictions = _build_false_predictions(gold_list, pred_list, texts_list, ids_list)
    flavors = {
        "exact": _compute_flavor(gold_list, pred_list, labels, _match_exact),
        "partial": _compute_flavor(gold_list, pred_list, labels, _match_partial),
        "type": _compute_flavor(gold_list, pred_list, labels, _match_type, label_for_tp="gold"),
    }
    if false_predictions is not None:
        for stat in flavors.values():
            stat.false_predictions = false_predictions
    return flavors
