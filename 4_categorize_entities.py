#!/usr/bin/env python3
"""
Categorize entities as person or event based on flattened statements.

Rule (explicit):
- person if any statement has key == "P31" and value == "Q5" (human)
- otherwise event

This module exposes functions and an optional minimal runner.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator
from tqdm import tqdm
import requests


# Vietnam-related country QIDs (any one qualifies a person for inclusion)
VIETNAM_COUNTRY_QIDS = {
    "Q430309", "Q881", "Q8733", "Q180573", "Q10841085", "Q1193879",
    "Q107368497", "Q10828323", "Q3111454", "Q10800791", "Q172640",
    "Q1034173", "Q878276", "Q1372154", "Q1317884", "Q1317991",
    "Q6500483", "Q878309", "Q10800789", "Q2623875", "Q1827039",
    "Q118441678", "Q1072362"
}

API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "WikiEntityTools/1.0 (contact: example@example.com)"
MAX_IDS_PER_REQUEST = 50

EVENT_ELIGIBLE_P31_QIDS = {
    "Q178561","Q198","Q131569","Q1691434","Q6934728","Q625298","Q3199915",
    "Q645883","Q831663","Q2001676","Q188055","Q350604","Q467011","Q45382",
    "Q25906438","Q10931","Q175331","Q273120","Q124757","Q40231","Q858439",
    "Q2618461","Q43109","Q3839081","Q7944","Q8068","Q140588","Q96096109",
    "Q15941028","Q8065","Q3002774","Q321839","Q18564543","Q7157512","Q637017",
    "Q542549","Q6056746","Q64001169","Q692412","Q864113","Q1707610","Q124734",
    "Q1261499","Q135010","Q12890393","Q1190554","Q997267"
}

def is_person_from_statements(statements_flat: Any) -> bool:
    if not isinstance(statements_flat, list):
        return False
    for st in statements_flat:
        if not isinstance(st, dict):
            continue
        if st.get("key") == "P31" and st.get("value") == "Q5":
            return True
    return False


def extract_countries_from_statements(statements_flat: Any) -> Any:
    """
    Return list of P27 values (country of citizenship) as strings.
    Values are already normalized (QIDs or strings) in statements_flat.
    """
    if not isinstance(statements_flat, list):
        return []
    countries = []
    for st in statements_flat:
        if not isinstance(st, dict):
            continue
        if st.get("key") == "P27":
            val = st.get("value")
            if isinstance(val, str):
                countries.append(val)
    # de-duplicate, keep order
    seen = set()
    uniq = []
    for v in countries:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def extract_values_for_key(statements_flat: Any, prop_id: str) -> Any:
    """
    Return list of values for a given prop_id from statements_flat.
    Values are expected to be strings (QIDs or plain strings) from prior normalization.
    """
    if not isinstance(statements_flat, list):
        return []
    values = []
    for st in statements_flat:
        if not isinstance(st, dict):
            continue
        if st.get("key") == prop_id:
            val = st.get("value")
            if isinstance(val, str):
                values.append(val)
    # de-duplicate, keep order
    seen = set()
    uniq = []
    for v in values:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def iter_categorized(jsonl_input_path: str, show_progress: bool = True) -> Iterator[str]:
    path = Path(jsonl_input_path)
    with path.open("r", encoding="utf-8") as f:
        pbar = tqdm(desc="Categorizing", unit="rec", disable=not show_progress)
        for line in f:
            s = line.strip()
            if not s:
                pbar.update(1)
                continue
            try:
                obj = json.loads(s)
            except Exception:
                pbar.update(1)
                continue
            wd = (obj or {}).get("wikidata") or {}
            statements_flat = wd.get("statements_flat") or []
            category = "person" if is_person_from_statements(statements_flat) else "event"
            obj["category"] = category
            yield json.dumps(obj, ensure_ascii=False)
            pbar.update(1)
        pbar.close()


def run_categorize(
    jsonl_input_path: str,
    jsonl_output_path: str,
    preview_jsonl_path: str,
    show_progress: bool = True
) -> Dict[str, int]:
    """Categorize entities as person or event and write to JSONL.

    Reads JSONL from previous pipeline step, classifies each entity as
    "person" or "event" based on Wikidata statements, and writes categorized
    entities to output JSONL. Also generates a preview file with sample
    entities and their categories.

    Args:
        jsonl_input_path: Path to input JSONL file with entity data.
        jsonl_output_path: Path to output JSONL file with categorized entities.
        preview_jsonl_path: Path to preview JSONL file with sample entities.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        Dictionary with summary statistics:
        - processed: Number of entities processed
        - written: Number of entities written to output
    """
    inp = Path(jsonl_input_path)
    outp = Path(jsonl_output_path)
    prev = Path(preview_jsonl_path)
    processed = 0
    written = 0
    if not inp.exists():
        return {"processed": 0, "written": 0}
    # Collect previews in-memory so we can resolve QIDs to labels before writing
    persons = []  # list of {title, category, country?}
    events = []   # list of {title, category, P31?}
    event_p31_qids = set()
    person_country_qids = set()

    with outp.open("w", encoding="utf-8") as fout:
        for rec in iter_categorized(jsonl_input_path, show_progress=show_progress):
            # Build preview line with title and category (and country for persons)
            try:
                obj = json.loads(rec)
                title = ((obj.get("wikipedia") or {}).get("title")
                         if isinstance(obj.get("wikipedia"), dict) else None)
                category = obj.get("category")
                preview = {"title": title, "category": category}
                keep = True
                if category == "person":
                    wd_obj = (obj or {}).get("wikidata") or {}
                    st_flat = wd_obj.get("statements_flat") or []
                    countries = extract_countries_from_statements(st_flat)
                    if countries:
                        preview["country"] = countries if len(countries) > 1 else countries[0]
                        for c in (countries if isinstance(preview["country"], list) else [preview["country"]]):
                            if isinstance(c, str) and c.startswith("Q"):
                                person_country_qids.add(c)
                    # Filter: only include persons with at least one Vietnam-related country QID
                    has_vn = any((c in VIETNAM_COUNTRY_QIDS) for c in countries)
                    if not countries or not has_vn:
                        keep = False
                if category == "person" and keep:
                    persons.append(preview)
                elif category == "event":
                    # For events, require P31 present and include values in preview
                    wd_obj = (obj or {}).get("wikidata") or {}
                    st_flat = wd_obj.get("statements_flat") or []
                    p31_vals = extract_values_for_key(st_flat, "P31")
                    if not p31_vals or not any((isinstance(q, str) and q in EVENT_ELIGIBLE_P31_QIDS) for q in p31_vals):
                        keep = False
                    else:
                        for q in p31_vals:
                            if isinstance(q, str) and q.startswith("Q"):
                                event_p31_qids.add(q)
                        preview["P31"] = p31_vals if len(p31_vals) > 1 else p31_vals[0]
                        events.append(preview)
                else:
                    keep = False
                if keep:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
                processed += 1
            except Exception:
                processed += 1
                continue

    # Resolve QIDs to labels (P27 for persons, P31 for events)
    qids_to_resolve = sorted((event_p31_qids | person_country_qids), key=lambda q: int(q[1:]) if q[1:].isdigit() else 10**12)
    labels = fetch_item_labels_en(qids_to_resolve) if qids_to_resolve else {}

    # Write grouped preview: persons first, then events, with labels substituted
    def _pair_code_label(v: Any) -> Dict[str, Any]:
        # Returns {"code": <original>, "label": <resolved or original>}
        if isinstance(v, str) and v.startswith("Q"):
            return {"code": v, "label": labels.get(v, v)}
        return {"code": v, "label": v}

    with prev.open("w", encoding="utf-8") as fprev:
        for p in persons:
            out = dict(p)
            if "country" in out:
                # Preserve both code and label in preview
                if isinstance(out["country"], list):
                    pairs = [_pair_code_label(x) for x in out["country"]]
                    out["country_code"] = [p["code"] for p in pairs]
                    out["country"] = [p["label"] for p in pairs]
                else:
                    pair = _pair_code_label(out["country"])
                    out["country_code"] = pair["code"]
                    out["country"] = pair["label"]
            fprev.write(json.dumps(out, ensure_ascii=False) + "\n")
        for e in events:
            out = dict(e)
            if "P31" in out:
                # Preserve both code and label in preview
                if isinstance(out["P31"], list):
                    pairs = [_pair_code_label(x) for x in out["P31"]]
                    out["P31_code"] = [p["code"] for p in pairs]
                    out["P31"] = [p["label"] for p in pairs]
                else:
                    pair = _pair_code_label(out["P31"])
                    out["P31_code"] = pair["code"]
                    out["P31"] = pair["label"]
            fprev.write(json.dumps(out, ensure_ascii=False) + "\n")

    return {"processed": processed, "written": written}


def fetch_item_labels_en(qids: list[str]) -> Dict[str, str]:
    """
    Resolve item QIDs to English labels using Wikidata API.
    Returns mapping { "Qxxx": "English label", ... }.
    """
    if not qids:
        return {}
    headers = {"User-Agent": USER_AGENT}
    result: Dict[str, str] = {}
    for i in range(0, len(qids), MAX_IDS_PER_REQUEST):
        batch = qids[i:i + MAX_IDS_PER_REQUEST]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "format": "json",
            "props": "labels",
            "languages": "en",
        }
        resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        entities = (data or {}).get("entities") or {}
        for qid, ent in entities.items():
            labels = (ent or {}).get("labels") or {}
            en = labels.get("en")
            if isinstance(en, dict):
                val = en.get("value")
                if val:
                    result[qid] = val
        time.sleep(0.1)
    return result


if __name__ == "__main__":
    # Minimal runner with project-relative paths
    from pathlib import Path as _Path

    data_raw = _Path(__file__).resolve().parent / "data_raw"

    summary = run_categorize(
        jsonl_input_path=str(data_raw / "wiki_entities_with_flat_statements.jsonl"),
        jsonl_output_path=str(data_raw / "wiki_entities_with_category.jsonl"),
        preview_jsonl_path=str(data_raw / "wiki_entities_category_preview.jsonl"),
        show_progress=True,
    )
    print(summary)


