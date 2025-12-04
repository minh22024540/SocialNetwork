#!/usr/bin/env python3
"""Stream-flatten Wikidata claims into simple statements and resolve property labels.

Outputs from run_flatten_and_resolve():
1) JSONL with original records plus wikidata.statements_flat: [{"key": "Pxxx", "value": "..."}]
2) JSON file mapping property codes to English labels: {"P31": "instance of", ...}
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple

import requests
from math import ceil
from tqdm import tqdm


API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "WikiEntityTools/1.0 (contact: example@example.com)"
MAX_IDS_PER_REQUEST = 50


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, dict):
        if "id" in value and isinstance(value["id"], str):
            return value["id"]
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def flatten_claims_string_or_entityid(claims: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for prop_id, claim_list in (claims or {}).items():
        if not isinstance(claim_list, list):
            continue
        for claim in claim_list:
            mainsnak = (claim or {}).get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            dv_type = datavalue.get("type")
            if dv_type not in {"string", "wikibase-entityid"}:
                continue
            value = datavalue.get("value")
            out.append({"key": prop_id, "value": normalize_value(value)})
    return out


def stream_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def fetch_property_labels_en(property_ids: List[str], show_progress: bool = False) -> Dict[str, str]:
    """
    Resolve property IDs to English labels using Wikidata API.
    Batches requests up to MAX_IDS_PER_REQUEST.
    """
    result: Dict[str, str] = {}
    headers = {"User-Agent": USER_AGENT}
    batches = range(0, len(property_ids), MAX_IDS_PER_REQUEST)
    pbar = tqdm(total=ceil(len(property_ids) / MAX_IDS_PER_REQUEST) if show_progress else None,
                desc="Resolving properties",
                unit="batch",
                disable=not show_progress)
    for i in batches:
        batch = property_ids[i:i + MAX_IDS_PER_REQUEST]
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
        for pid, ent in entities.items():
            labels = (ent or {}).get("labels") or {}
            en = labels.get("en") or {}
            label = en.get("value") if isinstance(en, dict) else None
            if label:
                result[pid] = label
        # gentle throttle
        time.sleep(0.1)
        pbar.update(1)
    pbar.close()
    return result


def run_flatten_and_resolve(
    input_path: str,
    out_jsonl_path: str,
    out_props_path: str,
    limit: int = 0,
    show_progress: bool = True
) -> Dict[str, int]:
    """Perform end-to-end flattening and property resolution.

    Processes Wikidata claims by:
    1. Flattening claims to statements_flat format (filtered to string/entityid)
    2. Resolving property IDs to English labels via Wikidata API
    3. Writing updated JSONL with flattened statements
    4. Writing property ID -> label mapping to JSON

    Args:
        input_path: Path to input JSONL file with entity data.
        out_jsonl_path: Path to output JSONL file with flattened statements.
        out_props_path: Path to output JSON file with property labels.
        limit: Maximum number of records to process (0 = no limit).
            Defaults to 0.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        Dictionary with summary statistics:
        - processed: Number of records processed
        - written: Number of records written
        - props: Number of unique properties found
        - labeled: Number of properties with resolved labels
    """
    in_path = Path(input_path)
    out_jsonl = Path(out_jsonl_path)
    out_props = Path(out_props_path)

    if not in_path.exists():
        return {"processed": 0, "written": 0, "props": 0, "labeled": 0}

    seen_props: Set[str] = set()
    processed = 0
    written = 0

    with in_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        pbar = tqdm(desc="Processing records", unit="rec", disable=not show_progress)
        for line in fin:
            if limit and processed >= limit:
                break
            s = line.strip()
            if not s:
                processed += 1
                pbar.update(1)
                continue
            try:
                obj = json.loads(s)
            except Exception:
                processed += 1
                pbar.update(1)
                continue

            wd = (obj or {}).get("wikidata") or {}
            claims = wd.get("claims") or {}
            if isinstance(claims, dict):
                flat = flatten_claims_string_or_entityid(claims)
                for st in flat:
                    pid = st.get("key")
                    if isinstance(pid, str) and pid.startswith("P"):
                        seen_props.add(pid)
                wd["statements_flat"] = flat
                obj["wikidata"] = wd

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            processed += 1
            pbar.update(1)
        pbar.close()

    prop_list = sorted(seen_props)
    labels_map = fetch_property_labels_en(prop_list, show_progress=show_progress) if prop_list else {}
    # Sort by property numeric code (PXXXX) ascending
    def _pid_num(pid: str) -> int:
        try:
            return int(pid[1:])
        except Exception:
            return 10**12  # push malformed ids to end
    sorted_items = sorted(labels_map.items(), key=lambda kv: _pid_num(kv[0]))
    labels_sorted = {k: v for k, v in sorted_items}
    out_props.write_text(json.dumps(labels_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "processed": processed,
        "written": written,
        "props": len(prop_list),
        "labeled": len(labels_sorted),
    }


if __name__ == "__main__":
    # Minimal, non-interactive example runner, using project-relative paths.
    from pathlib import Path as _Path

    project_root = _Path(__file__).resolve().parents[2]
    data_raw = _Path(__file__).resolve().parent / "data_raw"
    data_analysis = _Path(__file__).resolve().parent / "data_analysis"

    summary = run_flatten_and_resolve(
        input_path=data_raw / "wiki_entities_full_from_xml.jsonl",
        out_jsonl_path=data_raw / "wiki_entities_with_flat_statements.jsonl",
        out_props_path=data_analysis / "properties_en_labels.json",
        limit=0,
        show_progress=True,
    )
    print(summary)


