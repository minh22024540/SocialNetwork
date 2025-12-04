#!/usr/bin/env python3
"""
Code 8: Build page links for our JSONL dataset using pagelinks SQL dump.

Steps:
1) Read pagelinks SQL(.sql or .sql.gz) and stream-extract tuples of (pl_from, pl_namespace, pl_title)
2) Stream the JSONL (from code 7 full output), build:
   - title_to_id: map of normalized title -> page id
   - id_set: set of our page ids
3) While parsing pagelinks, for each source pl_from in our id_set and pl_namespace==0, if pl_title matches a title in our dataset, record a target page id link.
4) Stream the JSONL again to write an output JSONL: keep four fields only:
   { wikipedia, wikidata, category, links: [target_ids_in_our_dataset] }
"""

import json
import re
import gzip
from pathlib import Path
from typing import Dict, Set, List, Iterable, Tuple
import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from urllib.parse import unquote
import requests


def open_text(path: Path):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
    return path.open('r', encoding='utf-8', errors='ignore')


def normalize_title(title: str) -> str:
    # MediaWiki title normalization: spaces to underscores, capitalize first letter
    if title is None:
        return ''
    t = title.replace(' ', '_')
    return t[:1].upper() + t[1:] if t else ''


def build_title_and_id_sets(jsonl_path: Path) -> Tuple[Dict[str, int], Set[int]]:
    title_to_id: Dict[str, int] = {}
    id_set: Set[int] = set()
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Indexing JSONL', unit='rec'):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            wp = (obj or {}).get('wikipedia') or {}
            try:
                pid = int(wp.get('id'))
            except Exception:
                continue
            title = wp.get('title') or ''
            title_norm = normalize_title(title)
            if title_norm:
                title_to_id.setdefault(title_norm, pid)
            id_set.add(pid)
    return title_to_id, id_set


def augment_title_map_with_wikidata(jsonl_path: Path, title_to_id: Dict[str, int]) -> None:
    """
    Enrich title_to_id with Wikidata labels and aliases to better resolve links.
    Preference order: Vietnamese ('vi') labels/aliases first; then English as fallback.
    """
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Augmenting titles from Wikidata', unit='rec'):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            wp = (obj or {}).get('wikipedia') or {}
            try:
                pid = int(wp.get('id'))
            except Exception:
                continue
            wd = (obj or {}).get('wikidata') or {}
            if not isinstance(wd, dict):
                continue
            # Labels
            labels = wd.get('labels') or {}
            if isinstance(labels, dict):
                for lang in ('vi', 'en'):
                    val = labels.get(lang)
                    if isinstance(val, str) and val:
                        norm = normalize_title(val)
                        if norm:
                            title_to_id.setdefault(norm, pid)
            # Aliases
            aliases = wd.get('aliases') or {}
            if isinstance(aliases, dict):
                for lang in ('vi', 'en'):
                    arr = aliases.get(lang)
                    if isinstance(arr, list):
                        for alias in arr:
                            if isinstance(alias, str) and alias:
                                norm = normalize_title(alias)
                                if norm:
                                    title_to_id.setdefault(norm, pid)


def parse_redirect_sql(redirect_sql_path: Path) -> Dict[str, str]:
    """
    Parse redirect.sql(.gz) to build mapping: normalized_source_title -> normalized_target_title (namespace 0 only).
    Tuple format typically: (rd_from, rd_namespace, 'rd_title', ...)
    Requires a page_id -> title map to convert rd_from -> source title. This function only parses tuples;
    the join is performed in build_redirect_map_with_page.
    """
    rows = []
    if not redirect_sql_path.exists():
        return {}
    tuple_re = re.compile(r"\(([^\)]*)\)")
    field_re = re.compile(r"(?:'([^']*)'|([^,]*))(?:,|$)")
    with open_text(redirect_sql_path) as f:
        for line in tqdm(f, desc='Reading redirect SQL', unit='line'):
            if 'INSERT INTO' not in line:
                continue
            for tup in tuple_re.findall(line):
                fields = []
                idx = 0
                text = tup
                for m in field_re.finditer(text):
                    val = m.group(1) if m.group(1) is not None else m.group(2)
                    fields.append(val)
                    idx += 1
                    if idx >= 3:
                        break
                if len(fields) < 3:
                    continue
                try:
                    rd_from = int(fields[0])
                except Exception:
                    continue
                try:
                    rd_ns = int(fields[1])
                except Exception:
                    rd_ns = 0
                rd_title = fields[2] or ''
                rows.append((rd_from, rd_ns, rd_title))
    # Return raw; the join is outside
    return {str(i): t for (i, _, t) in rows if _ == 0}


def build_redirect_map_with_page(redirect_sql_path: Path, page_map: Dict[Tuple[int, str], int]) -> Dict[str, str]:
    """
    Build redirect mapping source_title_norm -> target_title_norm for namespace 0 by joining redirect rows with page.sql.
    """
    # Build id -> title_norm from page_map
    id_to_title_norm: Dict[int, str] = {}
    for (ns, title_norm), pid in page_map.items():
        if ns == 0:
            id_to_title_norm[pid] = title_norm
    # Parse redirect rows (rd_from, rd_ns, rd_title)
    tuple_re = re.compile(r"\(([^\)]*)\)")
    field_re = re.compile(r"(?:'([^']*)'|([^,]*))(?:,|$)")
    redir: Dict[str, str] = {}
    with open_text(redirect_sql_path) as f:
        for line in tqdm(f, desc='Reading redirect SQL', unit='line'):
            if 'INSERT INTO' not in line:
                continue
            for tup in tuple_re.findall(line):
                fields = []
                idx = 0
                text = tup
                for m in field_re.finditer(text):
                    val = m.group(1) if m.group(1) is not None else m.group(2)
                    fields.append(val)
                    idx += 1
                    if idx >= 3:
                        break
                if len(fields) < 3:
                    continue
                try:
                    rd_from = int(fields[0])
                except Exception:
                    continue
                try:
                    rd_ns = int(fields[1])
                except Exception:
                    rd_ns = 0
                if rd_ns != 0:
                    continue
                rd_title = fields[2] or ''
                target_norm = normalize_title(rd_title)
                src_norm = id_to_title_norm.get(rd_from)
                if src_norm and target_norm:
                    redir[src_norm] = target_norm
    return redir


def build_id_to_category(jsonl_path: Path) -> Dict[int, str]:
    id_to_category: Dict[int, str] = {}
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            wp = (obj or {}).get('wikipedia') or {}
            try:
                pid = int(wp.get('id'))
            except Exception:
                continue
            cat = (obj or {}).get('category')
            if isinstance(cat, str):
                id_to_category[pid] = cat
    return id_to_category


def build_id_to_title(jsonl_path: Path) -> Dict[int, str]:
    id_to_title: Dict[int, str] = {}
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            wp = (obj or {}).get('wikipedia') or {}
            try:
                pid = int(wp.get('id'))
            except Exception:
                continue
            title = wp.get('title')
            if isinstance(title, str):
                id_to_title[pid] = title
    return id_to_title


def parse_linktarget_sql(linktarget_sql: Path) -> Dict[int, Tuple[int, str]]:
    """
    Parse linktarget.sql(.gz) to build mapping: lt_id -> (lt_namespace, lt_title)
    """
    id_to_target: Dict[int, Tuple[int, str]] = {}
    tuple_re = re.compile(r"\(([^\)]*)\)")
    field_re = re.compile(r"(?:'([^']*)'|([^,]*))(?:,|$)")
    with open_text(linktarget_sql) as f:
        for line in tqdm(f, desc='Reading linktarget SQL', unit='line'):
            if 'INSERT INTO' not in line:
                continue
            # Expect tuples of (lt_id, lt_namespace, lt_title)
            for tup in tuple_re.findall(line):
                fields = []
                idx = 0
                text = tup
                for m in field_re.finditer(text):
                    val = m.group(1) if m.group(1) is not None else m.group(2)
                    fields.append(val)
                    idx += 1
                    if idx >= 3:
                        break
                if len(fields) < 3:
                    continue
                try:
                    lt_id = int(fields[0])
                except Exception:
                    continue
                try:
                    lt_ns = int(fields[1])
                except Exception:
                    lt_ns = 0
                lt_title = fields[2] or ''
                id_to_target[lt_id] = (lt_ns, lt_title)
    return id_to_target


def parse_page_sql(page_sql: Path) -> Dict[Tuple[int, str], int]:
    """
    Parse page.sql(.gz) to build mapping: (page_namespace, normalized_title) -> page_id
    """
    ns_title_to_id: Dict[Tuple[int, str], int] = {}
    tuple_re = re.compile(r"\(([^\)]*)\)")
    field_re = re.compile(r"(?:'([^']*)'|([^,]*))(?:,|$)")
    with open_text(page_sql) as f:
        for line in tqdm(f, desc='Reading page SQL', unit='line'):
            if 'INSERT INTO' not in line:
                continue
            for tup in tuple_re.findall(line):
                fields = []
                idx = 0
                text = tup
                for m in field_re.finditer(text):
                    val = m.group(1) if m.group(1) is not None else m.group(2)
                    fields.append(val)
                    idx += 1
                    if idx >= 3:
                        break
                if len(fields) < 3:
                    continue
                try:
                    page_id = int(fields[0])
                    page_ns = int(fields[1])
                except Exception:
                    continue
                page_title = fields[2] or ''
                ns_title_to_id[(page_ns, normalize_title(page_title))] = page_id
    return ns_title_to_id


def build_linktarget_to_page_id(
    linktarget_map: Dict[int, Tuple[int, str]] | None,
    page_map: Dict[Tuple[int, str], int] | None,
    id_set: Set[int],
) -> Dict[int, int]:
    """
    Precompute fast mapping: pl_target_id -> page_id (only namespace 0 and only targets present in our dataset).
    This avoids repeated title lookups during pagelinks scanning.
    """
    if not linktarget_map or not page_map:
        return {}
    out: Dict[int, int] = {}
    for lt_id, (ns, title) in linktarget_map.items():
        if ns != 0:
            continue
        page_id = page_map.get((0, normalize_title(title)))
        if page_id and page_id in id_set:
            out[lt_id] = page_id
    return out


def parse_pagelinks_sql(sql_path: Path, title_to_id: Dict[str, int], id_set: Set[int], linktarget_map: Dict[int, Tuple[int, str]] | None = None, page_map: Dict[Tuple[int, str], int] | None = None) -> Dict[int, List[int]]:
    """
    Return mapping: source_page_id -> sorted unique list of target ids that are in our dataset.
    Only namespace 0 targets considered.
    """
    src_to_targets: Dict[int, Set[int]] = {}

    # Regex to match INSERT lines and extract tuples
    # Typical legacy tuple: (pl_from, pl_namespace, 'pl_title', ...)
    # Newer schema: (pl_from, pl_target_id, pl_from_namespace)
    # We'll use a forgiving parser for (..),(..) sequences.
    # Fast-path regexes for common dump formats
    # Legacy schema: (pl_from,pl_namespace,'pl_title', ...)
    legacy_pat = re.compile(r"\((\d+),(\d+),'([^']*)'", re.ASCII)
    # New schema: strictly three columns per tuple: (pl_from,pl_target_id,pl_from_namespace)
    # Use a strict pattern to avoid catastrophic matching on long lines
    new_pat = re.compile(r"\((\d+),(\d+),(\d+)\)")

    # Precompute fast target resolver if possible
    lt_to_page: Dict[int, int] = {}
    if linktarget_map and page_map:
        lt_to_page = build_linktarget_to_page_id(linktarget_map, page_map, id_set)

    # Debug counters
    c_legacy_tuples = c_new_tuples = 0
    c_from_in_dataset = c_target_ns0 = c_target_resolved = c_links_added = 0
    c_resolved_fast = c_resolved_slow = 0
    ns_counts: Dict[int, int] = {}
    samples_printed = 0
    c_target_id_zero = 0

    with open_text(sql_path) as f:
        for line in tqdm(f, desc='Reading pagelinks SQL', unit='line'):
            if 'INSERT INTO' not in line:
                continue
            # Legacy matches
            for m in legacy_pat.finditer(line):
                c_legacy_tuples += 1
                try:
                    pl_from = int(m.group(1))
                    pl_namespace = int(m.group(2))
                except Exception:
                    continue
                if pl_from not in id_set or pl_namespace != 0:
                    continue
                c_from_in_dataset += 1
                pl_title = m.group(3) or ''
                target_title_norm = normalize_title(pl_title)
                if not target_title_norm:
                    continue
                if page_map is not None:
                    target_id = page_map.get((0, target_title_norm))
                else:
                    target_id = title_to_id.get(target_title_norm)
                if target_id is None:
                    continue
                c_target_resolved += 1
                src_to_targets.setdefault(pl_from, set()).add(target_id)
                c_links_added += 1

            # New schema matches (only if linktarget map provided)
            if linktarget_map:
                for m in new_pat.finditer(line):
                    c_new_tuples += 1
                    try:
                        pl_from = int(m.group(1))
                        pl_target_id = int(m.group(2))
                    except Exception:
                        continue
                    if pl_from not in id_set:
                        continue
                    c_from_in_dataset += 1
                    if pl_target_id == 0:
                        c_target_id_zero += 1
                        # Nothing to resolve if target id is 0
                        continue
                    # Fast path via precomputed map
                    target_id = lt_to_page.get(pl_target_id)
                    if target_id:
                        c_target_ns0 += 1
                        c_resolved_fast += 1
                    else:
                        # Fallback resolve on the fly
                        target = linktarget_map.get(pl_target_id)
                        if not target:
                            continue
                        target_ns, target_title = target
                        # Track namespace distribution
                        ns_counts[target_ns] = ns_counts.get(target_ns, 0) + 1
                        if samples_printed < 10 and pl_from in id_set:
                            print(f"SAMPLE pl_from={pl_from} target_id={pl_target_id} ns={target_ns} title={str(target_title)[:80]}")
                            samples_printed += 1
                        if target_ns != 0:
                            continue
                        c_target_ns0 += 1
                        target_title_norm = normalize_title(target_title)
                        if page_map is not None:
                            target_id = page_map.get((0, target_title_norm))
                        else:
                            target_id = title_to_id.get(target_title_norm)
                        if target_id:
                            c_resolved_slow += 1
                    if target_id is None:
                        continue
                    c_target_resolved += 1
                    src_to_targets.setdefault(pl_from, set()).add(target_id)
                    c_links_added += 1

    # Convert sets to sorted lists
    result = {src: sorted(list(tgts)) for src, tgts in src_to_targets.items()}
    # Print debug summary
    print(
        json.dumps(
            {
                "linktarget_size": len(linktarget_map) if linktarget_map else 0,
                "page_map_size": len(page_map) if page_map else 0,
                "lt_to_page_size": len(lt_to_page) if lt_to_page else 0,
                "legacy_tuples": c_legacy_tuples,
                "new_tuples": c_new_tuples,
                "from_in_dataset": c_from_in_dataset,
                "target_ns0": c_target_ns0,
                "target_resolved": c_target_resolved,
                "resolved_fast": c_resolved_fast,
                "resolved_slow": c_resolved_slow,
                "links_added": c_links_added,
                "sources_with_links": len(result),
                "target_ns_dist": {str(k): v for k, v in sorted(ns_counts.items(), key=lambda x: -x[1])[:10]},
                "pl_target_id_zero": c_target_id_zero,
            },
            ensure_ascii=False,
        )
    )
    return result


def extract_links_from_wikitext(text: str) -> List[str]:
    """
    Very simple extractor for internal links of the form [[Title]] or [[Title|...]].
    Ignores section links (#...) and files/categories via namespace prefixes.
    """
    if not text:
        return []
    links: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        start = text.find('[[', i)
        if start == -1:
            break
        end = text.find(']]', start + 2)
        if end == -1:
            break
        content = text[start + 2:end]
        i = end + 2
        if not content:
            continue
        # Split on pipe
        pipe = content.find('|')
        if pipe != -1:
            target = content[:pipe]
        else:
            target = content
        # Drop section anchors
        hash_pos = target.find('#')
        if hash_pos != -1:
            target = target[:hash_pos]
        target = target.strip()
        if not target:
            continue
        # Ignore interwiki and non-main namespaces by prefix (heuristic)
        # We only keep plain titles without a colon prefix to target main ns
        if ':' in target:
            # Keep only if it's a leading colon to force main namespace like [[:Title]]
            if not target.startswith(':'):
                continue
            target = target.lstrip(':').strip()
            if not target:
                continue
        # MediaWiki uses underscores in DB; normalize spaces to underscores here
        target = target.replace(' ', '_')
        links.append(target)
    return links


def extract_links_from_html(text: str) -> List[str]:
    """
    Extract /wiki/Title links from HTML-like content. Decodes percent-encoding and strips anchors/query.
    Keeps only main namespace-like targets (no colon prefix).
    """
    if not text:
        return []
    links: List[str] = []
    # Rough regex for hrefs; we avoid heavy HTML parsing for speed
    for m in re.finditer(r'href=\"/wiki/([^\"#?]+)', text):
        raw = m.group(1)
        # Decode percent-encoding
        title = unquote(raw)
        if not title:
            continue
        # Ignore non-main namespaces (have colon)
        if ':' in title and not title.startswith(':'):
            continue
        title = title.lstrip(':').strip()
        if not title:
            continue
        # Normalize spaces to underscores if any remain
        title = title.replace(' ', '_')
        links.append(title)
    return links


def extract_links_from_text(text: str) -> List[str]:
    """Unified extractor that handles both wikitext and HTML formats."""
    if not isinstance(text, str) or not text:
        return []
    out: List[str] = []
    # Prefer wikitext markers if present
    if '[[' in text and ']]' in text:
        out.extend(extract_links_from_wikitext(text))
    # Also scan for HTML anchors; some sources may store rendered HTML
    if '/wiki/' in text:
        out.extend(extract_links_from_html(text))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def parse_pages_articles_links(xml_bz2_path: Path, id_set: Set[int], title_to_id: Dict[str, int]) -> Dict[int, List[int]]:
    """
    Fallback path: stream the pages-articles multistream XML and extract internal links by title.
    Only collect links for pages whose page_id is in id_set and namespace == 0.
    """
    src_to_targets: Dict[int, Set[int]] = {}
    if not xml_bz2_path.exists():
        return {}
    # Iterparse to keep memory low
    with bz2.open(xml_bz2_path, 'rb') as f:
        context = ET.iterparse(f, events=("end",))
        # The XML uses tags in the MediaWiki namespace; strip all namespaces for simplicity
        def strip_ns(tag: str) -> str:
            if '}' in tag:
                return tag.split('}', 1)[1]
            return tag
        current_page_id = None
        current_ns = None
        current_title = None
        current_text = None
        for event, elem in context:
            tag = strip_ns(elem.tag)
            if tag == 'page':
                # End of a page: if relevant, extract links
                try:
                    if (current_page_id is not None and current_ns == 0 and current_page_id in id_set):
                        targets = extract_links_from_wikitext(current_text or '')
                        if targets:
                            for t in targets:
                                norm = normalize_title(t)
                                if not norm:
                                    continue
                                tgt_id = title_to_id.get(norm)
                                if tgt_id is None or tgt_id not in id_set:
                                    continue
                                src_to_targets.setdefault(current_page_id, set()).add(tgt_id)
                finally:
                    # Reset and free memory
                    elem.clear()
                    current_page_id = None
                    current_ns = None
                    current_title = None
                    current_text = None
            elif tag == 'id' and elem.getparent() is None:
                # xml.etree doesn't support getparent; alternative approach below
                pass
            elif tag == 'title':
                current_title = (elem.text or '')
                elem.clear()
            elif tag == 'ns':
                try:
                    current_ns = int(elem.text or '0')
                except Exception:
                    current_ns = 0
                elem.clear()
            elif tag == 'id' and current_page_id is None:
                # The first <id> under <page> is the page id
                try:
                    current_page_id = int(elem.text or '0')
                except Exception:
                    current_page_id = None
                elem.clear()
            elif tag == 'text':
                current_text = elem.text or ''
                elem.clear()
    # Convert sets to lists
    return {src: sorted(list(tgts)) for src, tgts in src_to_targets.items()}


def build_links_from_jsonl_text(jsonl_in_path: Path, id_set: Set[int], title_to_id: Dict[str, int], id_to_category: Dict[int, str], redirect_map: Dict[str, str] | None = None) -> Dict[int, List[int]]:
    """
    Fastest path: use already-available wikitext inside each JSONL object's wikipedia entity.
    We look for common fields that may contain full article text: 'text', 'content', 'wikitext', 'full_text'.
    Only keeps links to namespace 0 targets that exist in our dataset.
    Additionally filters links so that:
      - person pages only keep links to event pages
      - event pages only keep links to person pages
    Resolves redirects when possible using redirect_map (source_title -> target_title).
    """
    src_to_targets: Dict[int, Set[int]] = {}
    candidates = ("text", "content", "wikitext", "full_text")
    with jsonl_in_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning JSONL text for links", unit="rec"):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            wp = (obj or {}).get("wikipedia") or {}
            page_id = wp.get("id")
            if not isinstance(page_id, int) or page_id not in id_set:
                continue
            src_cat = id_to_category.get(page_id)
            text_val = None
            for key in candidates:
                val = wp.get(key)
                if isinstance(val, str) and val:
                    text_val = val
                    break
            if not text_val:
                continue
            targets = extract_links_from_text(text_val)
            if not targets:
                continue
            for t in targets:
                norm = normalize_title(t)
                if not norm:
                    continue
                # If not found, try redirect resolution (source title -> target title)
                if norm not in title_to_id and redirect_map:
                    tgt_title = redirect_map.get(norm)
                    if tgt_title:
                        norm = tgt_title
                tgt_id = title_to_id.get(norm)
                if tgt_id is None or tgt_id not in id_set:
                    continue
                # Category-based filtering
                tgt_cat = id_to_category.get(tgt_id)
                if src_cat == 'person' and tgt_cat != 'event':
                    continue
                if src_cat == 'event' and tgt_cat != 'person':
                    continue
                src_to_targets.setdefault(page_id, set()).add(tgt_id)
    return {src: sorted(list(tgts)) for src, tgts in src_to_targets.items()}


def _extract_best_time_from_claim_list(claim_list: list) -> str | None:
    if not isinstance(claim_list, list):
        return None
    # Prefer preferred rank, then normal
    def _rank_order(c):
        r = (c or {}).get('rank')
        if r == 'preferred':
            return 0
        if r == 'normal':
            return 1
        return 2
    for claim in sorted(claim_list, key=_rank_order):
        mainsnak = (claim or {}).get('mainsnak') or {}
        if mainsnak.get('snaktype') != 'value':
            continue
        datavalue = (mainsnak or {}).get('datavalue') or {}
        value = datavalue.get('value')
        if not isinstance(value, dict):
            continue
        time_str = value.get('time')
        if isinstance(time_str, str) and len(time_str) >= 5:
            # Normalize: strip leading '+', keep date part before 'T'
            if time_str.startswith('+'):
                time_str = time_str[1:]
            if 'T' in time_str:
                time_str = time_str.split('T', 1)[0]
            return time_str
    return None


def extract_temporal_fields(wikidata_obj: dict) -> dict:
    """
    From full wikidata.claims, extract DOB (P569), DOD (P570), start (P580), end (P582), point in time (P585).
    Returns dict with any found keys.
    """
    out: dict = {}
    if not isinstance(wikidata_obj, dict):
        return out
    claims = wikidata_obj.get('claims') or {}
    if not isinstance(claims, dict):
        return out
    # Persons
    dob = _extract_best_time_from_claim_list(claims.get('P569'))
    dod = _extract_best_time_from_claim_list(claims.get('P570'))
    if dob:
        out['date_of_birth'] = dob
    if dod:
        out['date_of_death'] = dod
    # Events
    start = _extract_best_time_from_claim_list(claims.get('P580'))
    end = _extract_best_time_from_claim_list(claims.get('P582'))
    pit = _extract_best_time_from_claim_list(claims.get('P585'))
    if start:
        out['start_time'] = start
    if end:
        out['end_time'] = end
    if pit:
        out['point_in_time'] = pit
    return out


def extract_entity_ids_from_claims(claims: dict, prop_id: str) -> list[str]:
    """
    From claims, collect unique wikibase-entityid values' id for the given property.
    """
    res: list[str] = []
    if not isinstance(claims, dict):
        return res
    lst = claims.get(prop_id)
    if not isinstance(lst, list):
        return res
    seen = set()
    for claim in lst:
        mainsnak = (claim or {}).get('mainsnak') or {}
        if mainsnak.get('snaktype') != 'value':
            continue
        dv = (mainsnak or {}).get('datavalue') or {}
        val = dv.get('value')
        if isinstance(val, dict):
            qid = val.get('id')
            if isinstance(qid, str) and qid and qid not in seen:
                seen.add(qid)
                res.append(qid)
    return res


def extract_coordinates_from_claims(claims: dict) -> tuple[float | None, float | None]:
    """
    Extract first coordinate location (P625) as (lat, lon) if present.
    """
    if not isinstance(claims, dict):
        return None, None
    lst = claims.get('P625')
    if not isinstance(lst, list):
        return None, None
    for claim in lst:
        mainsnak = (claim or {}).get('mainsnak') or {}
        if mainsnak.get('snaktype') != 'value':
            continue
        dv = (mainsnak or {}).get('datavalue') or {}
        val = dv.get('value')
        if isinstance(val, dict):
            lat = val.get('latitude')
            lon = val.get('longitude')
            try:
                latf = float(lat)
                lonf = float(lon)
                return latf, lonf
            except Exception:
                continue
    return None, None


def collect_additional_qids(jsonl_in: Path) -> set[str]:
    """
    Scan input to collect QIDs for additional properties for label resolution:
      - P106 occupation (persons)
      - P31 instance of (all)
      - P279 subclass of (all)
      - P276 location (all)
    Returns set of QIDs.
    """
    props = ('P106', 'P31', 'P279', 'P276')
    to_resolve: set[str] = set()
    with jsonl_in.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            wd = (obj or {}).get('wikidata') or {}
            claims = (wd or {}).get('claims') or {}
            if not isinstance(claims, dict):
                continue
            for pid in props:
                for q in extract_entity_ids_from_claims(claims, pid):
                    to_resolve.add(q)
    return to_resolve

def collect_person_qids_for_resolution(jsonl_in: Path) -> set[str]:
    """
    Scan input JSONL and collect QIDs to resolve for person fields:
    - P27 country of citizenship (list)
    - P21 gender (single)
    Returns a set of QIDs.
    """
    to_resolve: set[str] = set()
    with jsonl_in.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if (obj or {}).get('category') != 'person':
                continue
            wd = (obj or {}).get('wikidata') or {}
            claims = (wd or {}).get('claims') or {}
            if not isinstance(claims, dict):
                continue
            for q in extract_entity_ids_from_claims(claims, 'P27'):
                to_resolve.add(q)
            for q in extract_entity_ids_from_claims(claims, 'P21'):
                to_resolve.add(q)
    return to_resolve


def resolve_item_labels_en(qids: list[str]) -> dict[str, str]:
    """
    Resolve item QIDs to labels using Wikidata API.
    Preference: English; fallback to Vietnamese when English missing.
    Includes retries with exponential backoff to handle transient errors/rate limits.
    """
    if not qids:
        return {}
    API_URL = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": "WikiEntityTools/1.0 (contact: example@example.com)"}
    out: dict[str, str] = {}
    BATCH = 50
    MAX_RETRIES = 5
    for i in range(0, len(qids), BATCH):
        batch = qids[i:i + BATCH]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "format": "json",
            "props": "labels",
            # Ask for both English and Vietnamese; prefer English later
            "languages": "en|vi",
        }
        attempt = 0
        while True:
            try:
                resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    data = {}
                    break
                # exponential backoff: 0.5,1,2,4,... seconds
                import time
                time.sleep(0.5 * (2 ** (attempt - 1)))
        ents = (data or {}).get("entities") or {}
        for qid, ent in ents.items():
            lbls = (ent or {}).get("labels") or {}
            # Prefer English label
            label = None
            en = lbls.get("en")
            if isinstance(en, dict):
                label = en.get("value") or None
            if not label:
                vi = lbls.get("vi")
                if isinstance(vi, dict):
                    label = vi.get("value") or None
            if label:
                out[qid] = label
    return out


def write_augmented_jsonl(jsonl_in: Path, jsonl_out: Path, src_to_targets: Dict[int, List[int]], preview_out: Path | None = None, id_to_title: Dict[int, str] | None = None, neo4j_out: Path | None = None) -> None:
    prev_f = None
    neo4j_f = None
    if preview_out is not None:
        prev_f = preview_out.open('w', encoding='utf-8')
    if neo4j_out is not None:
        neo4j_f = neo4j_out.open('w', encoding='utf-8')
    with jsonl_in.open('r', encoding='utf-8') as fin, jsonl_out.open('w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc='Writing output', unit='rec'):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            wp = (obj or {}).get('wikipedia') or {}
            try:
                pid = int(wp.get('id'))
            except Exception:
                continue
            wd = (obj or {}).get('wikidata') or {}
            qid = wd.get('id') if isinstance(wd, dict) else None
            time_fields = extract_temporal_fields(wd if isinstance(wd, dict) else {})
            # Construct minimal output with exactly 4 fields plus time/person fields and extras
            out_obj = {
                'wikipedia': wp,
                'wikidata': wd,
                'category': (obj or {}).get('category'),
                'links': src_to_targets.get(pid, []),
            }
            if time_fields:
                out_obj.update(time_fields)
            # Person fields (resolved outside via label map injected later)
            if (obj or {}).get('category') == 'person':
                claims = (wd or {}).get('claims') or {}
                if isinstance(claims, dict):
                    countries = extract_entity_ids_from_claims(claims, 'P27')
                    genders = extract_entity_ids_from_claims(claims, 'P21')
                    occupations = extract_entity_ids_from_claims(claims, 'P106')
                    if countries:
                        out_obj['country_codes'] = countries
                    if genders:
                        out_obj['gender_code'] = genders[0]
                    if occupations:
                        out_obj['occupation_codes'] = occupations
            # Generic fields for all
            claims_all = (wd or {}).get('claims') or {}
            if isinstance(claims_all, dict):
                instance_of = extract_entity_ids_from_claims(claims_all, 'P31')
                subclass_of = extract_entity_ids_from_claims(claims_all, 'P279')
                location_codes = extract_entity_ids_from_claims(claims_all, 'P276')
                lat, lon = extract_coordinates_from_claims(claims_all)
                if instance_of:
                    out_obj['instance_of_codes'] = instance_of
                if subclass_of:
                    out_obj['subclass_of_codes'] = subclass_of
                if location_codes:
                    out_obj['location_codes'] = location_codes
                if lat is not None and lon is not None:
                    out_obj['latitude'] = lat
                    out_obj['longitude'] = lon
            fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
            if prev_f is not None:
                links_list = out_obj['links']
                titles_list = [id_to_title.get(t, None) for t in links_list] if id_to_title else None
                prev_obj = {
                    'wikipedia_id': pid,
                    'wikipedia_title': wp.get('title'),
                    'wikidata_id': qid,
                    'links': links_list,
                }
                if titles_list is not None:
                    prev_obj['links_titles'] = titles_list
                prev_f.write(json.dumps(prev_obj, ensure_ascii=False) + '\n')
            if neo4j_f is not None:
                links_list = out_obj['links']
                titles_list = [id_to_title.get(t, None) for t in links_list] if id_to_title else None
                neo_obj = {
                    'id': pid,
                    'title': wp.get('title'),
                    'category': (obj or {}).get('category'),
                    'wikidata_id': qid,
                    'wikipedia': wp,   # full wikipedia object (may include full_text)
                    'wikidata': wd,    # full wikidata entity
                    'links': links_list,
                    'links_titles': titles_list,
                }
                if time_fields:
                    neo_obj.update(time_fields)
                if (obj or {}).get('category') == 'person':
                    claims = (wd or {}).get('claims') or {}
                    if isinstance(claims, dict):
                        countries = extract_entity_ids_from_claims(claims, 'P27')
                        genders = extract_entity_ids_from_claims(claims, 'P21')
                        occupations = extract_entity_ids_from_claims(claims, 'P106')
                        if countries:
                            neo_obj['country_codes'] = countries
                        if genders:
                            neo_obj['gender_code'] = genders[0]
                        if occupations:
                            neo_obj['occupation_codes'] = occupations
                # Generic for all
                claims_all = (wd or {}).get('claims') or {}
                if isinstance(claims_all, dict):
                    instance_of = extract_entity_ids_from_claims(claims_all, 'P31')
                    subclass_of = extract_entity_ids_from_claims(claims_all, 'P279')
                    location_codes = extract_entity_ids_from_claims(claims_all, 'P276')
                    lat, lon = extract_coordinates_from_claims(claims_all)
                    if instance_of:
                        neo_obj['instance_of_codes'] = instance_of
                    if subclass_of:
                        neo_obj['subclass_of_codes'] = subclass_of
                    if location_codes:
                        neo_obj['location_codes'] = location_codes
                    if lat is not None and lon is not None:
                        neo_obj['latitude'] = lat
                        neo_obj['longitude'] = lon
                neo4j_f.write(json.dumps(neo_obj, ensure_ascii=False) + '\n')
    if prev_f is not None:
        prev_f.close()
    if neo4j_f is not None:
        neo4j_f.close()


def write_graph_outputs(jsonl_in: Path,
                        src_to_targets: Dict[int, List[int]],
                        id_to_title: Dict[int, str],
                        id_to_category: Dict[int, str],
                        nodes_out: Path,
                        edges_out: Path) -> None:
    # Nodes TSV: id\tcategory\ttitle
    with nodes_out.open('w', encoding='utf-8') as fn:
        fn.write('id\tcategory\ttitle\n')
        for pid, cat in id_to_category.items():
            title = id_to_title.get(pid, '')
            fn.write(f"{pid}\t{cat}\t{title}\n")
    # Edges TSV: source\ttarget (already filtered cross-category)
    with edges_out.open('w', encoding='utf-8') as fe:
        fe.write('source\ttarget\n')
        for src, tgts in src_to_targets.items():
            for tgt in tgts:
                fe.write(f"{src}\t{tgt}\n")


def write_bulk_import_csv(jsonl_in: Path,
                          src_to_targets: Dict[int, List[int]],
                          id_to_title: Dict[int, str],
                          id_to_category: Dict[int, str],
                          nodes_csv: Path,
                          edges_csv: Path) -> None:
    """
    Write Neo4j bulk-import CSVs that require no mapping:
      - Nodes: id:ID,title,category,wikidata_id,wikipedia:STRING,wikidata:STRING,start_time,end_time,point_in_time,date_of_birth,date_of_death,:LABEL
      - Edges: :START_ID,:END_ID,:TYPE
    """
    import csv
    # Nodes
    with nodes_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id:ID', 'title', 'category', 'wikidata_id', 'wikipedia:STRING', 'wikidata:STRING', 'start_time', 'end_time', 'point_in_time', 'date_of_birth', 'date_of_death', 'country_codes:STRING', 'country:STRING', 'gender_code', 'gender', 'occupation_codes:STRING', 'occupation:STRING', 'instance_of_codes:STRING', 'instance_of:STRING', 'subclass_of_codes:STRING', 'subclass_of:STRING', 'location_codes:STRING', 'location:STRING', 'latitude:FLOAT', 'longitude:FLOAT', ':LABEL'])
        with jsonl_in.open('r', encoding='utf-8') as fin:
            for line in fin:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                wp = (obj or {}).get('wikipedia') or {}
                try:
                    pid = int(wp.get('id'))
                except Exception:
                    continue
                title = id_to_title.get(pid, '')
                cat = id_to_category.get(pid, '')
                wd = (obj or {}).get('wikidata') or {}
                qid = wd.get('id') if isinstance(wd, dict) else None
                time_fields = extract_temporal_fields(wd if isinstance(wd, dict) else {})
                labels = 'Page'
                if cat == 'person':
                    labels = 'Page;Person'
                elif cat == 'event':
                    labels = 'Page;Event'
                # Person fields (codes only here; labels will be filled in run after resolution)
                countries = []
                gender_code = ''
                occupation_codes = []
                claims = (wd or {}).get('claims') or {}
                if isinstance(claims, dict):
                    countries = extract_entity_ids_from_claims(claims, 'P27') if cat == 'person' else []
                    g = extract_entity_ids_from_claims(claims, 'P21') if cat == 'person' else []
                    if g:
                        gender_code = g[0]
                    occupation_codes = extract_entity_ids_from_claims(claims, 'P106') if cat == 'person' else []
                    instance_of = extract_entity_ids_from_claims(claims, 'P31')
                    subclass_of = extract_entity_ids_from_claims(claims, 'P279')
                    location_codes = extract_entity_ids_from_claims(claims, 'P276')
                    lat, lon = extract_coordinates_from_claims(claims)
                country_str = ';'.join(countries) if countries else ''
                occ_str = ';'.join(occupation_codes) if occupation_codes else ''
                inst_str = ';'.join(instance_of) if 'instance_of' in locals() and instance_of else ''
                subc_str = ';'.join(subclass_of) if 'subclass_of' in locals() and subclass_of else ''
                loc_str = ';'.join(location_codes) if 'location_codes' in locals() and location_codes else ''
                w.writerow([
                    pid,
                    title,
                    cat,
                    qid or '',
                    json.dumps(wp, ensure_ascii=False),
                    json.dumps(wd, ensure_ascii=False),
                    time_fields.get('start_time', ''),
                    time_fields.get('end_time', ''),
                    time_fields.get('point_in_time', ''),
                    time_fields.get('date_of_birth', ''),
                    time_fields.get('date_of_death', ''),
                    country_str,
                    '',  # country labels filled later in run()
                    gender_code,
                    '',  # gender label filled later in run()
                    occ_str,
                    '',  # occupation labels later
                    inst_str,
                    '',  # instance_of labels later
                    subc_str,
                    '',  # subclass_of labels later
                    loc_str,
                    '',  # location labels later
                    lat if 'lat' in locals() and lat is not None else '',
                    lon if 'lon' in locals() and lon is not None else '',
                    labels,
                ])
    # Edges
    with edges_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([':START_ID', ':END_ID', ':TYPE'])
        for src, tgts in src_to_targets.items():
            for tgt in tgts:
                w.writerow([src, tgt, 'LINKS_TO'])


def run(sql_path_str: str,
        jsonl_in_path_str: str,
        jsonl_out_path_str: str,
        linktarget_sql_path_str: str | None = None,
        page_sql_path_str: str | None = None,
        pages_xml_bz2_path_str: str | None = None,
        preview_out_path_str: str | None = None,
        redirect_sql_path_str: str | None = None) -> None:
    jsonl_in = Path(jsonl_in_path_str)
    jsonl_out = Path(jsonl_out_path_str)

    title_to_id, id_set = build_title_and_id_sets(jsonl_in)
    # Enrich with Wikidata labels/aliases to catch redirects/alt names without SQL
    augment_title_map_with_wikidata(jsonl_in, title_to_id)
    id_to_category = build_id_to_category(jsonl_in)
    id_to_title = build_id_to_title(jsonl_in)
    redirect_map: Dict[str, str] | None = None
    # If we have both page.sql and redirect.sql, build a proper redirect map (optional)
    page_map = None
    if page_sql_path_str:
        pg_path = Path(page_sql_path_str)
        if pg_path.exists():
            page_map = parse_page_sql(pg_path)
    if redirect_sql_path_str and page_map is not None:
        rpath = Path(redirect_sql_path_str)
        if rpath.exists():
            redirect_map = build_redirect_map_with_page(rpath, page_map)
    # Per user request: ONLY extract links from JSONL wikipedia text with category filtering
    print("Extracting links from JSONL wikipedia text (by title) with category filtering ...")
    src_to_targets = build_links_from_jsonl_text(jsonl_in, id_set, title_to_id, id_to_category, redirect_map)

    # Resolve person field labels (countries, gender)
    person_qids = sorted(list(collect_person_qids_for_resolution(jsonl_in)))
    extra_qids = sorted(list(collect_additional_qids(jsonl_in)))
    all_qids = sorted(list(set(person_qids) | set(extra_qids)))
    qid_to_label = resolve_item_labels_en(all_qids) if all_qids else {}

    # Write normal and Neo4j JSONL, injecting resolved labels for people
    preview_out = Path(preview_out_path_str) if preview_out_path_str else None
    neo4j_out = Path(__file__).resolve().parent / "data_raw" / "wiki_entities_neo4j.jsonl"
    # To inject labels without rewriting extraction, post-process the written neo4j jsonl by streaming and adding fields
    # Simpler: during CSV export, we will insert labels; and we will also rewrite neo4j jsonl after writing.
    write_augmented_jsonl(jsonl_in, jsonl_out, src_to_targets, preview_out, id_to_title, neo4j_out)

    # Enrich Neo4j JSONL with resolved labels for people
    neo_in = neo4j_out
    neo_tmp = Path(str(neo4j_out) + '.tmp')
    with neo_in.open('r', encoding='utf-8') as fin, neo_tmp.open('w', encoding='utf-8') as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if obj.get('category') == 'person':
                # Countries
                cc = obj.get('country_codes')
                if isinstance(cc, list) and cc:
                    obj['country'] = [qid_to_label.get(q, q) for q in cc]
                # Gender
                gc = obj.get('gender_code')
                if isinstance(gc, str) and gc:
                    obj['gender'] = qid_to_label.get(gc, gc)
                # Occupations
                oc = obj.get('occupation_codes')
                if isinstance(oc, list) and oc:
                    obj['occupation'] = [qid_to_label.get(q, q) for q in oc]
            # Generic labels
            io = obj.get('instance_of_codes')
            if isinstance(io, list) and io:
                obj['instance_of'] = [qid_to_label.get(q, q) for q in io]
            sc = obj.get('subclass_of_codes')
            if isinstance(sc, list) and sc:
                obj['subclass_of'] = [qid_to_label.get(q, q) for q in sc]
            lc = obj.get('location_codes')
            if isinstance(lc, list) and lc:
                obj['location'] = [qid_to_label.get(q, q) for q in lc]
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    neo_tmp.replace(neo_in)

    # Graph exports (TSV for quick inspection)
    nodes_out = Path('/home/ubuntu/Videos/wiki_graph_nodes.tsv')
    edges_out = Path('/home/ubuntu/Videos/wiki_graph_edges.tsv')
    write_graph_outputs(jsonl_in, src_to_targets, id_to_title, id_to_category, nodes_out, edges_out)

    # Neo4j bulk-import CSVs (no mapping required)
    nodes_csv = Path('/home/ubuntu/Videos/wiki_nodes_bulk.csv')
    edges_csv = Path('/home/ubuntu/Videos/wiki_edges_bulk.csv')
    write_bulk_import_csv(jsonl_in, src_to_targets, id_to_title, id_to_category, nodes_csv, edges_csv)
    # Fill in country/gender labels in nodes CSV
    # Read and rewrite with labels for person rows
    import csv
    tmp_csv = Path(str(nodes_csv) + '.tmp')
    with nodes_csv.open('r', encoding='utf-8', newline='') as fin, tmp_csv.open('w', encoding='utf-8', newline='') as fout:
        # Increase CSV field size limit to handle large wikipedia/wikidata JSON columns
        try:
            import sys
            max_limit = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(max_limit)
                    break
                except OverflowError:
                    max_limit = int(max_limit / 10)
                    if max_limit < 10**7:
                        break
        except Exception:
            pass
        r = csv.reader(fin)
        w = csv.writer(fout)
        header = next(r, [])
        w.writerow(header)
        # Indices
        try:
            idx_cat = header.index('category')
            idx_country_codes = header.index('country_codes:STRING')
            idx_country = header.index('country:STRING')
            idx_gender_code = header.index('gender_code')
            idx_gender = header.index('gender')
        except ValueError:
            idx_cat = idx_country_codes = idx_country = idx_gender_code = idx_gender = -1
        for row in r:
            if idx_cat != -1 and row[idx_cat] == 'person':
                # country codes -> labels
                if idx_country_codes != -1 and idx_country != -1:
                    codes = row[idx_country_codes]
                    labels = []
                    if codes:
                        for q in codes.split(';'):
                            q = q.strip()
                            if q:
                                labels.append(qid_to_label.get(q, q))
                    row[idx_country] = ';'.join(labels) if labels else ''
                # gender code -> label
                if idx_gender_code != -1 and idx_gender != -1:
                    gc = row[idx_gender_code].strip()
                    if gc:
                        row[idx_gender] = qid_to_label.get(gc, gc)
            w.writerow(row)
    tmp_csv.replace(nodes_csv)


if __name__ == '__main__':
    # Adjust paths as needed, now project-relative
    from pathlib import Path as _Path

    project_root = _Path(__file__).resolve().parents[2]
    data_raw = _Path(__file__).resolve().parent / "data_raw"

    run(
        sql_path_str='',
        jsonl_in_path_str=str(data_raw / 'wiki_entities_with_category.jsonl'),
        jsonl_out_path_str=str(data_raw / 'wiki_entities_with_links.jsonl'),
        linktarget_sql_path_str=None,
        page_sql_path_str=str(project_root / 'viwiki_data' / 'viwiki-20251020-page.sql.gz'),
        pages_xml_bz2_path_str=None,
        preview_out_path_str=str(data_raw / 'wiki_entities_with_links_preview.jsonl'),
        redirect_sql_path_str=str(project_root / 'viwiki_data' / 'viwiki-20251020-redirect.sql.gz'),
    )


