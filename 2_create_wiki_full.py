#!/usr/bin/env python3
"""
Create full WikiEntity objects by combining Wikipedia articles with their Wikidata entities.
Reads from the dataset created by 0_create_wikidata_dataset.py and fetches Wikidata data.
Uses batch processing: fetches up to 50 Wikidata IDs per API request (API limit: 50).

Requirements:
    pip install aiohttp tqdm
    
API Optimization:
    - Fetches up to 50 QIDs per single API request (using pipe delimiter)
    - Reads 1000 lines from file, batches API requests in groups of 50
    - 0.5s delay between API requests to avoid overloading the server
    - Exponential backoff retry (up to 5 retries)
    - Respects Wikimedia API best practices
"""

import json
import asyncio
import aiohttp
from pathlib import Path
from tqdm import tqdm
from wikidata_entity import WikidataEntity
from wikipedia_entity import WikipediaEntity
from dataclasses import dataclass, asdict
from typing import Optional

# Bot credentials
BOT_USERNAME = "Minhtest@Wikidata_Fetch"
BOT_PASSWORD = "9qitqamrunbpet5gcuck3kmaeqmr5md0"

API_URL = "https://www.wikidata.org/w/api.php"

# API configuration
MAX_CONCURRENT_REQUESTS = 5  # Max parallel batch requests
API_TIMEOUT = 30  # 30 seconds timeout per request
MAX_RETRIES = 5  # Maximum retry attempts with exponential backoff
IDS_PER_REQUEST = 50  # Maximum Wikidata IDs per API request (API limit is 50)
API_DELAY_BETWEEN_BATCHES = 0.1  # Seconds to wait between batch requests (to avoid overload)


@dataclass
class WikiEntity:
    """
    Container for Wikipedia + Wikidata entities.
    Just a wrapper - each entity maintains its own methods.
    Both entities are mandatory.
    """
    wikipedia: WikipediaEntity
    wikidata: WikidataEntity
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "wikipedia": asdict(self.wikipedia),
            "wikidata": asdict(self.wikidata)
        }


async def login_to_wikidata(session: aiohttp.ClientSession) -> bool:
    """
    Login to Wikidata using modern token-based authentication.
    """
    headers = {"User-Agent": "WikiEntityBuilder/1.0 (Contact me@example.com)"}

    try:
        # Step 1: Get login token
        params_token = {
            "action": "query",
            "meta": "tokens",
            "type": "login",
            "format": "json",
        }
        async with session.get(API_URL, params=params_token, headers=headers) as resp:
            data = await resp.json()
            login_token = data["query"]["tokens"]["logintoken"]

        # Step 2: Perform login
        params_login = {
            "action": "login",
            "lgname": BOT_USERNAME,
            "lgpassword": BOT_PASSWORD,
            "lgtoken": login_token,
            "format": "json",
        }
        async with session.post(API_URL, data=params_login, headers=headers) as resp:
            data = await resp.json()
            if data.get("login", {}).get("result") == "Success":
                print("✅ Successfully logged in to Wikidata with bot account")
                return True
            else:
                print(f"❌ Login failed: {data}")
                return False

    except Exception as e:
        print(f"⚠️ Login error: {e}")
        return False


async def fetch_wikidata_entities_batch(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, qids: list) -> dict:
    """
    Async fetch multiple Wikidata entities in a single request with exponential backoff retry.
    Can handle up to 300 QIDs at once using pipe delimiter.

    Args:
        session: aiohttp session (already logged in)
        semaphore: Semaphore to limit concurrent requests
        qids: List of Wikidata QIDs

    Returns:
        Dict mapping QID -> entity data
    """
    if not qids:
        return {}

    # Join QIDs with pipe delimiter
    ids_param = "|".join(qids)

    params = {
        "action": "wbgetentities",
        "ids": ids_param,
        "format": "json",
        "languages": "vi",
    }

    headers = {"User-Agent": "WikiEntityBuilder/1.0 (Contact me@example.com)"}

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(API_URL, params=params, headers=headers, timeout=API_TIMEOUT) as response:
                    if response.status == 429:  # Rate limited
                        # Exponential backoff: 2^(attempt) seconds
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = await response.json()

                    # Return entities dict, empty dict if none found
                    if "entities" in data and data["entities"]:
                        # Filter out "missing" entities (entities that don't exist)
                        valid_entities = {k: v for k, v in data["entities"].items() if "missing" not in v}
                        return valid_entities
                    return {}

            except asyncio.TimeoutError:
                if attempt == MAX_RETRIES - 1:
                    return {}
                # Exponential backoff: 2^(attempt + 1) seconds
                wait_time = 2 ** (attempt + 1)
                await asyncio.sleep(wait_time)

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {}
                # Exponential backoff: 2^(attempt + 1) seconds
                wait_time = 2 ** (attempt + 1)
                await asyncio.sleep(wait_time)

    return {}


async def process_batch(batch_entries, session, semaphore, output_file, pbar):
    """
    Process a batch of entries: fetch Wikidata data in batches of up to 50 IDs and save to JSONL.

    Args:
        batch_entries: List of entry dictionaries from JSONL
        session: aiohttp session
        semaphore: Semaphore to limit concurrent requests
        output_file: File handle to write JSONL output
        pbar: Progress bar to update

    Returns:
        Tuple of (successful_count, failed_count, results)
    """
    successful = 0
    failed = 0
    results = []

    print(f"  Processing {len(batch_entries)} entries in this batch")

    # Filter entries with Wikidata IDs only and create mapping
    entry_map = {}
    for entry in batch_entries:
        if entry.get('wikidata_id'):
            qid = entry['wikidata_id']
            if qid not in entry_map:
                entry_map[qid] = []
            entry_map[qid].append(entry)

    print(f"  Found {len(entry_map)} unique QIDs in this batch")

    if not entry_map:
        print("  No entries with Wikidata IDs found in batch")
        return successful, failed, results

    # Split QIDs into chunks
    all_qids = list(entry_map.keys())
    chunk_size = IDS_PER_REQUEST

    print(f"  Splitting into chunks of {chunk_size} QIDs each")

    for i in range(0, len(all_qids), chunk_size):
        qid_chunk = all_qids[i:i + chunk_size]
        print(f"    Processing chunk {i//chunk_size + 1} with {len(qid_chunk)} QIDs: {qid_chunk[:3]}...")

        # Fetch batch of entities
        wikidata_results = await fetch_wikidata_entities_batch(session, semaphore, qid_chunk)
        print(f"    Fetched {len(wikidata_results)} entities from API")

        # Wait between API calls to avoid overloading the server
        if i + chunk_size < len(all_qids):
            await asyncio.sleep(API_DELAY_BETWEEN_BATCHES)

        # Process each fetched entity
        processed_in_chunk = 0
        for qid, wikidata_data in wikidata_results.items():
            if qid in entry_map and wikidata_data:
                for entry in entry_map[qid]:
                    try:
                        wikipedia_entity = WikipediaEntity(
                            id=entry['id'],
                            namespace=entry['namespace'],
                            title=entry['title'],
                            full_text=entry['full_text']
                        )

                        wikidata_entity = WikidataEntity.from_json({"entities": {qid: wikidata_data}})

                        # Create WikiEntity (both entities are required)
                        wiki_entity = WikiEntity(
                            wikipedia=wikipedia_entity,
                            wikidata=wikidata_entity
                        )

                        # Convert to dict and write to file
                        entity_dict = wiki_entity.to_dict()
                        output_file.write(json.dumps(entity_dict, ensure_ascii=False) + '\n')
                        output_file.flush()

                        results.append(entity_dict)
                        successful += 1
                        processed_in_chunk += 1

                        # Update progress bar for each successful entity
                        pbar.update(1)
                    except Exception as e:
                        # Print first error to see what's happening
                        if failed == 0:
                            print(f"\nError processing entity {qid}: {e}")
                        failed += 1
            else:
                # Entry in our map but not in results (fetch failed)
                if qid in entry_map:
                    failed += len(entry_map[qid])

        print(f"    Successfully processed {processed_in_chunk} entities in this chunk")

    print(f"  Batch complete: {successful} successful, {failed} failed")
    return successful, failed, results


def load_page_props_mapping(props_file_path: str) -> dict:
    """
    Load the page_props.sql.gz file and create a mapping from page_id to wikidata_id.
    Based on the approach from 2_prepare_wikidata_dumps.py

    Args:
        props_file_path: Path to the page_props.sql.gz file

    Returns:
        Dict mapping page_id (int) to wikidata_id (str)
    """
    import gzip
    import re

    print(f"Loading page props from: {props_file_path}")

    page_props_map = {}

    try:
        with gzip.open(props_file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc="Reading page props SQL", unit="line"):
                if "wikibase_item" in line:
                    # Extract QIDs using regex pattern from 2_prepare_wikidata_dumps.py
                    # Format: (page_id,'wikibase_item','Q123')
                    matches = re.findall(r"\((\d+),'wikibase_item','(Q\d+)'", line)
                    for page_id_str, qid in matches:
                        try:
                            page_id = int(page_id_str)
                            page_props_map[page_id] = qid
                        except ValueError:
                            continue

        print(f"Loaded {len(page_props_map):,} page-to-wikidata mappings")
        return page_props_map

    except Exception as e:
        print(f"Error loading page props: {e}")
        return {}


async def create_full_wiki_entities_from_full_text_async(
    input_file: str,
    props_file: str,
    output_file: str = "wiki_entities_full_from_xml.jsonl"
) -> bool:
    """Create full WikiEntity objects from Wikipedia full text data.

    Reads full text data, looks up Wikidata IDs from page_props.sql.gz,
    fetches Wikidata data via API, and saves combined entities to JSONL.

    Args:
        input_file: Path to wikipedia_full_text.json file.
        props_file: Path to page_props.sql.gz file.
        output_file: Path to output JSONL file. Defaults to
            "wiki_entities_full_from_xml.jsonl".

    Returns:
        True if processing completed successfully, False otherwise.
    """

    print("=" * 80)
    print("Creating Full WikiEntity Objects from Full Text Data")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Page props file: {props_file}")
    print(f"Output file: {output_file}")
    print()

    # Check if input files exist
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        return False

    if not Path(props_file).exists():
        print(f"Error: Page props file not found: {props_file}")
        return False

    # Load page props mapping
    page_props_map = load_page_props_mapping(props_file)
    if not page_props_map:
        print("Failed to load page props mapping. Exiting.")
        return False

    # Load full text data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            full_text_data = json.load(f)
        print(f"Loaded {len(full_text_data):,} entries from full text data")
    except Exception as e:
        print(f"Error loading full text data: {e}")
        return False

    # Statistics
    stats = {
        'total': len(full_text_data),
        'with_wikidata': 0,
        'skipped': 0,
        'successful': 0,
        'failed_fetches': 0,
        'batches_processed': 0
    }

    # Add Wikidata IDs to entries that have them, skip those that don't
    entries_with_qids = []
    for entry in full_text_data:
        page_id = entry.get('id')
        if page_id in page_props_map:
            wikidata_id = page_props_map[page_id]
            entry_copy = entry.copy()  # Don't modify original
            entry_copy['wikidata_id'] = wikidata_id
            entries_with_qids.append(entry_copy)
            stats['with_wikidata'] += 1
        else:
            stats['skipped'] += 1

    print(f"Total entries: {stats['total']:,}")
    print(f"Entries with Wikidata IDs: {stats['with_wikidata']:,}")
    print(f"Entries without Wikidata IDs: {stats['skipped']:,}")

    # Batch processing setup
    batch_size = 10000
    print(f"Processing: {batch_size} entries at a time")
    print(f"Efficiency: Up to {IDS_PER_REQUEST} Wikidata IDs per API request")
    print(f"Server-friendly: {API_DELAY_BETWEEN_BATCHES}s delay between API calls")
    print(f"Retry policy: Exponential backoff (max {MAX_RETRIES} retries)")
    print()

    async with aiohttp.ClientSession() as session:
        # Login with bot account first
        login_success = await login_to_wikidata(session)
        if not login_success:
            print("Failed to login with bot account. Exiting.")
            return False

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        with open(output_file, 'w', encoding='utf-8') as outf:
            pbar = tqdm(desc="Processing", unit="entries", total=len(entries_with_qids))

            # Process in batches
            for i in range(0, len(entries_with_qids), batch_size):
                batch = entries_with_qids[i:i + batch_size]
                stats['batches_processed'] += 1

                print(f"Processing batch {stats['batches_processed']} with {len(batch)} entries...")

                # Process batch: fetch Wikidata and write to file
                successful, failed, _ = await process_batch(batch, session, semaphore, outf, pbar)
                stats['successful'] += successful
                stats['failed_fetches'] += failed

                pbar.set_postfix({
                    'Success': stats['successful'],
                    'Failed': stats['failed_fetches'],
                    'Batches': stats['batches_processed']
                })

                # Small delay between batches to avoid API overload
                if i + batch_size < len(entries_with_qids):
                    await asyncio.sleep(0.5)

            pbar.close()

    print()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total entries processed: {stats['total']:,}")
    print(f"Entries with Wikidata ID: {stats['with_wikidata']:,}")
    print(f"Entries without Wikidata ID (skipped): {stats['skipped']:,}")
    print(f"Successfully created and saved WikiEntity objects: {stats['successful']:,}")
    print(f"Failed to fetch Wikidata data: {stats['failed_fetches']:,}")
    print(f"Batches processed: {stats['batches_processed']:,}")

    if stats['with_wikidata'] > 0:
        success_rate = (stats['successful'] / stats['with_wikidata']) * 100
        print(f"Success rate: {success_rate:.1f}%")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSaved {stats['successful']:,} WikiEntity objects to {output_file}")

    return True


def main() -> None:
    """Create full WikiEntity objects from Wikipedia and Wikidata data.

    Enriches Wikipedia entities with Wikidata data by fetching Wikidata
    properties for each entity via API and combining them into full
    WikiEntity objects.
    """
    input_file = "wikipedia_full_text.json"
    props_file = "viwiki_data/viwiki-20251020-page_props.sql.gz"
    output_file = "wiki_entities_full_from_xml.jsonl"

    # Create full WikiEntity objects from full text data
    success = asyncio.run(create_full_wiki_entities_from_full_text_async(input_file, props_file, output_file))

    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed!")


if __name__ == "__main__":
    main()

