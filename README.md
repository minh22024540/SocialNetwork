# Social Network Knowledge Graph Pipeline

This repository contains a complete pipeline for building a Vietnamese social network knowledge graph from Wikipedia data, enriching it with relationships, and creating a GraphRAG chatbot for multi-hop reasoning.

## Overview

The pipeline consists of 8 main phases:

1. **Parse Wikipedia** - Extract entities from Wikipedia XML dump
2. **Enrich with Wikidata** - Fetch Wikidata data for entities
3. **Flatten Properties** - Resolve and flatten Wikidata properties
4. **Categorize Entities** - Classify entities as Person/Event/etc.
5. **Build Page Links** - Extract inter-entity links from Wikipedia
6. **Enrich Relationships** - Extract semantic relationships using LLM
7. **Analyze Graph** - Calculate PageRank, communities, small-world metrics
8. **Generate Questions** - Create multi-hop question benchmark dataset

## Prerequisites

### System Requirements
- Python 3.8+
- Neo4j database (Docker recommended)
- Sufficient disk space (~10GB+ for full pipeline)

### Python Dependencies
```bash
pip install mwxml aiohttp tqdm neo4j networkx transformers torch
```

### External Services
- **Neo4j**: Running on `bolt://localhost:7687` (default)
  - Database: `wiki.db`
  - User: `neo4j`
  - Password: `neo4jtest12`
- **OpenAI API** (for relationship extraction and question generation):
  - Set `OPENAI_API_KEY` environment variable
  - Or configure `OPENAI_BASE_URL` for compatible API

## Directory Structure

```
SocialNetwork/
├── data/                    # Config and benchmark data
│   ├── 6_relationship_types.json
│   ├── entity_aliases_map.json
│   ├── graph_paths_for_questions.json
│   ├── multihop_questions.jsonl
│   └── sample_questions.jsonl
├── data_raw/                # Raw corpus and intermediate files
│   ├── wiki_entities_full_from_xml.jsonl
│   ├── wiki_entities_with_flat_statements.jsonl
│   ├── wiki_entities_with_category.jsonl
│   ├── wiki_entities_with_links.jsonl
│   ├── wiki_entities_with_weights.jsonl
│   ├── enriched_relationships.jsonl
│   └── enriched_relationships_cache.json
├── data_analysis/           # Graph analysis outputs
│   ├── pagerank_scores.json
│   ├── small_world_analysis.json
│   ├── communities.json
│   ├── edge_weights.json
│   └── properties_en_labels.json
└── chatbot/                # GraphRAG chatbot implementation
    ├── graph_store.py
    ├── graph_rag.py
    ├── alias_linker.py
    ├── ollama_client.py
    ├── evaluate_chatbot.py
    └── cli.py
```

## Pipeline Execution Order

### Phase 1: Data Extraction (Scripts 1-2)

**1. Parse Wikipedia XML Dump**
```bash
cd /home/ubuntu/Videos/code/SocialNetwork
python 1_parse_wiki.py
```
- **Input**: Wikipedia XML dump file (e.g., `viwiki-20251020-pages-articles.xml.bz2`)
- **Output**: `data_raw/wiki_entities_full_from_xml.jsonl`
- **Purpose**: Extract Wikipedia articles and identify Person/Event patterns

**2. Enrich with Wikidata**
```bash
python 2_create_wiki_full.py
```
- **Input**: `data_raw/wiki_entities_full_from_xml.jsonl`
- **Output**: `data_raw/wiki_entities_full_from_xml.jsonl` (enriched)
- **Purpose**: Fetch Wikidata properties for each entity via API

### Phase 2: Entity Processing (Scripts 3-4)

**3. Flatten and Resolve Properties**
```bash
python 3_flatten_and_resolve_properties.py
```
- **Input**: `data_raw/wiki_entities_full_from_xml.jsonl`
- **Output**: 
  - `data_raw/wiki_entities_with_flat_statements.jsonl`
  - `data_analysis/properties_en_labels.json`
- **Purpose**: Flatten nested Wikidata statements and resolve property labels

**4. Categorize Entities**
```bash
python 4_categorize_entities.py
```
- **Input**: `data_raw/wiki_entities_with_flat_statements.jsonl`
- **Output**: `data_raw/wiki_entities_with_category.jsonl`
- **Purpose**: Classify entities as Person, Event, Organization, Place, etc.

### Phase 3: Graph Construction (Script 5)

**5. Build Page Links Map**
```bash
python 5_build_pagelinks_map.py
```
- **Input**: 
  - `data_raw/wiki_entities_with_category.jsonl`
  - Wikipedia SQL dumps (page.sql.gz, redirect.sql.gz)
- **Output**: 
  - `data_raw/wiki_entities_with_links.jsonl`
  - `data_raw/wiki_entities_neo4j.jsonl`
- **Purpose**: Extract inter-entity links from Wikipedia pagelinks

### Phase 4: Relationship Enrichment (Scripts 6)

**6a. Build Entity Aliases**
```bash
python 6_build_entity_aliases.py
```
- **Input**: `data_raw/wiki_entities_with_links.jsonl`
- **Output**: `data/entity_aliases_map.json`
- **Purpose**: Extract all name variants (title, labels, aliases) for fuzzy matching

**6b. Calculate Edge Weights**
```bash
python 6_calculate_edge_weights.py
```
- **Input**: `data_raw/wiki_entities_with_links.jsonl`
- **Output**: 
  - `data_raw/wiki_entities_with_weights.jsonl`
  - `data_analysis/edge_weights.json`
- **Purpose**: Calculate edge weights based on link frequency and entity similarity

**6c. Extract Relationships with LLM**
```bash
python 6_extract_relationships_openai.py
```
- **Input**: `data_raw/wiki_entities_with_links.jsonl`
- **Output**: `data_raw/enriched_relationships.jsonl`
- **Purpose**: Use LLM to extract semantic relationships (PARTICIPATED_IN, FOUGHT_IN, etc.)
- **Note**: Requires `OPENAI_API_KEY` environment variable
- **Time**: This step can take several hours depending on dataset size

**6d. Import to Neo4j**
```bash
# Ensure Neo4j Docker container is running
docker start neo4j  # or your container name

# Copy data files to Neo4j import directory
docker cp data_raw/wiki_entities_with_weights.jsonl neo4j:/import/
docker cp data_raw/enriched_relationships.jsonl neo4j:/import/

# Run import script
bash 6_import_enriched_neo4j.sh
```
- **Purpose**: Import nodes and relationships into Neo4j for graph queries

### Phase 5: Graph Analysis (Scripts 7)

**7a. PageRank Analysis**
```bash
python 7_analyze_pagerank.py
```
- **Input**: `data_raw/wiki_entities_with_weights.jsonl`
- **Output**: `data_analysis/pagerank_scores.json`
- **Purpose**: Calculate PageRank scores to identify important entities

**7b. Small World Analysis**
```bash
python 7_analyze_small_world.py
```
- **Input**: `data_raw/wiki_entities_with_weights.jsonl`
- **Output**: `data_analysis/small_world_analysis.json`
- **Purpose**: Analyze small-world properties (clustering coefficient, path length)

**7c. Community Detection**
```bash
python 7_analyze_communities.py
```
- **Input**: `data_raw/wiki_entities_with_weights.jsonl`
- **Output**: `data_analysis/communities.json`
- **Purpose**: Detect communities using Louvain algorithm

### Phase 6: Question Generation (Scripts 8)

**8a. Sample Graph Paths**
```bash
python 8_sample_graph_paths.py
```
- **Input**: Neo4j database (via `GraphStore`)
- **Output**: `data/graph_paths_for_questions.json`
- **Purpose**: Sample 2000 multi-hop paths (2-hop, 3-hop, 4-hop) from the graph

**8b. Generate Questions**
```bash
# Phase 1: Generate sample questions for quality testing
python 8_generate_questions.py

# Review sample_questions.jsonl, refine prompt if needed

# Phase 2: Generate all 2000 questions
python 8_generate_questions.py --mass-produce
```
- **Input**: `data/graph_paths_for_questions.json`
- **Output**: 
  - `data/sample_questions.jsonl` (30 questions for testing)
  - `data/multihop_questions.jsonl` (2000 questions)
- **Purpose**: Use LLM to generate multi-hop questions from graph paths
- **Note**: Requires `OPENAI_API_KEY` environment variable

## Running the GraphRAG Chatbot

### Setup

1. **Install Transformers** (if not already installed):
```bash
pip install transformers torch
```

2. **Download Qwen2.5-0.5B model** (first run will auto-download):
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
```

### Evaluation on Benchmark

Run evaluation on the 2000-question benchmark:

```bash
cd /home/ubuntu/Videos/code/SocialNetwork
PYTHONPATH=code python -m SocialNetwork.chatbot.evaluate_chatbot
```

This will:
- Load questions from `data/multihop_questions.jsonl`
- For each question, retrieve relevant paths from Neo4j using GraphRAG
- Generate answers using Qwen2.5-0.5B
- Calculate accuracy by hop count and question type

### Interactive CLI Demo

Test the chatbot on a specific question:

```bash
PYTHONPATH=code python -m SocialNetwork.chatbot.cli --id q_000123
```

This will show:
- The graph context retrieved for the question
- The model's answer and reasoning

## Expected Outputs

After completing the full pipeline, you should have:

- **Graph Database**: Neo4j with Person/Event nodes and HAS_RELATIONSHIP edges
- **Benchmark Dataset**: 2000 multi-hop questions in `data/multihop_questions.jsonl`
- **Graph Analysis**: PageRank scores, communities, small-world metrics
- **Chatbot**: GraphRAG system that can answer multi-hop questions using the graph

## Troubleshooting

### Neo4j Connection Issues
- Ensure Neo4j container is running: `docker ps | grep neo4j`
- Check connection: `docker exec neo4j cypher-shell -u neo4j -p neo4jtest12 "RETURN 1"`
- Verify database exists: `SHOW DATABASES`

### Missing Data Files
- Check that previous pipeline steps completed successfully
- Verify files exist in `data_raw/` or `data/` directories
- Some scripts have fallback paths to `old_code/` directory

### API Rate Limits
- Scripts 6c and 8b make many API calls
- Adjust `MAX_CONCURRENT_REQUESTS` in the scripts if hitting rate limits
- Scripts include retry logic and exponential backoff

### Memory Issues
- Large JSONL files may require significant RAM
- Consider processing in batches or using streaming for very large files
- Neo4j import may require adjusting heap size: `-Xmx4g`

## Notes

- All file paths are **relative to the SocialNetwork directory**
- Scripts use `Path(__file__).resolve().parent` to find data directories
- The pipeline is designed to be **resumable** - scripts check for existing outputs
- Question generation (`8_generate_questions.py`) supports `--mass-produce` flag for full dataset

## License

[Add your license here]

