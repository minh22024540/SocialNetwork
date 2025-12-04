"""
Generate multi-hop questions from graph paths using OpenAI.

This module generates Vietnamese questions (True/False, Yes/No, Multiple Choice)
based on multi-hop paths in the social network graph.
"""

import json
import os
import sys
import asyncio
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from tqdm import tqdm

try:
    from openai import AsyncOpenAI
    from openai import RateLimitError, APIError
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    raise


# OpenAI API configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxx")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Rate limiting and concurrency
MAX_CONCURRENT_REQUESTS = 30
MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 1.0
BATCH_SIZE = 50

# Cached map: relationship type -> English description (from data/6_relationship_types.json)
REL_TYPE_DESCRIPTIONS: Dict[str, str] = {}


def load_relationship_type_descriptions() -> Dict[str, str]:
    """Load mapping from relationship type to description (once per run).

    Loads relationship type descriptions from 6_relationship_types.json and
    caches them globally to avoid repeated file reads.

    Returns:
        Dictionary mapping relationship type strings to their English
        descriptions. Returns empty dict if file not found or on error.
    """
    global REL_TYPE_DESCRIPTIONS
    if REL_TYPE_DESCRIPTIONS:
        return REL_TYPE_DESCRIPTIONS

    types_path = Path(__file__).resolve().parent / "data" / "6_relationship_types.json"
    if not types_path.exists():
        return {}

    try:
        with types_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    type_map: Dict[str, str] = {}
    for item in data.get("relationship_types", []):
        t = item.get("type")
        desc = item.get("description")
        if t and desc:
            type_map[t] = desc

    REL_TYPE_DESCRIPTIONS = type_map
    return REL_TYPE_DESCRIPTIONS


def load_graph_paths(paths_file: Path) -> List[Dict]:
    """Load graph paths from JSON file.

    Args:
        paths_file: Path to JSON file containing graph paths.

    Returns:
        List of path dictionaries, each containing hop_count, path, and
        relationships keys.

    Raises:
        FileNotFoundError: If paths_file does not exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    with paths_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_path_for_prompt(path_data: Dict) -> str:
    """Format a graph path for inclusion in the prompt.

    Converts a graph path structure into a human-readable string format
    suitable for LLM prompts, including node details and relationship
    information with evidence text.

    Args:
        path_data: Path dictionary with 'path' and 'relationships' keys.
            Path should contain nodes with type, name, id, description.
            Relationships should contain type, confidence, evidence_text.

    Returns:
        Formatted multi-line string describing the path, with nodes and
        edges clearly labeled. Evidence text is truncated if >400 chars.
    """
    path = path_data.get("path", [])
    relationships = path_data.get("relationships", [])

    rel_type_desc = load_relationship_type_descriptions()

    lines: List[str] = []
    for i, node in enumerate(path):
        node_type = node.get("type", "")
        node_name = node.get("name", "")
        node_desc = node.get("description") or node.get("summary") or ""
        node_id = node.get("id") or node.get("neo4j_id") or i

        lines.append(f"Node {i+1}:")
        lines.append(f"- id: {node_id}")
        lines.append(f"- loại: {node_type}")
        lines.append(f"- tên: {node_name}")
        if node_desc:
            lines.append(f"- mô tả: {node_desc}")

        # Add relationship if available (relationships connect to next node)
        # For path: P1 -> E1 -> P2, relationships are [P1->E1, E1<-P2]
        # So relationship[i] connects node[i] to node[i+1]
        if i < len(path) - 1 and i < len(relationships):
            rel = relationships[i]
            rel_type = rel.get("type", "")
            rel_conf = rel.get("confidence")
            rel_evidence = (rel.get("evidence_text") or "").strip()
            # English description of this relationship type from 6_relationship_types.json
            rel_type_description = rel_type_desc.get(rel_type, "")

            lines.append(f"Edge {i+1}:")
            lines.append(f"- từ: Node {i+1}")
            lines.append(f"- đến: Node {i+2}")
            if rel_type:
                lines.append(f"- loại quan hệ (type): {rel_type}")
            if rel_type_description:
                lines.append(f"- mô tả loại quan hệ (tiếng Anh): {rel_type_description}")
            if isinstance(rel_conf, (int, float)):
                lines.append(f"- độ tin cậy (confidence): {rel_conf:.2f}")
            if rel_evidence:
                # Truncate very long evidence for prompt readability
                ev = rel_evidence.replace("\n", " ").strip()
                if len(ev) > 400:
                    ev = ev[:400].rstrip() + "..."
                lines.append(f"- trích dẫn gốc (evidence_text): \"{ev}\"")

        lines.append("")  # blank line between blocks

    return "\n".join(lines).strip()


def create_question_prompt(
    path_data: Dict, question_type: Optional[str] = None
) -> str:
    """Create a prompt for generating a question from a graph path.

    Builds a comprehensive prompt in Vietnamese that instructs the LLM to
    generate a multi-hop reasoning question based on the provided graph path.

    Args:
        path_data: Path dictionary containing hop_count, path, and
            relationships. Used to format the path context.
        question_type: Optional question type hint. One of:
            - "true_false": Generate True/False question
            - "yes_no": Generate Yes/No question
            - "multiple_choice": Generate multiple choice with 4 options
            If None, LLM chooses the type.

    Returns:
        Complete prompt string in Vietnamese with instructions for question
        generation, including path context and formatting requirements.
    """
    hop_count = path_data.get("hop_count", 2)
    path_str = format_path_for_prompt(path_data)
    
    type_instruction = ""
    if question_type:
        if question_type == "true_false":
            type_instruction = "Tạo một câu hỏi Đúng/Sai."
        elif question_type == "yes_no":
            type_instruction = "Tạo một câu hỏi Có/Không."
        elif question_type == "multiple_choice":
            type_instruction = "Tạo một câu hỏi trắc nghiệm với 4 lựa chọn (A, B, C, D)."
    else:
        type_instruction = "Tạo một câu hỏi (có thể là Đúng/Sai, Có/Không, hoặc trắc nghiệm 4 lựa chọn)."
    
    prompt = f"""Bạn là một chuyên gia tạo câu hỏi đánh giá khả năng suy luận đa bước của mô hình ngôn ngữ dựa trên đồ thị tri thức về con người và sự kiện.

Bạn được cung cấp MỘT đường đi trong đồ thị tri thức. Mỗi node là một thực thể (Person, Event, Organization, Place, ...) và mỗi edge là một quan hệ giữa hai node với:
- loại quan hệ (ví dụ: PARTICIPATED_IN, FOUGHT_IN, SUPPORTED, LED, ...)
- mô tả tiếng Anh ngắn gọn về loại quan hệ đó
- điểm độ tin cậy (confidence) cho quan hệ, càng cao càng đáng tin cậy.
- một trích dẫn gốc (evidence_text) từ văn bản để minh hoạ quan hệ đó.

ĐÂY LÀ ĐƯỜNG ĐI (yêu cầu suy luận đúng {hop_count}-hop, tức là đi qua {hop_count} quan hệ liên tiếp):

{path_str}

{type_instruction}

HƯỚNG DẪN TẠO CÂU HỎI:
1. Tạo câu hỏi bằng tiếng Việt, tự nhiên và trực tiếp như một câu hỏi lịch sử thực sự. Hãy đặt câu hỏi như thể bạn đang hỏi học sinh về lịch sử Việt Nam, sử dụng ngôn ngữ đời thường và trực tiếp (ví dụ: "Ai là người...", "Có phải...", "Sự kiện nào...", "Mối quan hệ giữa... và... là gì?").

   CẤM TUYỆT ĐỐI sử dụng các cụm từ "meta" như:
   - "Dựa trên các bằng chứng được cung cấp"
   - "Dựa trên thông tin đã cho"
   - "Theo đường đi trong đồ thị"
   - "Thông qua các quan hệ được mô tả"
   - "Dựa trên dữ liệu"
   - Bất kỳ cụm từ nào ám chỉ rằng câu hỏi dựa trên dữ liệu được cung cấp
   
   Câu hỏi phải đọc như một câu hỏi lịch sử tự nhiên, không có dấu hiệu nào cho thấy nó được tạo từ dữ liệu có cấu trúc.

2. Câu hỏi phải yêu cầu người trả lời kết hợp thông tin qua đúng {hop_count} bước quan hệ liên tiếp. Hãy đảm bảo người trả lời cần suy luận qua tất cả các bước trong đường đi để trả lời đúng.

3. Viết câu hỏi súc tích và tập trung vào điểm chính. Hãy để phần giải thích chi tiết về các bước suy luận trong trường "reasoning", còn phần "question" chỉ cần đặt câu hỏi một cách trực tiếp. Câu hỏi nên ngắn gọn (1-2 câu), không cần liệt kê tất cả các bước suy luận trong câu hỏi.

4. Với đường đi 2-hop (Person – Event – Person), hãy tạo câu hỏi khám phá mối quan hệ hoặc điểm chung giữa hai nhân vật trong bối cảnh sự kiện trung gian. Ví dụ: hỏi về vai trò của họ trong sự kiện, mối liên hệ giữa họ thông qua sự kiện, hoặc so sánh trải nghiệm của họ.

5. Sử dụng các động từ và cụm từ tiếng Việt tự nhiên để diễn đạt các loại quan hệ: "tham gia", "chiến đấu trong", "ủng hộ", "lãnh đạo", "chứng kiến", "có liên hệ với", "ảnh hưởng đến", v.v. QUAN TRỌNG: Hãy đọc kỹ evidence_text để hiểu chính xác ý nghĩa của quan hệ. Ví dụ: "làm phim về" một sự kiện KHÁC với "tham gia" sự kiện đó. "Làm phim tài liệu về Chiến tranh Đông Dương" không có nghĩa là "tham gia Chiến tranh Đông Dương".

6. Chỉ sử dụng thông tin có trong các node và edge của đường đi này (bao gồm cả evidence_text). Hãy đọc kỹ evidence_text để đảm bảo câu hỏi phản ánh đúng những gì bằng chứng thực sự nói, không phải suy diễn sai. KHÔNG tạo các mối quan hệ nhân quả ("thông qua", "dẫn đến") nếu evidence_text không chỉ ra điều đó một cách rõ ràng.

7. Tạo câu hỏi có độ khó phù hợp, yêu cầu người trả lời phải suy luận thực sự qua các bước trong đường đi, không quá dễ hoặc quá hiển nhiên.

8. Nếu là câu hỏi trắc nghiệm, tạo 4 lựa chọn (A, B, C, D) với chỉ một đáp án đúng. Các lựa chọn sai nên hợp lý và có liên quan nhưng không khớp với thông tin trong đường đi.

9. Đảm bảo đáp án đúng có thể suy ra được duy nhất từ đường đi trong đồ thị, dựa trên các quan hệ và evidence_text được cung cấp.

ĐỊNH DẠNG ĐẦU RA:
Trả về MỘT đối tượng JSON với format sau (không có giải thích thêm bên ngoài JSON):
{{
  "question": "Câu hỏi tiếng Việt...",
  "type": "true_false|yes_no|multiple_choice",
  "correct_answer": "True|False|Yes|No|A|B|C|D",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],  // Chỉ có nếu type là multiple_choice
  "reasoning": "Giải thích rõ ràng các bước suy luận bằng cách mô tả mối quan hệ thực tế giữa các nhân vật và sự kiện dựa trên evidence_text. Với mỗi bước, hãy nêu: (1) nhân vật/sự kiện nào liên kết với nhân vật/sự kiện nào, (2) loại quan hệ cụ thể là gì dựa trên evidence_text (ví dụ: 'tham gia' nếu evidence nói về việc nhập ngũ/chiến đấu, 'làm phim về' nếu evidence nói về việc làm phim tài liệu, 'lãnh đạo' nếu evidence nói về vai trò chỉ huy...), và (3) trích dẫn ngắn gọn từ evidence_text để minh chứng. QUAN TRỌNG: Phân biệt rõ các loại quan hệ - 'làm phim về' khác với 'tham gia', 'có liên hệ với' khác với 'chiến đấu trong'. KHÔNG dùng các thuật ngữ kỹ thuật như 'node', 'edge', 'bước 1 đi từ node X qua edge Y'. Thay vào đó, hãy mô tả như một câu chuyện lịch sử tự nhiên dựa trên bằng chứng thực tế."
}}

Chỉ trả về JSON, không trả về bất kỳ text nào khác ngoài JSON."""

    return prompt


async def generate_question_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    path_data: Dict,
    question_id: str,
    question_type: Optional[str] = None,
    token_usage: Optional[Dict] = None
) -> Optional[Dict]:
    """Generate a question from a graph path using OpenAI (async).
    
    Args:
        client: AsyncOpenAI client instance.
        semaphore: Semaphore for rate limiting.
        path_data: Path dictionary.
        question_id: Unique question ID.
        question_type: Optional question type hint.
        token_usage: Optional dictionary to accumulate token usage stats.
        
    Returns:
        Question dictionary or None if generation fails.
    """
    prompt = create_question_prompt(path_data, question_type)
    
    for attempt in range(MAX_RETRIES):
        try:
            # Build request parameters
            request_params = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "Bạn là chuyên gia tạo câu hỏi. Trả về chỉ JSON hợp lệ."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4
            }
            
            # Only hold semaphore during actual API call
            wait_start = time.time()
            async with semaphore:
                wait_time = time.time() - wait_start
                # Only log if waited very long for semaphore (>30s suggests serious bottleneck)
                if wait_time > 30.0:
                    print(f"[Semaphore] {question_id} waited {wait_time:.2f}s for semaphore")
                api_start = time.time()
                try:
                    # Set 60 second timeout for API call
                    response = await asyncio.wait_for(
                        client.chat.completions.create(**request_params),
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    api_time = time.time() - api_start
                    print(f"[Timeout] {question_id} API call timed out after {api_time:.1f}s")
                    # Semaphore will be released when exiting this block
                    if attempt < MAX_RETRIES - 1:
                        # Release semaphore before retrying (exit async with block)
                        pass  # Will exit block and release semaphore
                    else:
                        return None
                    # Sleep outside semaphore block
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                api_time = time.time() - api_start
                # Only log very slow API calls (>50s)
                if api_time > 50:
                    print(f"[API] {question_id} API call took {api_time:.1f}s")
            
            # Extract token usage if available (outside semaphore)
            if token_usage is not None and hasattr(response, 'usage') and response.usage:
                usage = response.usage
                
                # Basic token counts
                token_usage['prompt_tokens'] = token_usage.get('prompt_tokens', 0) + (usage.prompt_tokens or 0)
                token_usage['completion_tokens'] = token_usage.get('completion_tokens', 0) + (usage.completion_tokens or 0)
                token_usage['total_tokens'] = token_usage.get('total_tokens', 0) + (usage.total_tokens or 0)
                
                # Reasoning tokens from completion_tokens_details
                if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                    details = usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens is not None:
                        token_usage['reasoning_tokens'] = token_usage.get('reasoning_tokens', 0) + (details.reasoning_tokens or 0)
                
                # Cached tokens from prompt_tokens_details
                if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                    details = usage.prompt_tokens_details
                    if hasattr(details, 'cached_tokens') and details.cached_tokens is not None:
                        token_usage['cached_tokens'] = token_usage.get('cached_tokens', 0) + (details.cached_tokens or 0)
            
            if not response.choices:
                if attempt == MAX_RETRIES - 1:  # Log on final attempt
                    print(f"[NoChoices] question_id={question_id}, attempt={attempt+1}/{MAX_RETRIES}")
                    print(f"  Response: {response}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return None
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response - try multiple patterns
            json_str = None
            
            # If content starts with {, use it directly (most reliable)
            content_stripped = content.strip()
            if content_stripped.startswith('{'):
                json_str = content_stripped
            else:
                # Try code block
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    # Try to find JSON object directly (may span multiple lines)
                    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
            
            if not json_str:
                # No JSON found in response - retry
                if attempt == MAX_RETRIES - 1:  # Log on final attempt
                    print(f"[NoJSON] question_id={question_id}, attempt={attempt+1}/{MAX_RETRIES}")
                    print(f"  Response content (first 500 chars): {content[:500]}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return None
            
            # Parse JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                # Remove trailing commas before } or ]
                json_str_fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                try:
                    result = json.loads(json_str_fixed)
                except json.JSONDecodeError as e2:
                    if attempt == MAX_RETRIES - 1:  # Log on final attempt
                        print(f"[JSONParseError] question_id={question_id}, attempt={attempt+1}/{MAX_RETRIES}")
                        print(f"  Original error: {e}")
                        print(f"  Fixed error: {e2}")
                        print(f"  JSON string (first 500 chars): {json_str[:500]}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    return None
            
            # Validate and format result
            if "question" in result and "type" in result and "correct_answer" in result:
                question_obj = {
                    "id": question_id,
                    "question": result["question"],
                    "type": result["type"],
                    "correct_answer": result["correct_answer"],
                    "hop_count": path_data.get("hop_count", 2),
                    "source_path": path_data.get("path", []),
                    "reasoning": result.get("reasoning", ""),
                    "difficulty": "medium" if path_data.get("hop_count", 2) <= 2 else "hard"
                }
                
                # Add options for multiple choice
                if result["type"] == "multiple_choice" and "options" in result:
                    question_obj["options"] = result["options"]
                
                return question_obj
            else:
                # Missing required fields
                if attempt == MAX_RETRIES - 1:  # Log on final attempt
                    print(f"[MissingFields] question_id={question_id}, attempt={attempt+1}/{MAX_RETRIES}")
                    print(f"  Result keys: {list(result.keys())}")
                    print(f"  Result: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return None
                
        except json.JSONDecodeError as e:
            # This shouldn't happen since we catch it above, but log if it does
            print(f"[JSONDecodeError-Outer] question_id={question_id}, attempt={attempt+1}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES - 1:
                print(f"  This error occurred outside the expected JSON parsing block")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            return None
            
        except RateLimitError as e:
            backoff_time = RATE_LIMIT_BASE_DELAY * (2 ** attempt)
            print(f"[RateLimitError] question_id={question_id}, attempt={attempt+1}/{MAX_RETRIES}, backoff={backoff_time:.1f}s: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(backoff_time)
                continue
            return None
            
        except APIError as e:
            is_rate_limit = False
            if hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit = True
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                if e.response.status_code == 429:
                    is_rate_limit = True
            
            # Log API error details
            error_msg = f"[APIError] question_id={question_id}, attempt={attempt+1}/{MAX_RETRIES}: {e}"
            if hasattr(e, 'status_code'):
                error_msg += f", status_code={e.status_code}"
            if hasattr(e, 'response'):
                try:
                    if hasattr(e.response, 'text'):
                        error_msg += f"\n  Response text: {e.response.text[:500]}"
                    if hasattr(e.response, 'json'):
                        error_msg += f"\n  Response JSON: {e.response.json()}"
                except:
                    pass
            print(error_msg)
            
            if is_rate_limit:
                backoff_time = RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff_time)
                    continue
            else:
                backoff_time = RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff_time)
                    continue
            return None
            
        except Exception as e:
            error_msg = f"[Exception] question_id={question_id}, attempt={attempt+1}/{MAX_RETRIES}: {type(e).__name__}: {e}"
            # Try to get more details about the exception
            if hasattr(e, '__dict__'):
                error_msg += f"\n  Exception details: {e.__dict__}"
            print(error_msg)
            import traceback
            if attempt == MAX_RETRIES - 1:  # Full traceback on final attempt
                print(f"  Traceback:\n{traceback.format_exc()}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            return None
        
        return None


async def generate_questions_batch(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    paths: List[Dict],
    start_id: int = 1,
    question_types: Optional[List[str]] = None,
    token_usage: Optional[Dict] = None,
    pbar: Optional[tqdm] = None,
    output_file: Optional[Path] = None,
    write_lock: Optional[asyncio.Lock] = None
) -> List[Dict]:
    """Generate questions for a batch of paths.
    
    Args:
        client: AsyncOpenAI client instance.
        semaphore: Semaphore for rate limiting.
        paths: List of path dictionaries.
        start_id: Starting question ID.
        question_types: Optional list of question types to use (for distribution).
        token_usage: Optional dictionary to accumulate token usage stats.
        pbar: Optional tqdm progress bar to update.
        output_file: Optional path to output file for immediate writing.
        write_lock: Optional asyncio lock for thread-safe file writing.
        
    Returns:
        List of generated question dictionaries.
    """
    async def generate_with_progress(path, idx, q_type, q_id):
        """Wrapper to update progress bar and write immediately when task completes."""
        start_time = time.time()
        try:
            result = await generate_question_async(client, semaphore, path, q_id, q_type, token_usage)
            elapsed = time.time() - start_time
            if pbar:
                pbar.update(1)
            # Only log very slow requests (>60s)
            if elapsed > 60:
                print(f"[Slow] {q_id} completed in {elapsed:.1f}s")
            
            # Write immediately to file if output_file is provided
            if result and output_file and write_lock:
                async with write_lock:
                    with output_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()  # Ensure it's written to disk immediately
            
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            if pbar:
                pbar.update(1)
            print(f"[Failed] {q_id} failed after {elapsed:.1f}s: {type(e).__name__}: {e}")
            return e
    
    # Create all tasks upfront - they will start concurrently up to semaphore limit
    tasks = []
    for i, path in enumerate(paths):
        question_id = f"q_{start_id + i:06d}"
        # Distribute question types if provided
        q_type = None
        if question_types:
            q_type = question_types[i % len(question_types)]
        
        # Create task immediately - it will wait for semaphore but start as soon as slot available
        task = asyncio.create_task(generate_with_progress(path, i, q_type, question_id))
        tasks.append(task)
    
    # Wait for all tasks to complete - they run concurrently up to semaphore limit
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    questions = []
    error_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_count += 1
            if error_count <= 3:  # Log first 3 errors
                print(f"Error in task {i}: {type(result).__name__}: {result}")
                import traceback
                traceback.print_exc()
            continue
        if result:
            questions.append(result)
        else:
            error_count += 1
    
    if error_count > 0:
        print(f"Warning: {error_count} questions failed to generate (returned None or Exception)")
    
    return questions


def get_path_identifier(path: Dict) -> str:
    """Generate a unique identifier for a path based on node IDs.
    
    Args:
        path: Path dictionary with 'path' key containing nodes.
        
    Returns:
        String identifier: "id1-id2-id3-..." based on node IDs in order.
    """
    node_ids = [str(node.get("id", "")) for node in path.get("path", [])]
    return "-".join(node_ids)


def load_existing_questions(output_file: Path) -> Dict[str, Dict]:
    """Load existing questions from output file and return a map of path_id -> question.
    
    Args:
        output_file: Path to JSONL file with existing questions.
        
    Returns:
        Dictionary mapping path identifiers to question dictionaries.
    """
    existing = {}
    if not output_file.exists():
        return existing
    
    try:
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    # Extract path identifier from source_path
                    source_path = q.get("source_path", [])
                    if source_path:
                        path_ids = [str(node.get("id", "")) for node in source_path]
                        path_id = "-".join(path_ids)
                        existing[path_id] = q
    except Exception as e:
        print(f"Warning: Could not load existing questions: {e}")
    
    return existing


def deduplicate_questions(questions: List[Dict]) -> List[Dict]:
    """Remove duplicate questions (exact and semantic).
    
    Args:
        questions: List of question dictionaries.
        
    Returns:
        Deduplicated list of questions.
    """
    seen_questions: Set[str] = set()
    unique_questions = []
    
    for q in questions:
        # Normalize question text for comparison
        question_text = q.get("question", "").lower().strip()
        # Remove extra whitespace
        question_text = re.sub(r'\s+', ' ', question_text)
        
        if question_text and question_text not in seen_questions:
            seen_questions.add(question_text)
            unique_questions.append(q)
    
    return unique_questions


async def main() -> None:
    """Main function to generate questions.

    Orchestrates the question generation pipeline:
    1. Loads graph paths from JSON file
    2. Phase 1 (if not --mass-produce): Generates 30 sample questions
    3. Phase 2 (if --mass-produce): Generates all 2000 questions with
       resume capability and incremental writing.

    Supports resumable generation by checking existing questions and
    skipping already-processed paths.
    """
    print("=" * 80)
    print("Generating Multi-hop Questions")
    print("=" * 80)
    
    # Load graph paths (relative to SocialNetwork project root)
    paths_file = Path(__file__).resolve().parent / "data" / "graph_paths_for_questions.json"
    if not paths_file.exists():
        print(f"Error: Paths file not found: {paths_file}")
        print("Please run 8_sample_graph_paths.py first.")
        return
    
    all_paths = load_graph_paths(paths_file)
    print(f"Loaded {len(all_paths)} graph paths")
    
    # Separate by hop count
    paths_2hop = [p for p in all_paths if p.get("hop_count") == 2]
    paths_3hop = [p for p in all_paths if p.get("hop_count") == 3]
    paths_4hop = [p for p in all_paths if p.get("hop_count") == 4]
    
    print(f"  {len(paths_2hop)} 2-hop paths")
    print(f"  {len(paths_3hop)} 3-hop paths")
    print(f"  {len(paths_4hop)} 4-hop paths")
    
    # Initialize OpenAI client with timeout
    try:
        from httpx import Timeout
        timeout_config = Timeout(60.0, connect=10.0)  # 60s total, 10s connect timeout
    except ImportError:
        timeout_config = 60.0  # Fallback to float timeout
    
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=timeout_config
    )
    print(f"\nUsing OpenAI API:")
    print(f"  Model: {OPENAI_MODEL}")
    print(f"  Base URL: {OPENAI_BASE_URL}")
    print(f"  Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Question type distribution (mix)
    question_types = ["true_false", "yes_no", "multiple_choice"]
    
    # Token usage tracking
    token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'reasoning_tokens': 0,  # For models that support it (o1, o3, etc.)
        'cached_tokens': 0  # From prompt_tokens_details.cached_tokens (cached/reused tokens, cheaper)
    }
    
    # Check if mass production mode
    mass_produce = "--mass-produce" in sys.argv
    
    # Phase 1: Generate sample questions for quality testing (skip if mass production)
    if not mass_produce:
        print("\n" + "=" * 80)
        print("Phase 1: Generating Sample Questions (Quality Testing)")
        print("=" * 80)
        
        sample_size = 30  # 10 per hop level
        sample_paths = (
            paths_2hop[:10] + 
            paths_3hop[:10] + 
            (paths_4hop[:10] if paths_4hop else paths_2hop[10:20])
        )
        
        print(f"Generating {len(sample_paths)} sample questions...")
        with tqdm(total=len(sample_paths), desc="Sample questions", unit="question") as pbar:
            sample_questions = await generate_questions_batch(
                client, semaphore, sample_paths, start_id=1, question_types=question_types, token_usage=token_usage, pbar=pbar
            )
        
        # Save sample questions
        sample_output = Path(__file__).resolve().parent / "data" / "sample_questions.jsonl"
        with sample_output.open("w", encoding="utf-8") as f:
            for q in sample_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        
        print(f"\nGenerated {len(sample_questions)} sample questions")
        print(f"Saved to: {sample_output}")
        
        # Display token usage
        print("\nToken Usage (Sample Generation):")
        print(f"  Prompt tokens: {token_usage['prompt_tokens']:,}")
        print(f"  Completion tokens: {token_usage['completion_tokens']:,}")
        if token_usage['reasoning_tokens'] > 0:
            print(f"  Reasoning tokens: {token_usage['reasoning_tokens']:,}")
        if token_usage['cached_tokens'] > 0:
            print(f"  Cached tokens: {token_usage['cached_tokens']:,} (reused, lower cost)")
        print(f"  Total tokens: {token_usage['total_tokens']:,}")
        
        print("\nPlease review the sample questions and refine the prompt if needed.")
        print("Then run this script again with --mass-produce flag to generate all 2000 questions.")
    
    # Phase 2: Mass production (all 2000 questions)
    if mass_produce:
        print("\n" + "=" * 80)
        print("Phase 2: Mass Production (All 2000 Questions)")
        print("=" * 80)
        
        # Check for resume capability
        output_file = Path(__file__).resolve().parent / "data" / "multihop_questions.jsonl"
        existing_questions = load_existing_questions(output_file)
        
        if existing_questions:
            print(f"\nResuming: Found {len(existing_questions)} existing questions")
            print("Skipping paths that already have questions generated...")
        else:
            print("\nStarting fresh generation...")
        
        # Filter out paths that already have questions
        def filter_existing_paths(paths: List[Dict]) -> List[Dict]:
            """Filter out paths that already have questions."""
            filtered = []
            skipped = 0
            for path in paths:
                path_id = get_path_identifier(path)
                if path_id not in existing_questions:
                    filtered.append(path)
                else:
                    skipped += 1
            if skipped > 0:
                print(f"  Skipped {skipped} paths that already have questions")
            return filtered
        
        paths_2hop_filtered = filter_existing_paths(paths_2hop)
        paths_3hop_filtered = filter_existing_paths(paths_3hop)
        paths_4hop_filtered = filter_existing_paths(paths_4hop)
        
        total_remaining = len(paths_2hop_filtered) + len(paths_3hop_filtered) + len(paths_4hop_filtered)
        total_original = len(paths_2hop) + len(paths_3hop) + len(paths_4hop)
        
        print(f"\nRemaining paths to process: {total_remaining} / {total_original}")
        
        if total_remaining == 0:
            print("\nAll questions already generated! Nothing to do.")
            return
        
        all_questions = []
        # Start question_id from existing count + 1
        question_id = len(existing_questions) + 1
        
        # Create lock for thread-safe file writing
        write_lock = asyncio.Lock()
        
        # Generate questions for each hop level
        with tqdm(total=total_remaining, desc="Generating questions", unit="question") as main_pbar:
            for hop_count, paths in [(2, paths_2hop_filtered), (3, paths_3hop_filtered), (4, paths_4hop_filtered)]:
                if not paths:
                    continue
                
                print(f"\nGenerating questions for {hop_count}-hop paths ({len(paths)} paths)...")
                
                # Process in batches
                for i in range(0, len(paths), BATCH_SIZE):
                    batch_paths = paths[i:i+BATCH_SIZE]
                    batch_questions = await generate_questions_batch(
                        client, semaphore, batch_paths, 
                        start_id=question_id, question_types=question_types, token_usage=token_usage, 
                        pbar=main_pbar, output_file=output_file, write_lock=write_lock
                    )
                    all_questions.extend(batch_questions)
                    question_id += len(batch_paths)
        
        # Combine existing and new questions
        all_questions.extend(existing_questions.values())
        
        # Deduplicate
        print("\nDeduplicating questions...")
        unique_questions = deduplicate_questions(all_questions)
        print(f"After deduplication: {len(unique_questions)} questions (including {len(existing_questions)} existing)")
        
        # Re-save all questions (overwrite with deduplicated version)
        with output_file.open("w", encoding="utf-8") as f:
            for q in unique_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        
        print(f"\nSaved {len(unique_questions)} questions to: {output_file}")
        
        # Statistics
        by_type = {}
        by_hop = {}
        for q in unique_questions:
            q_type = q.get("type", "unknown")
            hop = q.get("hop_count", 0)
            by_type[q_type] = by_type.get(q_type, 0) + 1
            by_hop[hop] = by_hop.get(hop, 0) + 1
        
        print("\nStatistics:")
        print("By type:", by_type)
        print("By hop count:", by_hop)
        
        # Display final token usage
        print("\n" + "=" * 80)
        print("Token Usage Summary")
        print("=" * 80)
        print(f"Prompt tokens (input): {token_usage['prompt_tokens']:,}")
        print(f"Completion tokens (output): {token_usage['completion_tokens']:,}")
        if token_usage['reasoning_tokens'] > 0:
            print(f"Reasoning tokens: {token_usage['reasoning_tokens']:,}")
        if token_usage['cached_tokens'] > 0:
            print(f"Cached tokens: {token_usage['cached_tokens']:,} (reused, lower cost)")
        print(f"Total tokens: {token_usage['total_tokens']:,}")
        
        # Calculate average tokens per question
        if len(unique_questions) > 0:
            avg_prompt = token_usage['prompt_tokens'] / len(unique_questions)
            avg_completion = token_usage['completion_tokens'] / len(unique_questions)
            avg_total = token_usage['total_tokens'] / len(unique_questions)
            print(f"\nAverage per question:")
            print(f"  Prompt tokens: {avg_prompt:.1f}")
            print(f"  Completion tokens: {avg_completion:.1f}")
            if token_usage['reasoning_tokens'] > 0:
                avg_reasoning = token_usage['reasoning_tokens'] / len(unique_questions)
                print(f"  Reasoning tokens: {avg_reasoning:.1f}")
            if token_usage['cached_tokens'] > 0:
                avg_cached = token_usage['cached_tokens'] / len(unique_questions)
                print(f"  Cached tokens: {avg_cached:.1f}")
            print(f"  Total tokens: {avg_total:.1f}")


if __name__ == "__main__":
    asyncio.run(main())

