#!/usr/bin/env python3
"""
Script to parse the first 5 pages from a Vietnamese Wikipedia dump file.
"""

import mwxml
import bz2
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def check_page_patterns(text_content):
    """
    Simplified page-level pattern check - just check for keyword presence.
    Returns tuple: (has_people_pattern, has_historical_pattern, matched_patterns)
    """
    import re

    matched_patterns = []

    # Normalize content to lowercase for consistent matching
    content_lower = text_content.lower()

    # Pattern 1: People pattern - just check for birth/death keywords
    english_birth_keywords = ['birth =', 'birth=', 'born =', 'born=', 'birthdate =', 'birthdate=', 'birth_date =', 'birth_date=']
    english_death_keywords = ['death =', 'death=', 'died =', 'died=', 'deathdate =', 'deathdate=', 'death_date =', 'death_date=']
    vietnamese_birth_keywords = ['sinh =', 'sinh=', 'sinh năm =', 'sinh năm=', 'ngày sinh =', 'ngày sinh=', 'ngày ra đời =', 'ngày ra đời=']
    vietnamese_death_keywords = ['mất =', 'mất=', 'mất năm =', 'mất năm=', 'ngày mất =', 'ngày mất=', 'qua đời =', 'qua đời=']

    has_english_birth = any(keyword in content_lower for keyword in english_birth_keywords)
    has_english_death = any(keyword in content_lower for keyword in english_death_keywords)
    has_vietnamese_birth = any(keyword in content_lower for keyword in vietnamese_birth_keywords)
    has_vietnamese_death = any(keyword in content_lower for keyword in vietnamese_death_keywords)

    if has_english_birth and has_english_death:
        matched_patterns.append("people_english_pattern")
    if has_vietnamese_birth and has_vietnamese_death:
        matched_patterns.append("people_vietnamese_pattern")
    if (has_english_birth or has_vietnamese_birth) and (has_english_death or has_vietnamese_death):
        matched_patterns.append("people_mixed_language")

    # Pattern 2: Historical events pattern - just check for time/date keywords
    english_time_keywords = ['date =', 'date=', 'time =', 'time=', 'when =', 'when=', 'period =', 'period=', 'duration =', 'duration=', 'start =', 'start=', 'end =', 'end=']
    vietnamese_time_keywords = ['thời gian =', 'thời gian=', 'ngày =', 'ngày=', 'thời kỳ =', 'thời kỳ=', 'khoảng thời gian =', 'khoảng thời gian=', 'bắt đầu =', 'bắt đầu=', 'kết thúc =', 'kết thúc=', 'từ =', 'từ=', 'đến =', 'đến=']

    has_english_time = any(keyword in content_lower for keyword in english_time_keywords)
    has_vietnamese_time = any(keyword in content_lower for keyword in vietnamese_time_keywords)

    if has_english_time:
        matched_patterns.append("historical_english_pattern")
    if has_vietnamese_time:
        matched_patterns.append("historical_vietnamese_pattern")

    has_people = any("people" in p for p in matched_patterns)
    has_historical = any("historical" in p for p in matched_patterns)

    return has_people, has_historical, matched_patterns

def extract_all_templates(text_content):
    """
    Extract templates that match specific patterns:
    1. People pattern: sinh=value|mất=value (with possible newlines/spaces)
    2. Historical events pattern: date=..... (like date = 3 tháng 11 năm 1840...)
    """
    import mwparserfromhell
    import re
    
    infobox_info = {
        "found": False,
        "template_name": None,
        "first_line": None,
        "line_count": 0,
        "sample_content": [],
        "parameter_count": 0,
        "confidence_score": 0,
        "section_number": None,
        "section_name": None,
        "position_in_section": None,
        "pattern_type": None,  # "people" or "historical" or "other"
        "matched_patterns": []  # List of matched patterns found
    }
    
    # Parse the wikitext using mwparserfromhell
    try:
        wikicode = mwparserfromhell.parse(text_content)
    except Exception as e:
        # If parsing fails, return empty result
        return infobox_info
    
    # Extract sections and their positions
    sections = []
    lines = text_content.split('\n')
    current_section = 0
    current_section_name = "Introduction"
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Check for section headers (== Header ==)
        if line_stripped.startswith('==') and line_stripped.endswith('=='):
            # Extract section name
            section_name = line_stripped[2:-2].strip()
            if section_name:  # Only count non-empty section names
                current_section += 1
                current_section_name = section_name
                sections.append({
                    'number': current_section,
                    'name': section_name,
                    'start_line': i,
                    'end_line': len(lines)  # Will be updated when next section is found
                })
    
    # Update end lines for sections
    for i in range(len(sections)):
        if i < len(sections) - 1:
            sections[i]['end_line'] = sections[i + 1]['start_line']
    
    def check_patterns(template_content):
        """
        Check if template content matches our target patterns.
        Simplified to just check for keyword presence, not specific values or formats.
        Returns tuple: (has_people_pattern, has_historical_pattern, matched_patterns)
        """
        matched_patterns = []

        # Normalize template content to lowercase for consistent matching
        template_lower = template_content.lower()

        # Pattern 1: People pattern - just check for birth/death keywords
        english_birth_keywords = ['birth =', 'birth=', 'born =', 'born=', 'birthdate =', 'birthdate=', 'birth_date =', 'birth_date=']
        english_death_keywords = ['death =', 'death=', 'died =', 'died=', 'deathdate =', 'deathdate=', 'death_date =', 'death_date=']
        vietnamese_birth_keywords = ['sinh =', 'sinh=', 'sinh năm =', 'sinh năm=', 'ngày sinh =', 'ngày sinh=', 'ngày ra đời =', 'ngày ra đời=']
        vietnamese_death_keywords = ['mất =', 'mất=', 'mất năm =', 'mất năm=', 'ngày mất =', 'ngày mất=', 'qua đời =', 'qua đời=']

        has_english_birth = any(keyword in template_lower for keyword in english_birth_keywords)
        has_english_death = any(keyword in template_lower for keyword in english_death_keywords)
        has_vietnamese_birth = any(keyword in template_lower for keyword in vietnamese_birth_keywords)
        has_vietnamese_death = any(keyword in template_lower for keyword in vietnamese_death_keywords)

        if has_english_birth and has_english_death:
            matched_patterns.append("people_english_pattern")
        if has_vietnamese_birth and has_vietnamese_death:
            matched_patterns.append("people_vietnamese_pattern")
        if (has_english_birth or has_vietnamese_birth) and (has_english_death or has_vietnamese_death):
            matched_patterns.append("people_mixed_language")

        # Pattern 2: Historical events pattern - just check for time/date keywords
        english_time_keywords = ['date =', 'date=', 'time =', 'time=', 'when =', 'when=', 'period =', 'period=', 'duration =', 'duration=', 'start =', 'start=', 'end =', 'end=']
        vietnamese_time_keywords = ['thời gian =', 'thời gian=', 'ngày =', 'ngày=', 'thời kỳ =', 'thời kỳ=', 'khoảng thời gian =', 'khoảng thời gian=', 'bắt đầu =', 'bắt đầu=', 'kết thúc =', 'kết thúc=', 'từ =', 'từ=', 'đến =', 'đến=']

        has_english_time = any(keyword in template_lower for keyword in english_time_keywords)
        has_vietnamese_time = any(keyword in template_lower for keyword in vietnamese_time_keywords)

        if has_english_time:
            matched_patterns.append("historical_english_pattern")
        if has_vietnamese_time:
            matched_patterns.append("historical_vietnamese_pattern")

        has_people = any("people" in p for p in matched_patterns)
        has_historical = any("historical" in p for p in matched_patterns)

        return has_people, has_historical, matched_patterns

    templates_found = []
    
    # Extract ALL templates and check for patterns
    for template in wikicode.filter_templates():
        template_name = str(template.name).strip()
        template_lower = template_name.lower()
        
        # Count parameters (excluding the template name)
        parameter_count = len(template.params)
        
        # Get template content as string
        template_content = str(template)
        template_lines = template_content.split('\n')
        
        # Check if this template matches our target patterns
        has_people, has_historical, matched_patterns = check_patterns(template_content)
        
        # Only process templates that match our patterns
        if not (has_people or has_historical):
            continue
        
        # Find which section this template belongs to
        # Get the position of the template in the original text
        template_start_pos = text_content.find(template_content)
        template_start_line = text_content[:template_start_pos].count('\n')
        
        # Determine section information
        section_number = 0  # Introduction section
        section_name = "Introduction"
        position_in_section = "beginning"
        
        # Find which section contains this template
        for section in sections:
            if section['start_line'] <= template_start_line < section['end_line']:
                section_number = section['number']
                section_name = section['name']
                # Calculate position within section
                lines_in_section = section['end_line'] - section['start_line']
                template_position_in_section = template_start_line - section['start_line']
                if template_position_in_section < lines_in_section * 0.3:
                    position_in_section = "beginning"
                elif template_position_in_section < lines_in_section * 0.7:
                    position_in_section = "middle"
                else:
                    position_in_section = "end"
                break
        
        # Determine pattern type with language information
        pattern_type = "other"
        if has_people and has_historical:
            pattern_type = "mixed"
        elif has_people:
            # Check language for people patterns
            if any("english" in p for p in matched_patterns) and any("vietnamese" in p for p in matched_patterns):
                pattern_type = "people_mixed_language"
            elif any("english" in p for p in matched_patterns):
                pattern_type = "people_english"
            elif any("vietnamese" in p for p in matched_patterns):
                pattern_type = "people_vietnamese"
            else:
                pattern_type = "people"
        elif has_historical:
            # Check language for historical patterns
            if any("english" in p for p in matched_patterns) and any("vietnamese" in p for p in matched_patterns):
                pattern_type = "historical_mixed_language"
            elif any("english" in p for p in matched_patterns):
                pattern_type = "historical_english"
            elif any("vietnamese" in p for p in matched_patterns):
                pattern_type = "historical_vietnamese"
            else:
                pattern_type = "historical"
        
        templates_found.append({
            'name': template_name,
            'name_lower': template_lower,
            'full_text': template_content,
            'parameter_count': parameter_count,
            'line_count': len(template_lines),
            'lines': template_lines,
            'first_line': template_lines[0] if template_lines else '',
            'section_number': section_number,
            'section_name': section_name,
            'position_in_section': position_in_section,
            'start_line': template_start_line,
            'pattern_type': pattern_type,
            'matched_patterns': matched_patterns
        })
    
    # Filter out templates without proper structure (name + attributes)
    # Templates must have at least 1 parameter to be considered meaningful
    meaningful_templates = [t for t in templates_found if t['parameter_count'] > 0]
    
    # If we found meaningful templates, prioritize by pattern type and section
    if meaningful_templates:
        # Sort templates by priority: pattern type, section, line count, parameter count
        def template_priority(t):
            # Priority 1: Pattern type (people > historical > mixed > other)
            # Language-specific patterns get higher priority
            pattern_priority_map = {
                "people_mixed_language": 6,
                "people_english": 5,
                "people_vietnamese": 5,
                "people": 4,
                "historical_mixed_language": 5,
                "historical_english": 4,
                "historical_vietnamese": 4,
                "historical": 3,
                "mixed": 2,
                "other": 1
            }
            pattern_priority = pattern_priority_map.get(t['pattern_type'], 0)
            
            # Priority 2: Section (Section 0 is most important)
            section_priority = 2 if t['section_number'] == 0 else 1
            
            # Priority 3: Line count (multi-line is better)
            line_priority = 2 if t['line_count'] > 1 else 1
            
            # Priority 4: Parameter count
            param_priority = min(t['parameter_count'] // 5, 3)  # Cap at 3
            
            return (pattern_priority, section_priority, line_priority, param_priority)
        
        meaningful_templates.sort(key=template_priority, reverse=True)
        
        # Select the best template
        best_template = meaningful_templates[0]
        
        if best_template:
            # Calculate confidence based on pattern matching, structure, and section
            confidence = 0
            
            # Factor 1: Pattern matching (highest priority)
            pattern_confidence_map = {
                "people_mixed_language": 12,  # Highest - both languages
                "people_english": 10,         # High - English people
                "people_vietnamese": 10,      # High - Vietnamese people
                "people": 8,                  # Medium - generic people
                "historical_mixed_language": 10,  # High - both languages
                "historical_english": 8,      # Medium - English historical
                "historical_vietnamese": 8,   # Medium - Vietnamese historical
                "historical": 6,              # Medium - generic historical
                "mixed": 9,                   # High - mixed patterns
                "other": 3                    # Low - other patterns
            }
            confidence += pattern_confidence_map.get(best_template['pattern_type'], 3)
            
            # Factor 2: Number of matched patterns
            confidence += len(best_template['matched_patterns']) * 2
            
            # Factor 3: Section priority (Section 0 is most important)
            if best_template['section_number'] == 0:
                confidence += 3  # Section 0 templates are main infoboxes
            else:
                confidence += 1  # Other sections are specialized templates
            
            # Factor 4: Multi-line structure (strong indicator of infobox)
            if best_template['line_count'] > 1:
                confidence += 5  # Multi-line templates are very likely infoboxes
            else:
                confidence += 1  # Single-line templates are less likely
            
            # Factor 5: Template name patterns (less important now)
            if any(keyword in best_template['name_lower'] for keyword in 
                   ['infobox', 'hộp thông tin', 'thông tin', 'box', 'info']):
                confidence += 2
            elif any(keyword in best_template['name_lower'] for keyword in 
                     ['đơn vị hành chính', 'bộ', 'tỉnh', 'thành phố', 'tóm tắt', 'công ty']):
                confidence += 1
            
            # Factor 6: Parameter count (more parameters = more likely infobox)
            if best_template['parameter_count'] >= 10:
                confidence += 2
            elif best_template['parameter_count'] >= 5:
                confidence += 1
            
            # Factor 7: Exclude citation templates (false positives)
            if any(citation_word in best_template['name_lower'] for citation_word in 
                   ['chú thích', 'citation', 'ref', 'web', 'sách', 'báo']):
                confidence -= 2  # Reduce confidence for citation templates
            
            infobox_info["found"] = True
            infobox_info["template_name"] = best_template['name']
            infobox_info["first_line"] = best_template['first_line']
            infobox_info["line_count"] = best_template['line_count']
            infobox_info["parameter_count"] = best_template['parameter_count']
            infobox_info["confidence_score"] = confidence
            infobox_info["sample_content"] = best_template['lines'][:5]
            infobox_info["section_number"] = best_template['section_number']
            infobox_info["section_name"] = best_template['section_name']
            infobox_info["position_in_section"] = best_template['position_in_section']
            infobox_info["pattern_type"] = best_template['pattern_type']
            infobox_info["matched_patterns"] = best_template['matched_patterns']
    
    return infobox_info

def process_single_page(page_data):
    """Process a single Wikipedia page and return result if it matches criteria.

    Args:
        page_data: Tuple of (page, text_content) from mwxml parser.

    Returns:
        Dictionary with page data if it matches patterns, None otherwise.
        Compatible with multiprocessing.Pool.map().
    """
    page, text_content = page_data
    
    # Skip pages that are not in main namespace (0) or have redirects
    if page.namespace != 0 or page.redirect:
        return None
    
    # First, attempt to extract infobox to determine category
    infobox_attempt = extract_all_templates(text_content)
    
    # Determine category based on infobox template first
    page_category = "unknown"
    page_pattern_type = "other"
    
    if infobox_attempt["found"]:
        pattern_type = infobox_attempt.get("pattern_type", "")

        # Check based on pattern type from template analysis
        if "people" in pattern_type:
            page_category = "people"
            page_pattern_type = pattern_type
        elif "historical" in pattern_type:
            page_category = "event"
            page_pattern_type = pattern_type
        else:
            # If infobox found but doesn't match people or historical patterns, skip
            return None
    else:
        # No infobox found, check patterns as fallback
        has_people, has_historical, page_patterns = check_page_patterns(text_content)
        
        # Skip pages that don't match any of our target patterns
        if not (has_people or has_historical):
            return None
        
        # Only categorize as event if it has strong historical event patterns
        # AND no organization indicators
        if has_historical and not has_people:
            # Additional check: look for organization indicators in content
            content_lower = text_content.lower()
            has_org_indicators = any(org_indicator in content_lower for org_indicator in 
                                   ["tổ chức", "organization", "society", "association", 
                                    "institution", "company", "corporation", "foundation"])
            
            if not has_org_indicators:
                page_category = "event"
                if any("english" in p for p in page_patterns) and any("vietnamese" in p for p in page_patterns):
                    page_pattern_type = "historical_mixed_language"
                elif any("english" in p for p in page_patterns):
                    page_pattern_type = "historical_english"
                elif any("vietnamese" in p for p in page_patterns):
                    page_pattern_type = "historical_vietnamese"
                else:
                    page_pattern_type = "historical"
            else:
                # Has organization indicators, skip
                return None
        elif has_people and not has_historical:
            page_category = "people"
            if any("english" in p for p in page_patterns) and any("vietnamese" in p for p in page_patterns):
                page_pattern_type = "people_mixed_language"
            elif any("english" in p for p in page_patterns):
                page_pattern_type = "people_english"
            elif any("vietnamese" in p for p in page_patterns):
                page_pattern_type = "people_vietnamese"
            else:
                page_pattern_type = "people"
        else:
            # Mixed patterns without clear infobox, skip
            return None
    
    # Extract categories from the end of the page
    import re
    category_pattern = re.compile(r'\[\[Thể loại:([^\]]+)\]\]', re.IGNORECASE)
    categories = category_pattern.findall(text_content)

    # Check if page is Vietnam-related
    vietnam_keywords = ['việt nam', 'viet nam', 'vietnam', 'Việt Nam', 'Viet Nam', 'Vietnam']
    is_vietnam_related = any(
        any(vietnam_keyword.lower() in category.lower() for vietnam_keyword in vietnam_keywords)
        for category in categories
    )

    # Skip pages that are not Vietnam-related
    if not is_vietnam_related:
        return None

    # Create preview (first 200 characters)
    preview = text_content[:200] + "..." if len(text_content) > 200 else text_content

    # Page data for both outputs
    page_data = {
        "id": page.id,
        "title": page.title,
        "namespace": page.namespace,
        "redirect": page.redirect,
        "page_category": page_category,
        "page_pattern_type": page_pattern_type,
        "page_patterns": page_patterns if 'page_patterns' in locals() else [],
        "categories": categories,
        "is_vietnam_related": is_vietnam_related,
        "infobox_attempt": infobox_attempt
    }
    
    # Add preview data
    preview_page = page_data.copy()
    preview_page["preview"] = preview
    preview_data = preview_page
    
    # Add full text data
    full_text_page = page_data.copy()
    full_text_page["full_text"] = text_content
    full_text_data = full_text_page
    
    return preview_data, full_text_data

def parse_all_pages(dump_file_path):
    """
    Parse all pages from a Wikipedia dump file to extract people and events.
    
    Args:
        dump_file_path (str): Path to the .bz2 Wikipedia dump file
    """
    
    print(f"Parsing all pages from: {dump_file_path}")
    print("=" * 60)
    
    # Initialize data structures for JSON output
    preview_data = []
    full_text_data = []
    
    try:
        # Open the compressed dump file
        with bz2.open(dump_file_path, 'rt', encoding='utf-8') as f:
            dump = mwxml.Dump.from_file(f)
            
            page_count = 0
            skipped_redirects = 0
            skipped_no_match = 0
            total_processed = 0
            batch_size = 10000  # Process pages in batches
            num_processes = min(mp.cpu_count(), 16)  # Use up to 16 cores
            
            print(f"Using {num_processes} processes for parallel processing...")
            print("Processing pages in batches...")
            
            # Print page attributes for first page only
            first_page_printed = False
            
            with mp.Pool(processes=num_processes) as pool:
                batch = []
                
                with tqdm(desc="Processing pages", unit="page") as pbar:
                    for page in dump:
                        total_processed += 1
                        
                        # Skip pages that are not in main namespace (0) or have redirects
                        if page.namespace != 0 or page.redirect:
                            skipped_redirects += 1
                            pbar.set_postfix({"Skipped": skipped_redirects, "Found": page_count, "No Match": skipped_no_match})
                            continue
                        
                        # Print page attributes for first page only
                        if not first_page_printed:
                            print("All page attributes:")
                            for attr in dir(page):
                                if not attr.startswith('_'):
                                    try:
                                        value = getattr(page, attr)
                                        print(f"  {attr}: {value}")
                                    except Exception as e:
                                        print(f"  {attr}: <Error: {e}>")
                            print("\n" + "="*60)
                            first_page_printed = True
                        
                        # Get the latest revision and full text
                        for revision in page:
                            text_content = revision.text or ""
                            batch.append((page, text_content))
                            break  # Only get the latest revision
                        
                        # Process batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            # Process the batch in parallel
                            batch_found = 0
                            for result in pool.imap(process_single_page, batch, chunksize=50):
                                if result is not None:
                                    preview_page, full_text_page = result
                                    preview_data.append(preview_page)
                                    full_text_data.append(full_text_page)
                                    page_count += 1
                                    batch_found += 1
                                pbar.update(1)
                            
                            skipped_no_match += len(batch) - batch_found
                            batch = []  # Clear the batch
                            pbar.set_postfix({"Skipped": skipped_redirects, "Found": page_count, "No Match": skipped_no_match, "Current": page.title[:30]})
                    
                    # Process remaining pages in the last batch
                    if batch:
                        batch_found = 0
                        for result in pool.imap(process_single_page, batch, chunksize=50):
                            if result is not None:
                                preview_page, full_text_page = result
                                preview_data.append(preview_page)
                                full_text_data.append(full_text_page)
                                page_count += 1
                                batch_found += 1
                            pbar.update(1)
                        
                        skipped_no_match += len(batch) - batch_found
                        pbar.set_postfix({"Skipped": skipped_redirects, "Found": page_count, "No Match": skipped_no_match})
                
    except FileNotFoundError:
        print(f"Error: File '{dump_file_path}' not found.")
        return False
    except Exception as e:
        print(f"Error parsing dump file: {e}")
        return False
    
    # Save to JSON files
    with open("wikipedia_preview.json", 'w', encoding='utf-8') as json_file:
        json.dump(preview_data, json_file, ensure_ascii=False, indent=2)
    
    with open("wikipedia_full_text.json", 'w', encoding='utf-8') as json_file:
        json.dump(full_text_data, json_file, ensure_ascii=False, indent=2)
    
    # Calculate pattern statistics
    page_category_stats = {}
    page_pattern_stats = {}
    template_pattern_stats = {}
    
    for page_data in preview_data:
        # Page category statistics
        page_category = page_data.get("page_category", "unknown")
        page_category_stats[page_category] = page_category_stats.get(page_category, 0) + 1
        
        # Page-level pattern statistics
        page_pattern_type = page_data.get("page_pattern_type", "unknown")
        page_pattern_stats[page_pattern_type] = page_pattern_stats.get(page_pattern_type, 0) + 1
        
        # Template-level pattern statistics
        infobox = page_data.get("infobox_attempt", {})
        if infobox.get("found"):
            template_pattern_type = infobox.get("pattern_type", "unknown")
            template_pattern_stats[template_pattern_type] = template_pattern_stats.get(template_pattern_type, 0) + 1
    
    print(f"\n{'=' * 60}")
    print(f"Successfully processed {page_count} pages with people or event content.")
    print(f"Skipped {skipped_redirects} pages (redirects/non-main namespace).")
    print(f"Skipped {skipped_no_match} pages (no matching patterns).")
    print(f"Total pages processed: {total_processed}")
    print(f"Math check: {skipped_redirects} + {skipped_no_match} + {page_count} = {skipped_redirects + skipped_no_match + page_count}")
    print(f"Preview data saved to: wikipedia_preview.json")
    print(f"Full text data saved to: wikipedia_full_text.json")
    
    if page_category_stats:
        print(f"\nPage Category Statistics:")
        for category, count in sorted(page_category_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} pages")
    
    if page_pattern_stats:
        print(f"\nPage-Level Pattern Statistics:")
        for pattern_type, count in sorted(page_pattern_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern_type}: {count} pages")
    
    if template_pattern_stats:
        print(f"\nTemplate-Level Pattern Statistics:")
        for pattern_type, count in sorted(template_pattern_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern_type}: {count} pages")
    
    return True

def main() -> None:
    """Main function to run the Wikipedia parser.

    Parses Wikipedia XML dump file to extract entities (persons and events)
    and saves them to JSONL format for further processing in the pipeline.
    """
    
    # Check if dump file exists
    dump_file = Path("viwiki_data/viwiki-20251020-pages-articles.xml.bz2")
    
    if not dump_file.exists():
        print(f"Error: Dump file not found at {dump_file}")
        print("Please make sure the file exists in the viwiki_data folder.")
        return
    
    # Parse all pages to extract people and events
    success = parse_all_pages(str(dump_file))
    
    if success:
        print("\nParsing completed successfully!")
    else:
        print("\nParsing failed!")

if __name__ == "__main__":
    main()
