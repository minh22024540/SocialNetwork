"""
Test script to evaluate NER models on Vietnamese questions.
Tests multiple models to find one that reliably extracts person names.
"""

from transformers import pipeline
import torch

# Test questions with known person names
test_questions = [
    "Điểm chung nào sau đây giữa quân nhân Nguyễn Văn Tư và ông Trịnh Tố Tâm là đúng?",
    "Phạm Văn Đồng và nhà văn Minh Chuyên đều là những nhân vật có liên quan đến Chiến tranh Việt Nam.",
    "Có phải ông Đặng Xuân Hải, người từng cùng tham gia Chiến tranh Việt Nam?",
    "Tướng Võ Nguyên Giáp và nhà văn Minh Chuyên đều tham gia Chiến tranh Việt Nam.",
    "Nguyễn Văn Bảy (B) và Võ Văn Mẫn có điểm chung gì?",
]

# Expected person names in each question
expected_names = [
    ["Nguyễn Văn Tư", "Trịnh Tố Tâm"],
    ["Phạm Văn Đồng", "Minh Chuyên"],
    ["Đặng Xuân Hải"],
    ["Võ Nguyên Giáp", "Minh Chuyên"],
    ["Nguyễn Văn Bảy", "Võ Văn Mẫn"],
]

# Models to test
models_to_test = [
    "dslim/bert-base-NER",  # English-only, lightweight
    "dslim/bert-base-multilingual-cased-ner",  # Multilingual
    "Babelscape/wikineural-multilingual-ner",  # Multilingual, good for many languages
    "xlm-roberta-base-finetuned-panx-de",  # XLM-RoBERTa multilingual
    "Davlan/xlm-roberta-base-ner-hrl",  # Multilingual NER
    "vinai/phobert-base",  # Vietnamese-specific (but might not be NER)
]

def test_ner_model(model_name: str):
    """Test a single NER model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device,
        )
        print(f"✓ Model loaded successfully on {'cuda' if device >= 0 else 'cpu'}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    total_correct = 0
    total_expected = 0
    
    for i, question in enumerate(test_questions):
        print(f"\nQuestion {i+1}: {question[:60]}...")
        expected = expected_names[i]
        total_expected += len(expected)
        
        try:
            results = ner_pipeline(question)
            print(f"  NER results: {results}")
            
            # Extract person names
            extracted_names = []
            for result in results:
                entity_group = result.get("entity_group", "").upper()
                word = result.get("word", "").strip()
                
                # Check if it's a person entity
                if entity_group in {"PER", "PERSON"} or entity_group.startswith("PER"):
                    # Clean up
                    word = word.replace("##", "").replace("▁", " ").strip()
                    # Remove Vietnamese prefixes
                    prefixes = ["ông", "bà", "nhà văn", "diễn viên", "tướng", "quân nhân", "thiếu tướng", "đại tá"]
                    for prefix in prefixes:
                        if word.lower().startswith(prefix + " "):
                            word = word[len(prefix) + 1:].strip()
                    if word and len(word) > 2:
                        extracted_names.append(word)
            
            print(f"  Extracted names: {extracted_names}")
            print(f"  Expected names: {expected}")
            
            # Check matches
            correct = 0
            for exp_name in expected:
                # Check if any extracted name contains the expected name or vice versa
                exp_lower = exp_name.lower()
                for ext_name in extracted_names:
                    ext_lower = ext_name.lower()
                    if exp_lower in ext_lower or ext_lower in exp_lower:
                        correct += 1
                        break
            
            total_correct += correct
            print(f"  Matches: {correct}/{len(expected)}")
            
        except Exception as e:
            print(f"  ✗ Error processing: {e}")
    
    accuracy = (total_correct / total_expected * 100) if total_expected > 0 else 0
    print(f"\n{'='*60}")
    print(f"Overall accuracy: {total_correct}/{total_expected} ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    return {
        "model": model_name,
        "correct": total_correct,
        "total": total_expected,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    print("Testing NER models on Vietnamese questions...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    results = []
    for model_name in models_to_test:
        result = test_ner_model(model_name)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY - Best performing models:")
    print(f"{'='*60}")
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    for r in results[:3]:
        print(f"{r['model']}: {r['accuracy']:.1f}% ({r['correct']}/{r['total']})")

