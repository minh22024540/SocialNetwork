"""
Small test to verify NER extracts both person names and event names correctly.
"""

from chatbot.ner_extractor import extract_person_names, extract_event_names, extract_all_entity_names

test_questions = [
    "Điểm chung nào sau đây giữa quân nhân Nguyễn Văn Tư và ông Trịnh Tố Tâm là đúng?",
    "Phạm Văn Đồng và nhà văn Minh Chuyên đều là những nhân vật có liên quan đến Chiến tranh Việt Nam.",
    "Có phải ông Đặng Xuân Hải, người từng cùng tham gia Chiến tranh Việt Nam?",
    "Tướng Võ Nguyên Giáp và nhà văn Minh Chuyên đều tham gia Chiến tranh Việt Nam.",
]

print("Testing NER extraction on small sample...\n")

for i, question in enumerate(test_questions, 1):
    print(f"Question {i}: {question[:70]}...")
    
    person_names = extract_person_names(question)
    event_names = extract_event_names(question)
    all_names = extract_all_entity_names(question)
    
    print(f"  Person names: {person_names}")
    print(f"  Event names: {event_names}")
    print(f"  All entities: {all_names}")
    print()

print("✓ NER extraction test completed")

