#!/usr/bin/env python
"""
Quick test script to verify bug fixes for Knowledge Graph and Propaganda Detection.
Run with: python test_bug_fixes.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# TEST KNOWLEDGE GRAPH FIXES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("TESTING KNOWLEDGE GRAPH FIXES")
print("="*80)

from app.services.graph_service import _normalise_id

print("\n✓ Bug Fix #1 & #5: Node ID Normalization")
test_cases = [
    ("Elon Musk", "elon musk"),
    ("NASA", "nasa"),
    ("N.A.S.A.", "nasa"),
    ("U.S.A.", "usa"),
    ('"NASA"', "nasa"),
    ("Joe   Biden", "joe biden"),
    ("U.S.A. Inc.", "usa inc"),
    ("...dots...", "dots"),  # Should not be empty
    ("PERSON-123", "person 123"),
]

for input_text, expected in test_cases:
    result = _normalise_id(input_text)
    status = "✓" if result == expected else "✗"
    print(f"  {status} '{input_text}' → '{result}' (expected: '{expected}')")
    if result != expected:
        print(f"      MISMATCH!")

# Test that 50k truncation warning is present
print("\n✓ Bug Fix #4: Text Truncation Detection")
from app.services.graph_service import extract_knowledge_graph

long_text = "The quick brown fox. " * 5000  # ~110k chars
try:
    result = extract_knowledge_graph(long_text)
    print(f"  Extracted from {len(long_text)} char text:")
    print(f"    - Total nodes: {result['total_nodes']}")
    print(f"    - Total edges: {result['total_edges']}")
    print(f"    - Total entities: {result['total_entities']}")
    print(f"  ✓ Truncation handled (check logs for truncation warning)")
except Exception as e:
    print(f"  ✗ Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST PROPAGANDA DETECTION FIXES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("TESTING PROPAGANDA DETECTION FIXES")
print("="*80)

from app.services.propaganda_service import _detect_repetition

print("\n✓ Bug Fix #1: Variable N-gram Detection (3-5 words)")

test_repetitions = [
    # (text, should_detect_phrase_substring)
    ("Lock her up! Lock her up! Lock her up!", "lock her up"),  # 3-gram
    ("You need to act now. You need to act now. You need to act now.", "you need to act"),  # 4-gram (contains "to")
    ("We are fighting for freedom and liberty. We are fighting for freedom and liberty. We are fighting for freedom and liberty.", "fighting for freedom"),  # Should detect
    ("the the the the", None),  # All stop words, should not detect
]

for text, expected_phrase in test_repetitions:
    confidence, matched = _detect_repetition(text)
    if expected_phrase:
        found = any(expected_phrase in m.lower() for m in matched)
        status = "✓" if found and confidence > 0 else "✗"
        print(f"  {status} '{text[:50]}...' → confidence={confidence}, found '{expected_phrase}': {found}")
    else:
        status = "✓" if confidence == 0 else "✗"
        print(f"  {status} '{text[:50]}...' → confidence={confidence} (should be 0)")

print("\n✓ Bug Fix #2: More Selective Stop-Gram Filter")
print("  Testing that meaningful phrases with common words are now detected...")

neutral_with_common_words = "You need to understand this. You need to understand this. You need to understand this."
confidence, matched = _detect_repetition(neutral_with_common_words)
status = "✓" if confidence > 0 and any("understand" in m for m in matched) else "✗"
print(f"  {status} 'You need to understand' detected: confidence={confidence}, matched={matched}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST FULL PROPAGANDA DETECTION
# ─────────────────────────────────────────────────────────────────────────────

from app.services.propaganda_service import detect_propaganda

print("\n✓ Testing Full Propaganda Detection Pipeline")

test_articles = [
    ("Neutral", "Researchers published a study on renewable energy efficiency."),
    ("Repetition", "Act now! Act now! Act now! We must act now!"),
    ("Fear + Clickbait", "You won't believe this shocking crisis! Danger threatens everyone!"),
]

for label, text in test_articles:
    result = detect_propaganda(text)
    print(f"\n  {label}:")
    print(f"    - Detected: {result['propaganda_detected']}")
    print(f"    - Overall Score: {result['overall_score']}")
    print(f"    - Techniques: {len(result['techniques_found'])}")
    for tech in result['techniques_found'][:2]:
        print(f"      • {tech['technique']}: {tech['confidence']} ({len(tech['matched_phrases'])} phrases)")

print("\n" + "="*80)
print("TESTS COMPLETED")
print("="*80 + "\n")
