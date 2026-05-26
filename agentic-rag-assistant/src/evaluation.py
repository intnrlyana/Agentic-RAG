"""Golden-set evaluation helpers for demo-quality review."""

from __future__ import annotations

import re
from typing import Any


TestCase = dict[str, Any]
TestResult = dict[str, Any]


def get_default_test_cases() -> list[TestCase]:
    """Return predefined evaluation cases with expected facts and source pages."""
    return [
        {
            "test_name": "Annual Leave Entitlement",
            "question": "How many annual leave days does an employee get?",
            "expected_behavior": "Return the annual leave entitlement table with the leave section cited.",
            "test_type": "factual",
            "expected_terms": ["annual", "leave", "days"],
            "should_fallback": False,
            "expected_pages": [13],
            "expected_facts": [
                {"label": "0 to <2 years = 8 days", "any_of": ["8", "eight"]},
                {"label": "2 to <5 years = 12 days", "any_of": ["12", "twelve"]},
                {"label": "above 5 years = 16 days", "any_of": ["16", "sixteen"]},
            ],
        },
        {
            "test_name": "Carry Forward Leave",
            "question": "Can annual leave be carried forward?",
            "expected_behavior": "State the maximum carry-forward days and forfeiture deadline.",
            "test_type": "policy_fact",
            "expected_terms": ["carry", "forward", "leave"],
            "should_fallback": False,
            "expected_pages": [13],
            "expected_facts": [
                {"label": "maximum 5 days", "any_of": ["five (5) days", "five days", "5 days"]},
                {
                    "label": "must be cleared by first quarter",
                    "any_of": ["first quarter", "end of the first quarter"],
                },
                {"label": "unused leave is forfeited", "any_of": ["forfeited", "forfeit"]},
            ],
        },
        {
            "test_name": "Public Holiday Entitlement",
            "question": "How many public holidays are employees entitled to?",
            "expected_behavior": "Return the public holiday entitlement with the public holiday section cited.",
            "test_type": "factual",
            "expected_terms": ["public", "holidays"],
            "should_fallback": False,
            "expected_pages": [16],
            "expected_facts": [
                {"label": "15 public holidays", "any_of": ["fifteen", "15 public holidays", "15"]},
            ],
        },
        {
            "test_name": "Working Hours Or Attendance Rules",
            "question": "Summarize the working hours or attendance rules.",
            "expected_behavior": "Summarize explicit attendance, work-hour, and no-pay-leave rules without mixing unrelated sections.",
            "test_type": "summary",
            "expected_terms": ["working", "hours", "attendance"],
            "should_fallback": False,
            "expected_pages": [8, 13, 18],
            "expected_facts": [
                {
                    "label": "report at assigned or scheduled hours",
                    "any_of": ["assigned/scheduled work hours", "assigned work hours", "scheduled work hours"],
                },
                {"label": "one rest day every seven days", "any_of": ["one rest day every seven", "rest day every seven"]},
                {"label": "absence without approval leads to no-pay leave", "any_of": ["no-pay leave", "disciplinary actions"]},
            ],
        },
        {
            "test_name": "Resignation Notice Period",
            "question": "What is the resignation notice period?",
            "expected_behavior": "State the notice period during probation and after confirmation.",
            "test_type": "factual",
            "expected_terms": ["resignation", "notice", "period"],
            "should_fallback": False,
            "expected_pages": [20],
            "expected_facts": [
                {"label": "2 weeks during probation", "any_of": ["two (2) weeks", "2 weeks", "two weeks"]},
                {"label": "4 weeks upon confirmation", "any_of": ["four (4) weeks", "4 weeks", "four weeks"]},
            ],
        },
        {
            "test_name": "Resignation Rules Summary",
            "question": "Summarize the resignation rules.",
            "expected_behavior": "Cover notice, separation form, final wages, clearance, and exit interview accurately.",
            "test_type": "summary",
            "expected_terms": ["resignation", "rules"],
            "should_fallback": False,
            "expected_pages": [20, 21, 22],
            "expected_facts": [
                {"label": "notice period", "any_of": ["2 weeks", "4 weeks", "two weeks", "four weeks"]},
                {"label": "separation form required", "any_of": ["separation form"]},
                {"label": "verbal resignation not accepted", "any_of": ["verbal resignation will not be entertained", "verbal resignation"]},
                {"label": "final wages include unused earned leave", "any_of": ["final wages", "encashment of unutilised earned leave", "unutilised earned leave"]},
                {"label": "clearance or exit interview", "any_of": ["clearance process", "exit interview"]},
            ],
        },
        {
            "test_name": "Leave And Resignation Summary",
            "question": "Summarize the leave and resignation rules.",
            "expected_behavior": "Cover both leave and resignation topics with explicit rules from the handbook.",
            "test_type": "multi_topic_summary",
            "expected_terms": ["leave", "resignation"],
            "should_fallback": False,
            "expected_pages": [13, 14, 15, 16, 17, 20, 21, 22],
            "expected_facts": [
                {"label": "annual leave entitlement", "any_of": ["annual leave", "entitled to paid annual leave"]},
                {"label": "carry forward rule", "any_of": ["carry forward", "5 days"]},
                {"label": "notice period", "any_of": ["2 weeks", "4 weeks", "two weeks", "four weeks"]},
                {"label": "final wages or unused leave", "any_of": ["final wages", "encashment", "unutilised earned leave"]},
            ],
        },
        {
            "test_name": "Unavailable Information Test",
            "question": "What is the moon allowance policy?",
            "expected_behavior": "Return the fallback sentence instead of inventing unsupported information.",
            "test_type": "hallucination_prevention",
            "expected_terms": ["moon", "allowance"],
            "should_fallback": True,
            "expected_pages": [],
            "expected_facts": [],
        },
    ]
def summarize_test_results(results: list[TestResult]) -> dict[str, Any]:
    """Summarize high-level evaluation metrics for the UI."""
    if not results:
        return {
            "total_tests": 0,
            "fallback_count": 0,
            "average_latency": 0.0,
            "tests_with_citations": 0,
            "tests_with_retry": 0,
            "average_groundedness": 0.0,
            "retrieval_hit_rate": 0.0,
            "average_fact_coverage": 0.0,
            "average_page_coverage": 0.0,
        }

    total_tests = len(results)
    fallback_count = sum(1 for result in results if result["fallback_triggered"])
    average_latency = round(sum(result["latency_seconds"] for result in results) / total_tests, 2)
    tests_with_citations = sum(1 for result in results if result["citations"])
    tests_with_retry = sum(1 for result in results if result["retrieval_attempts"] > 1)
    metrics_results = [result["metrics"] for result in results if isinstance(result.get("metrics"), dict)]

    average_groundedness = round(
        sum(metrics.get("groundedness_score", 0.0) for metrics in metrics_results) / max(1, len(metrics_results)),
        2,
    )
    retrieval_hit_rate = round(
        sum(metrics.get("retrieval_hit", 0.0) for metrics in metrics_results) / max(1, len(metrics_results)),
        2,
    )
    average_fact_coverage = round(
        sum(metrics.get("fact_coverage", 0.0) for metrics in metrics_results) / max(1, len(metrics_results)),
        2,
    )
    average_page_coverage = round(
        sum(metrics.get("page_coverage", 0.0) for metrics in metrics_results) / max(1, len(metrics_results)),
        2,
    )

    return {
        "total_tests": total_tests,
        "fallback_count": fallback_count,
        "average_latency": average_latency,
        "tests_with_citations": tests_with_citations,
        "tests_with_retry": tests_with_retry,
        "average_groundedness": average_groundedness,
        "retrieval_hit_rate": retrieval_hit_rate,
        "average_fact_coverage": average_fact_coverage,
        "average_page_coverage": average_page_coverage,
    }


def score_test_result(test_case: TestCase, pipeline_result: dict[str, Any]) -> dict[str, Any]:
    """Compute lightweight quality metrics with expected-fact coverage."""
    expected_terms = [term.lower() for term in test_case.get("expected_terms", [])]
    retrieved_chunks = pipeline_result.get("retrieved_chunks", [])
    top_chunk_text = " ".join(chunk.get("chunk_text", "") for chunk in retrieved_chunks[:4]).lower()
    retrieved_heading_text = " ".join(
        " ".join(
            value
            for value in [
                chunk.get("section_title", ""),
                chunk.get("subsection_title", ""),
                chunk.get("heading_path", ""),
            ]
            if value
        )
        for chunk in retrieved_chunks[:4]
    ).lower()

    retrieval_hit = int(any(term in top_chunk_text or term in retrieved_heading_text for term in expected_terms))
    answer_text = _normalize_text(str(pipeline_result.get("final_answer", "")))
    citations_present = int(bool(pipeline_result.get("citations")))
    verifier = pipeline_result.get("answer_verification", {})
    groundedness_score = 1.0 if verifier.get("verified", True) else 0.0
    fallback_correct = int(bool(pipeline_result.get("fallback_triggered")) == bool(test_case.get("should_fallback")))

    fact_checks = test_case.get("expected_facts", [])
    fact_hits: list[dict[str, Any]] = []
    for fact in fact_checks:
        variants = [_normalize_text(text) for text in fact.get("any_of", [])]
        matched = any(variant and variant in answer_text for variant in variants)
        fact_hits.append({"label": fact.get("label", ""), "matched": matched, "any_of": fact.get("any_of", [])})
    fact_coverage = round(sum(1 for item in fact_hits if item["matched"]) / max(1, len(fact_hits)), 2) if fact_hits else 0.0

    expected_pages = set(int(page) for page in test_case.get("expected_pages", []))
    cited_pages = set()
    for citation in pipeline_result.get("citations", []):
        page_number = citation.get("page_number")
        if isinstance(page_number, int):
            cited_pages.add(page_number)
        elif isinstance(page_number, str) and page_number.isdigit():
            cited_pages.add(int(page_number))
    page_matches = sorted(expected_pages & cited_pages)
    page_coverage = round(len(page_matches) / max(1, len(expected_pages)), 2) if expected_pages else 0.0

    answer_term_coverage = (
        round(sum(1 for term in expected_terms if term in answer_text) / len(expected_terms), 2)
        if expected_terms
        else 0.0
    )

    return {
        "retrieval_hit": retrieval_hit,
        "answer_term_coverage": answer_term_coverage,
        "citations_present": citations_present,
        "groundedness_score": groundedness_score,
        "fallback_correct": fallback_correct,
        "fact_coverage": fact_coverage,
        "page_coverage": page_coverage,
        "fact_hits": fact_hits,
        "expected_pages": sorted(expected_pages),
        "matched_pages": page_matches,
    }


def _normalize_text(text: str) -> str:
    """Normalize text for lightweight substring fact checks."""
    cleaned = text.lower()
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
