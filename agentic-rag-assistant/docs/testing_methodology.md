# Testing Methodology

## Purpose

This project uses a small golden-set testing section to show that the Agentic RAG workflow behaves sensibly across common scenarios.

The goal is not full benchmark-style automated grading. The goal is to make evaluation visible, explainable, and tied to explicit facts from the source PDF.

## Test Case Table

| Test Name | Question | Expected Behavior |
|---|---|---|
| Annual Leave Entitlement | How many annual leave days does an employee get? | Return the entitlement table and cite page 13. |
| Carry Forward Leave | Can annual leave be carried forward? | Return the 5-day rule, first-quarter deadline, and forfeiture rule. |
| Public Holiday Entitlement | How many public holidays are employees entitled to? | Return the 15-day entitlement and cite the public holiday section. |
| Working Hours Or Attendance Rules | Summarize the working hours or attendance rules. | Cover assigned hours, rest day, and no-pay-leave / attendance rules. |
| Resignation Notice Period | What is the resignation notice period? | Return the 2-week and 4-week notice rules from page 20. |
| Resignation Rules Summary | Summarize the resignation rules. | Cover notice, separation form, final wages, and clearance / exit interview. |
| Leave And Resignation Summary | Summarize the leave and resignation rules. | Cover both leave and resignation topics with explicit rules. |
| Unavailable Information Test | What is the moon allowance policy? | Trigger fallback instead of hallucinating. |

## What The Golden Check Measures

- Expected-fact coverage:
  - does the answer include the specific handbook facts we expect?
- Citation-page coverage:
  - do the returned citations include the expected source pages?
- Retrieval hit:
  - did the retrieved evidence contain the expected terms?
- Fallback correctness:
  - did the system abstain when the answer was unavailable?

## Manual Review Checklist

- Does the answer match the retrieved evidence?
- Do the citations point to the right document pages?
- Does the answer cover the expected facts for that question?
- Did fallback trigger when the information was unavailable?
- Did the system avoid inventing unsupported details?
- Was latency acceptable for a local demo?

## How To Explain Testing During An Interview

1. Show that each test uses the same live Agentic RAG pipeline as the normal app flow.
2. Explain that evaluation uses a small golden fact set from the handbook rather than opaque LLM-as-judge scoring.
3. Point out that expected facts, expected pages, citations, retries, and fallback behavior are all visible.
4. Clarify that the scores are lightweight coverage checks, and final judgment is still manual review.
5. Mention that this is appropriate for a demo because it is easy to inspect, defend, and tie back to the PDF.
