# Testing Methodology

## Purpose

This project uses lightweight evaluation to show that the final document assistant behaves sensibly on realistic policy-style questions.

The goal is not to claim benchmark-level testing. The goal is to make evaluation:

- easy to explain
- tied to source documents
- visible during a demo
- grounded in citations and retrieved evidence

## What Is Being Evaluated

The final system is evaluated as a document-grounded assistant with:

- document upload and processing
- retrieval quality
- answer grounding
- citation behavior
- fallback behavior when evidence is weak

## Example Evaluation Scenarios

Representative questions include:

- `How many annual leave days does an employee get?`
- `Can annual leave be carried forward?`
- `How many public holidays are employees entitled to?`
- `Summarize the working hours or attendance rules.`
- `What is the resignation notice period?`
- `Summarize the resignation rules.`
- `Summarize the leave and resignation rules.`
- `What is the moon allowance policy?`

These scenarios cover:

- direct factual retrieval
- policy interpretation with caution
- multi-topic summarization
- unsupported-question fallback

## What Review Looks For

### 1. Grounded Answers

- Does the answer stay close to the uploaded documents?
- Does it avoid unsupported claims?
- Does it use careful wording when the document is not fully explicit?

### 2. Citation Quality

- Do the citations point to the right source document?
- Do the page references align with the retrieved evidence where page data exists?
- Are the citations useful for manual verification?

### 3. Retrieval Quality

- Did the system retrieve relevant chunks?
- Did the returned evidence contain the expected policy content?
- Did the retrieval path avoid obviously irrelevant chunks?

### 4. Fallback Behavior

- If the answer is not supported by the documents, did the system avoid hallucinating?
- Did the fallback behavior remain conservative and document-grounded?

### 5. Usability In Demo

- Can the answer path be followed quickly?
- Are the supporting details visible enough to inspect?
- Is latency acceptable for a technical-task demo?

## Manual Review Checklist

- Ask a direct fact question and confirm the answer against the document.
- Ask a summary question and check whether the answer covers the main policy points.
- Ask a question that is not supported by the uploaded documents.
- Review the citations and retrieved chunk details.
- Check whether the system stays grounded rather than sounding confident without evidence.

## Evaluation Notes

1. The same live pipeline is used for both demo interaction and evaluation-style checks.
2. The emphasis is on document-grounded behavior, not black-box scoring.
3. Reviewers can inspect retrieved evidence, citations, and answer details directly.
4. This makes the system easier to defend technically because the evaluation is tied to visible artifacts.

## Scope Note

This evaluation approach is intentionally lightweight. It is suitable for:

- a small engineering prototype
- a walkthrough or engineering review
- a reader who needs quick evidence that the system is grounded and inspectable

It is not intended to replace large-scale benchmark testing or full production QA.
