# Agentic RAG Flow

## Workflow Diagram

```text
User Query
  |
  v
Query Rewriter
  |
  v
FAISS Retrieval
  |
  v
Retrieval Evaluation
  |
  +--> Relevant enough? ---- yes ----> Final Context Selection
  |                                      |
  |                                      v
  |                                Ollama Grounded Answer
  |                                      |
  |                                      v
  |                                 Final Answer
  |
  +--> no ----> Retry With Broader Query ----> FAISS Retrieval
                       |
                       v
                 Still weak?
                       |
                       v
                    Fallback
```

## Step By Step

1. The user asks a question.
2. The system rewrites the question into a shorter retrieval-focused phrase.
3. FAISS retrieves the top matching chunks.
4. The system evaluates whether the retrieval looks strong enough.
5. If retrieval is weak, the query is broadened once and retrieval runs again.
6. If retrieval becomes strong enough, the retrieved chunks are sent to Ollama.
7. Ollama answers using only the provided context.
8. Citations are generated from the retrieved chunks used as context.
9. If retrieval is still weak, the system returns the fallback response.

## Successful Retrieval Example

User query:

```text
Can I bring forward leave?
```

Rewritten query:

```text
unused annual leave carry forward policy
```

Outcome:

- leave-policy chunks are retrieved
- scores are high enough
- the grounded answer is generated from the retrieved context

## Failed Retrieval Example

User query:

```text
What is the internal project codename for the new expansion?
```

Outcome:

- chunks may be retrieved, but scores are weak or content is not meaningful
- retrieval evaluation marks the evidence as insufficient
- the system retries once with a broader query

## Fallback Case

If retrieval is still weak after retry, the system returns:

```text
I could not find this information in the uploaded documents.
```

This keeps the demo grounded and reduces unsupported answers.
