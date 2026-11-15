# W5D6 Model Routing Strategy

| Query | Route | Reason |
|---|---|---|
| What is a Python list? | llama3.2:3b | Use the local Llama model for a straightforward question. |
| Explain the machine-learning workflow. | qwen2.5:3b | Use the local Qwen model for a structured explanation. |
| Critically analyse the limitations of this research method. | Cloud API (proposed only) | Complex reasoning may benefit from a stronger model. No cloud API call is made in this practical. |

## Routing Rules

- Repeated similar queries: check Redis first.
- Cache hit: reuse answer only when similarity is >= 0.92.
- Simple questions: llama3.2:3b.
- Medium-complexity questions: qwen2.5:3b.
- Complex questions: cloud model may be considered after budget and privacy checks.
- No paid cloud API calls are made by this practical.

## Limitations

- This is a beginner-friendly keyword-based router.
- It is not a trained complexity classifier.
- Local inference still consumes hardware resources and electricity.