## LLM validation subset results, n=100

We evaluated DeepSeek-chat on a stratified validation subset of 100 cases under four LLM conditions.

| Method | Claim Acc | Claim Macro-F1 | Hyp Acc | Hyp Macro-F1 | Temporal Leakage | Retro Misuse | Retro Recognition |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM visible-only raw | 0.44 | 0.481 | 0.83 | 0.753 | 0.000 | 0.000 | 0.000 |
| LLM visible-only + verifier | 0.81 | 0.789 | 0.83 | 0.753 | 0.000 | 0.000 | 0.000 |
| LLM full-context unmarked raw | 0.45 | 0.471 | 0.81 | 0.794 | 0.0017 | 0.0462 | 0.844 |
| LLM full-context unmarked + verifier | 0.81 | 0.776 | 0.81 | 0.794 | 0.000 | 0.0766 | 0.844 |
| LLM full-context with provenance raw | 0.40 | 0.418 | 0.72 | 0.640 | 0.000 | 0.000 | 0.956 |
| LLM full-context with provenance + verifier | 0.67 | 0.562 | 0.72 | 0.640 | 0.000 | 0.000 | 0.956 |
| LLM C-BELIEF prompted raw | 0.39 | 0.407 | 0.88 | 0.890 | 0.000 | 0.000 | 0.996 |
| LLM C-BELIEF prompted + verifier | 0.81 | 0.762 | 0.88 | 0.890 | 0.000 | 0.000 | 0.996 |

Key observations:
1. Full-context unmarked LLM misused retrospective evidence as supporting evidence.
2. Explicit provenance eliminated temporal and retrospective evidence misuse.
3. C-BELIEF prompting achieved the strongest hypothesis macro-F1 and retrospective evidence recognition.
4. A lightweight consistency verifier corrected claim-status over-support while preserving temporal safety.