# C-BELIEF-Stream Internal Results Log

Last updated: current development stage  
Project: C-BELIEF-Stream  
Task: temporally grounded renal deterioration claim verification

---

## 1. Dataset / Adapter Summary

### Raw split files

```text
data/stream/train_v2_1.jsonl
data/stream/val_v2_1.jsonl
data/stream/test_v2_1.jsonl
```

### Adapted split files

```text
data/adapted/train.jsonl
data/adapted/val.jsonl
data/adapted/test.jsonl
```

### Adapter output

| Split | Samples | Visible events | Future observed events | Retrospective events |
|---|---:|---:|---:|---:|
| Train | 5590 | 36873 | 20130 | 14501 |
| Validation | 1216 | 8014 | 4171 | 3268 |
| Test | 1194 | 7758 | 4272 | 3081 |

Adapter status:

```text
PASS
```

The adapter successfully separated:

```text
visible_events
future_observed_events
retrospective_events
```

This confirms that the dataset supports the Temporal Evidence Trap experiment.

---

## 2. Training Set Label Distribution

### Initial claim status

| Label | Count |
|---|---:|
| insufficient | 3566 |
| supported | 1659 |
| partially_supported | 365 |

### Initial primary hypothesis

| Hypothesis | Count |
|---|---:|
| uncertain_or_transient_abnormality | 3566 |
| acute_renal_deterioration | 1548 |
| chronic_renal_dysfunction | 476 |

### Sample subtype distribution

| Sample subtype | Count |
|---|---:|
| uncertain_stable_creatinine | 1676 |
| chronic_by_future_ckd_code | 1179 |
| acute_by_visible_trend | 1152 |
| comorbid_aki_ckd | 637 |
| chronic_by_stable_high_creatinine | 474 |
| acute_by_future_diagnosis_or_rrt_only | 398 |
| low_confidence_ratio_trigger | 74 |

### Delayed reattribution

```text
requires_delayed_reattribution = 2524
```

---

## 3. C-BELIEF v0

Description:

```text
Conservative rule baseline.
Uses only visible evidence.
Does not predict chronic_renal_dysfunction.
Does not predict partially_supported.
```

### Validation set

| Metric | Value |
|---|---:|
| n | 1216 |
| claim_status_accuracy | 0.9054 |
| claim_status_macro_f1 | 0.6224 |
| primary_hypothesis_accuracy | 0.9161 |
| primary_hypothesis_macro_f1 | 0.6460 |
| temporal_leakage_rate | 0.0000 |
| retrospective_misuse_rate | 0.0000 |
| retrospective_recognition_rate | 1.0000 |

### Validation label distributions

Gold claim distribution:

```text
insufficient: 773
supported: 365
partially_supported: 78
```

Predicted claim distribution:

```text
insufficient: 875
supported: 341
```

Gold hypothesis distribution:

```text
uncertain_or_transient_abnormality: 773
acute_renal_deterioration: 341
chronic_renal_dysfunction: 102
```

Predicted hypothesis distribution:

```text
uncertain_or_transient_abnormality: 875
acute_renal_deterioration: 341
```

Interpretation:

```text
v0 is temporally safe but too conservative. It does not identify chronic renal dysfunction or partially supported claims.
```

---

## 4. C-BELIEF v0.1

Description:

```text
Hypothesis-aware scoring version.
Improved primary hypothesis prediction.
Over-predicted partially_supported claim status.
```

### Validation set

| Metric | Value |
|---|---:|
| n | 1216 |
| claim_status_accuracy | 0.6168 |
| claim_status_macro_f1 | 0.3872 |
| primary_hypothesis_accuracy | 0.9400 |
| primary_hypothesis_macro_f1 | 0.8956 |
| temporal_leakage_rate | 0.0000 |
| retrospective_misuse_rate | 0.0000 |
| retrospective_recognition_rate | 1.0000 |

### Validation label distributions

Gold claim distribution:

```text
insufficient: 773
supported: 365
partially_supported: 78
```

Predicted claim distribution:

```text
insufficient: 700
partially_supported: 414
supported: 102
```

Gold hypothesis distribution:

```text
uncertain_or_transient_abnormality: 773
acute_renal_deterioration: 341
chronic_renal_dysfunction: 102
```

Predicted hypothesis distribution:

```text
uncertain_or_transient_abnormality: 700
acute_renal_deterioration: 341
chronic_renal_dysfunction: 175
```

Interpretation:

```text
v0.1 substantially improved hypothesis discrimination but made the claim verifier too conservative / too partial.
```

---

## 5. C-BELIEF v0.1 Error Analysis

### Claim status classification report

```text
                     precision    recall  f1-score   support

supported               0.36      0.10      0.16       365
partially_supported     0.03      0.17      0.05        78
insufficient            1.00      0.91      0.95       773

accuracy                                    0.62      1216
macro avg               0.46      0.39      0.39      1216
weighted avg            0.75      0.62      0.66      1216
```

### Confusion matrix

Rows = gold, columns = prediction.

Labels:

```text
supported
partially_supported
insufficient
```

Matrix:

```text
[[ 37, 328,   0],
 [ 65,  13,   0],
 [  0,  73, 700]]
```

### Error by subtype

| Subtype | Gold | Prediction |
|---|---|---|
| acute_by_future_diagnosis_or_rrt_only | insufficient: 96 | insufficient: 96 |
| acute_by_visible_trend | supported: 259 | partially_supported: 259 |
| chronic_by_future_ckd_code | insufficient: 259 | insufficient: 259 |
| chronic_by_stable_high_creatinine | partially_supported: 65; supported: 37 | supported: 102 |
| comorbid_aki_ckd | supported: 69; insufficient: 73 | partially_supported: 142 |
| low_confidence_ratio_trigger | partially_supported: 13 | partially_supported: 13 |
| uncertain_stable_creatinine | insufficient: 345 | insufficient: 345 |

Main finding:

```text
The major failure mode was that acute_by_visible_trend samples were incorrectly classified as partially_supported rather than supported.
```

---

## 6. C-BELIEF v0.2-margin

Description:

```text
Current main method.
Uses v0.1 hypothesis scoring plus margin-based claim verification.
Uses only query-time visible evidence for initial claim verification.
Future and retrospective evidence are explicitly excluded from query-time support.
```

Source file:

```text
src/cbelief/methods/run_cbelief_v0_2_margin.py
```

Prediction files:

```text
data/predictions/cbelief/val_predictions_v0_1.jsonl
data/predictions/cbelief/test_predictions_v0_2_margin.jsonl
```

Note:

```text
The validation prediction filename still contains v0_1, but the logic used was v0.2-margin.
```

### Validation set

| Metric | Value |
|---|---:|
| n | 1216 |
| claim_status_accuracy | 0.8865 |
| claim_status_macro_f1 | 0.6924 |
| primary_hypothesis_accuracy | 0.9400 |
| primary_hypothesis_macro_f1 | 0.8956 |
| temporal_leakage_rate | 0.0000 |
| retrospective_misuse_rate | 0.0000 |
| retrospective_recognition_rate | 1.0000 |

### Validation label distributions

Gold claim distribution:

```text
insufficient: 773
supported: 365
partially_supported: 78
```

Predicted claim distribution:

```text
insufficient: 700
supported: 503
partially_supported: 13
```

Gold hypothesis distribution:

```text
uncertain_or_transient_abnormality: 773
acute_renal_deterioration: 341
chronic_renal_dysfunction: 102
```

Predicted hypothesis distribution:

```text
uncertain_or_transient_abnormality: 700
acute_renal_deterioration: 341
chronic_renal_dysfunction: 175
```

### Test set

| Metric | Value |
|---|---:|
| n | 1194 |
| claim_status_accuracy | 0.8911 |
| claim_status_macro_f1 | 0.6770 |
| primary_hypothesis_accuracy | 0.9523 |
| primary_hypothesis_macro_f1 | 0.9156 |
| temporal_leakage_rate | 0.0000 |
| retrospective_misuse_rate | 0.0000 |
| retrospective_recognition_rate | 1.0000 |

### Test label distributions

Gold claim distribution:

```text
insufficient: 775
partially_supported: 84
supported: 335
```

Predicted claim distribution:

```text
insufficient: 718
supported: 465
partially_supported: 11
```

Gold hypothesis distribution:

```text
uncertain_or_transient_abnormality: 775
chronic_renal_dysfunction: 104
acute_renal_deterioration: 315
```

Predicted hypothesis distribution:

```text
uncertain_or_transient_abnormality: 718
chronic_renal_dysfunction: 161
acute_renal_deterioration: 315
```

Interpretation:

```text
C-BELIEF v0.2-margin achieved stable validation/test performance.
It maintained zero temporal leakage and zero retrospective evidence misuse.
It substantially improved primary hypothesis recognition compared with v0.
```

---

## 7. Full-context Leaky Rule Baseline

Description:

```text
Leaky baseline.
Uses visible, future observed, and retrospective evidence together without temporal gatekeeping.
Simulates the risk of full-context EHR reasoning when future and discharge-level evidence are not explicitly separated.
```

Source file:

```text
src/cbelief/baselines/full_context_leaky_rule.py
```

Prediction file:

```text
data/predictions/full_context_leaky_rule/test_predictions.jsonl
```

Metrics file:

```text
data/metrics/full_context_leaky_rule_test.json
```

### Test set

| Metric | Value |
|---|---:|
| n | 1194 |
| claim_status_accuracy | 0.5946 |
| claim_status_macro_f1 | 0.4109 |
| primary_hypothesis_accuracy | 0.6407 |
| primary_hypothesis_macro_f1 | 0.6194 |
| temporal_leakage_rate | 0.3443 |
| retrospective_misuse_rate | 0.5492 |
| retrospective_recognition_rate | 0.0000 |

### Test label distributions

Gold claim distribution:

```text
insufficient: 775
partially_supported: 84
supported: 335
```

Predicted claim distribution:

```text
supported: 819
insufficient: 375
```

Gold hypothesis distribution:

```text
uncertain_or_transient_abnormality: 775
chronic_renal_dysfunction: 104
acute_renal_deterioration: 315
```

Predicted hypothesis distribution:

```text
chronic_renal_dysfunction: 442
uncertain_or_transient_abnormality: 375
acute_renal_deterioration: 377
```

Interpretation:

```text
The full-context leaky rule baseline shows substantial temporal evidence misuse.
It selected future evidence as supporting evidence in 34.4% of selected support cases.
It selected retrospective evidence as supporting evidence in 54.9% of selected support cases.
It failed to recognize retrospective-only evidence.
```

---

## 8. Main Results Table v1

| Method | Split | Claim Acc | Claim Macro-F1 | Hyp Acc | Hyp Macro-F1 | Temporal Leakage | Retro Misuse | Retro Recognition |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C-BELIEF v0 | Val | 0.9054 | 0.6224 | 0.9161 | 0.6460 | 0.0000 | 0.0000 | 1.0000 |
| C-BELIEF v0.1 | Val | 0.6168 | 0.3872 | 0.9400 | 0.8956 | 0.0000 | 0.0000 | 1.0000 |
| C-BELIEF v0.2-margin | Val | 0.8865 | 0.6924 | 0.9400 | 0.8956 | 0.0000 | 0.0000 | 1.0000 |
| C-BELIEF v0.2-margin | Test | 0.8911 | 0.6770 | 0.9523 | 0.9156 | 0.0000 | 0.0000 | 1.0000 |
| Full-context leaky rule | Test | 0.5946 | 0.4109 | 0.6407 | 0.6194 | 0.3443 | 0.5492 | 0.0000 |

---

## 9. Current Interpretation

The current evidence supports the first version of the Temporal Evidence Trap result:

```text
C-BELIEF-Stream v0.2-margin maintains strong claim verification and hypothesis recognition while completely avoiding temporal leakage and retrospective evidence misuse.

In contrast, a full-context leaky rule baseline that does not separate query-time visible evidence from future and retrospective evidence shows substantial temporal leakage and retrospective evidence misuse.
```

Main internal conclusion:

```text
C-BELIEF v0.2-margin is the current frozen main method.
Full-context leaky rule provides the first negative control for Temporal Evidence Trap.
```

---

## 10. Current Frozen Method

Frozen method name:

```text
C-BELIEF-Stream v0.2-margin
```

Frozen source file:

```text
src/cbelief/methods/run_cbelief_v0_2_margin.py
```

Frozen test prediction file:

```text
data/predictions/cbelief/test_predictions_v0_2_margin.jsonl
```

Frozen test metrics file:

```text
data/metrics/cbelief_v0_2_margin_test.json
```

---

## 11. Next Baselines to Run

Priority order:

```text
1. visible_only_rule
2. full_context_with_provenance_rule
3. rule_based_renal_criteria
4. visible_only_llm
5. full_context_llm
6. full_context_with_provenance_llm
7. time_aware_rag
```

Temporal Evidence Trap target design:

| Condition | Description |
|---|---|
| A. Visible-only | Uses only query-time visible evidence |
| B. Full-context unmarked | Uses visible + future + retrospective evidence without provenance |
| C. Full-context with provenance | Sees all evidence but receives visibility labels |
| D. C-BELIEF-Stream | Uses temporal gatekeeping + claim-grounded evidence attribution |

Current completed conditions:

```text
B. Full-context unmarked / leaky rule
D. C-BELIEF-Stream
```

Remaining immediate conditions:

```text
A. Visible-only rule
C. Full-context with provenance rule
```
