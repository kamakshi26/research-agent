## Evaluation

Tested against 5 factual questions from IRS Publication 15 (2026).

| Question | Expected | Got | Pass |
|---|---|---|---|
| Social security tax rate | 6.2% | 6.2% | ✓ |
| FUTA tax rate | 6.0% | I don't know | ✗ |
| Late deposit penalty (8 days) | 5% | 5% | ✓ |
| SS wage base limit | $184,500 | $184,500 | ✓ |
| Backup withholding rate | 24% | 24% | ✓ |

**Accuracy: 4/5 (80%)**
**RAGAS string similarity: 0.466**

Known limitation: FUTA rate retrieval fails because the 
6.0% figure appears in a dense regulatory section that 
doesn't chunk cleanly. Fix: add the FUTA section as a 
dedicated chunk with explicit metadata tagging.
