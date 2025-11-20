# NLP Earnings Analyzer

This project is a small end-to-end experiment which uses modern NLP to turn earnings call transcripts into numerical vectors, then learn which patterns predict short-term stock reactions.

---

## Overview

For each earnings call:

1. **Input**  
   - A plain-text transcript file, named with the ticker and call date:
     - Example: `AAPL_2025-07-31-Q3.txt`

2. **Target (what we try to predict)**  
   - Short-window benchmark-adjusted return:
     - 3-trading-day log return of the stock after the call  
     minus  
     3-trading-day log return of a benchmark (default: SPY).
   - This is stored as `target_excess_return`.

3. **Label (optional classification target)**  
   - Once all targets are computed, they are mapped to coarse buckets:
     - Bottom 30% → `-1` (relatively bad reactions)
     - Middle 40% → `0` (middle reactions)
     - Top 30% → `1` (relatively good reactions)
   - This is stored as `label`.

4. **Text representation (features)**  
   - The transcript text is encoded using a sentence-embedding model:
     - Default: `sentence-transformers/all-MiniLM-L6-v2`.
   - Each transcript becomes a dense vector of floats (an embedding).

5. **Model**  
   - If there are enough samples and class diversity:
     - A multinomial logistic regression predicts the label (`-1, 0, 1`).
   - If there are too few samples / classes:
     - A Ridge regression predicts the numeric `target_excess_return`.

6. **Output**  
   - For classification: a classification report (precision/recall/F1) per class.
   - For regression: R² and sample predicted vs actual returns.

---

## Project Structure

Typical layout:

```text
nlp-earnings-analyzer/
├─ main.py                # Main pipeline script
├─ transcripts/           # Folder containing earnings call transcripts
│  ├─ AAPL_2025-07-31-Q3.txt
│  ├─ MU_2025-06-25-Q3.txt
│  └─ ...
└─ README.md              # This file
