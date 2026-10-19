# Dataset Instructions

## Source
UCI Machine Learning Repository  
Online Shoppers Purchasing Intention Dataset  
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

## Access
The reproducibility script uses:

```python
from ucimlrepo import fetch_ucirepo
shopping_data = fetch_ucirepo(id=468)
```

An internet connection is needed the first time the dataset is fetched.

## Preparation Used in This Project
1. Combine the UCI features and `Revenue` target.
2. Identify exact duplicate rows.
3. Remove exact duplicates before the train/test split.
4. Treat `OperatingSystems`, `Browser`, `Region`, and `TrafficType` as categorical variables even though their values are numeric codes.
5. Use an 80/20 stratified train/test split with `random_state=42`.

No private or personally identifiable customer information is added to this repository.
