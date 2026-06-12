# slang-normalisation

## Sources

`gen_zz_words.csv`:
https://www.kaggle.com/datasets/tawfiayeasmin/gen-z-words-and-phrases-dataset/data

`genz_dataset.csv`:
https://huggingface.co/datasets/Programmer-RD-AI/genz-slang-pairs-1k/blob/main/genz_dataset.csv

`genz_slang.csv` and `genz_emojis.csv`:
https://github.com/kaspercools/genz-dataset

`manual_cleanup.csv`:
manually reviewed duplicates that appeared in more source datasets, consolidated meaning/example/source

`manual_additions.csv`:
manually added recent slang, sourced from:
- https://gabb.com/blog/teen-slang/
- https://axis.org/resource/a-parent-guide-to-teen-slang/
- https://socialdad.ca/2026/05/01/teen-slang-decoded-what-your-kid-is-actually-saying-in-2025-and-2026/
- https://social.colostate.edu/best-practices/new-year-new-slang-words-social-media-managers-should-know-in-2026/
- our own knowledge as internet users and slang found while reviewing source comment datasets

## Dataset Description

The dataset contains **3,133 English social media comments** annotated for slang normalization and primary emotion. Each example consists of an informal or slang-heavy comment, its normalized standard-English version, and an emotion label.

The dataset combines two sources:

| Source | Count | Share |
|---|---:|---:|
| Human-written comments | 698 | 22.3% |
| Synthetic comments | 2,435 | 77.7% |
| **Total** | **3,133** | **100%** |

The full dataset was split into train, validation, and test sets using an approximately **80/10/10 split**, while preserving the human/synthetic ratio across the main splits.

| Split | Human | Synthetic | Total |
|---|---:|---:|---:|
| Train | 558 | 1,948 | 2,506 |
| Validation | 70 | 244 | 314 |
| Test | 70 | 243 | 313 |
| **Total** | **698** | **2,435** | **3,133** |

The average comment length is similar across the splits, which helps ensure that model performance is not biased by one split containing much shorter or longer comments.

| Split | Average comment length |
|---|---:|
| Train | 9.81 words |
| Validation | 9.59 words |
| Test | 9.93 words |

The emotion distribution is also approximately balanced across train, validation, and test sets.

| Emotion | Train | Validation | Test |
|---|---:|---:|---:|
| Joy | 55.3% | 58.0% | 58.8% |
| Disgust | 20.0% | 19.4% | 20.8% |
| Surprise | 10.1% | 9.2% | 8.6% |
| Anger | 8.5% | 7.6% | 7.7% |
| Sadness | 5.1% | 5.1% | 3.5% |
| Fear | 1.0% | 0.6% | 0.6% |

A separate balanced emotion test set was also created to support evaluation on an evenly distributed emotion sample.

| Emotion | Count |
|---|---:|
| Anger | 44 |
| Disgust | 35 |
| Fear | 37 |
| Joy | 39 |
| Sadness | 44 |
| Surprise | 35 |
| **Total** | **234** |

In addition to the main test set, a separate **human-only test set** was created:

| Test set | Count | Purpose |
|---|---:|---|
| Full test set | 313 | Overall evaluation on human and synthetic comments |
| Human-only test set | 70 | Realistic evaluation on manually collected human-written comments |

Because synthetic comments make up the majority of the dataset, model performance should be reported both on the full test set and on the human-only test set. The full test set shows overall performance, while the human-only test set gives a more realistic estimate of how well the model generalizes to real social media comments.