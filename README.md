# TikTok Day-30 Views Forecast

WeCloudData DS Bootcamp, Week 6 competition (Kaggle: `predictive-modelling-ds`).

**Task:** predict the views a TikTok video reaches on day 30, using only its first 5 days of engagement, creator stats and metadata. **Metric:** RMSE.

**Data:** the full Hugging Face dataset (`lingbow/tiktok-video-engagement-200k`). 70,939 labelled videos are used for training; the competition test videos are excluded.

## Steps

1. **Load data.** Only numeric and categorical columns are read; text columns are never loaded.
2. **Prevent leakage.** Engagement after day 5 is dropped before any feature is built. Test videos are removed from training.
3. **Features (days 0-5 only).**
   - Growth speed over the last 1, 2 and 3 days, log growth, acceleration, decay of daily increments
   - Like, comment, share, save and download rates
   - Author and music context from the author's other videos (no labels)
   - Creator snapshot, video metadata, resolution, original-sound flag
4. **Target.** Growth after day 5 (`views_day30 / views_last_observed`). Forecasts are floored at the observed views and the growth ratio is clipped to [1, 6].
5. **Model.** LightGBM with each video weighted by its (day-5 views)², so the loss matches raw-scale RMSE.
   - More than 5,000 views: average of two LightGBM models (log growth and capped growth ratio), each averaged over 5 seeds
   - Other videos: one small LightGBM
6. **Validation.** 5-fold cross-validation on the 70,939 labelled videos.
7. **Submission.** Train on all labelled videos, predict the 3,001 test videos, run format checks.

## Result

| | RMSE |
|---|---|
| Baseline (1.16 x day-5 views), cross-validation | 147,738 |
| This model, cross-validation | 123,250 (0.83 of baseline) |

The score depends heavily on a few viral videos, so leaderboard values can differ a lot from cross-validation.

## Compliance

| Requirement | Implementation |
|---|---|
| Tabular modeling | Only numeric and categorical features; text columns, language models and embeddings are excluded |
| No data leakage | Test video ids are held out of training; engagement after day 5 is excluded from feature building; creator snapshots are cut at post date + 5 days |
| Hand-built pipeline | Custom feature engineering, LightGBM and scikit-learn |
| Reproducibility | Fixed random seeds; the notebook regenerates an identical submission on every run |

## How to run

```bash
pip install -r requirements.txt
# put the competition CSVs in ./data and the Hugging Face parquet files
# (videos, engagement_daily, creator_daily) in ./hf
jupyter nbconvert --to notebook --execute tiktok_day30_views_full_data.ipynb --inplace
```

Runtime is about 2 minutes. The notebook writes `submission_full_data.csv`. Paths can also be set with the `DATA_DIR` and `HF_DIR` environment variables.

## Files

- `tiktok_day30_views_full_data.ipynb`: the solution
- `submission_full_data.csv`: predictions for the 3,001 test videos
- `requirements.txt`: Python packages

The datasets are not included.
