# Dataset Instructions

This project uses the **Kaggle Fake News Dataset** for training and evaluation.

## Download Instructions

1. Visit the Kaggle competition page:  
   👉 https://www.kaggle.com/c/fake-news/data

2. Sign in or create a free Kaggle account.

3. Accept the competition rules and download the dataset.

4. Place the downloaded CSV files into this `data/` directory:
   ```
   data/
   ├── train.csv     ← Main training dataset (required)
   └── test.csv      ← Test dataset (optional)
   ```

## Using Kaggle API (Alternative)

You can also download the dataset using the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c fake-news
unzip fake-news.zip -d data/
```

> **Note:** You must have a `kaggle.json` API token configured.  
> See: https://www.kaggle.com/docs/api

## Dataset Format

The training file (`train.csv`) contains the following columns:

| Column   | Type    | Description                              |
|----------|---------|------------------------------------------|
| `id`     | Integer | Unique identifier for each article       |
| `title`  | String  | The headline of the news article         |
| `author` | String  | The author of the news article           |
| `text`   | String  | The full body text of the news article   |
| `label`  | Integer | `1` = Fake News, `0` = Real News         |

## Notes

- Dataset files (`.csv`) are excluded from version control via `.gitignore`
- The dataset contains approximately 20,800 labelled news articles
- Missing values exist in `author`, `title`, and `text` columns — these are handled automatically by the preprocessing script
