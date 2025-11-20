import os
from datetime import datetime, timedelta

from collections import Counter
import numpy as np
import pandas as pd
import yfinance as yf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------
# Config
# -------------------

TRANSCRIPT_DIR = "transcripts"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BENCHMARK = "SPY"
EVENT_WINDOW_TRADING_DAYS = 3


# -------------------
# Data loading
# -------------------

def load_transcripts(transcript_dir: str) -> pd.DataFrame:
    rows = []
    for fname in os.listdir(transcript_dir):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(transcript_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Example: AAPL_2025-07-31-Q3.txt
        base = fname[:-4]  # strip .txt
        ticker, rest = base.split("_", 1)
        date_str = "-".join(rest.split("-")[:3])
        call_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        rows.append(
            {
                "ticker": ticker.upper(),
                "filename": fname,
                "call_date": call_date,
                "text": text,
            }
        )

    return pd.DataFrame(rows)


# -------------------
# Return calculation
# -------------------
def compute_event_excess_return(
    ticker,
    call_date,
    window_trading_days=3,
    benchmark="SPY",
):
    """
    Event-style excess return:
    - event window: [0, window_trading_days] in trading days
    - 0 = first trading day on or after call_date
    """
    # Pad the date range to be safe around holidays, etc.
    start = call_date - timedelta(days=2)
    end = call_date + timedelta(days=window_trading_days * 2 + 5)

    stock = yf.download(ticker, start=start, end=end, progress=False)
    bench = yf.download(benchmark, start=start, end=end, progress=False)

    if stock.empty or bench.empty:
        return np.nan

    # Ensure sorted by date
    stock = stock.sort_index()
    bench = bench.sort_index()

    stock_dates = stock.index
    bench_dates = bench.index

    # First trading day on or after call_date
    stock_start_idx = stock_dates.searchsorted(np.datetime64(call_date), side="left")
    bench_start_idx = bench_dates.searchsorted(np.datetime64(call_date), side="left")

    stock_end_idx = stock_start_idx + window_trading_days
    bench_end_idx = bench_start_idx + window_trading_days

    # Check bounds
    if (
        stock_start_idx >= len(stock_dates)
        or stock_end_idx >= len(stock_dates)
        or bench_start_idx >= len(bench_dates)
        or bench_end_idx >= len(bench_dates)
    ):
        return np.nan

    # Extract as plain floats (this avoids Series/array ambiguity)
    stock_start_price = float(stock["Close"].iloc[stock_start_idx])
    stock_end_price   = float(stock["Close"].iloc[stock_end_idx])
    bench_start_price = float(bench["Close"].iloc[bench_start_idx])
    bench_end_price   = float(bench["Close"].iloc[bench_end_idx])

    # Basic sanity check
    if (
        stock_start_price <= 0
        or stock_end_price <= 0
        or bench_start_price <= 0
        or bench_end_price <= 0
    ):
        return np.nan

    # Log returns
    r_stock = np.log(stock_end_price / stock_start_price)
    r_bench = np.log(bench_end_price / bench_start_price)

    return r_stock - r_bench

def add_return_labels(df: pd.DataFrame, col="target_excess_return") -> pd.DataFrame:
    df = df.dropna(subset=[col]).copy()

    if df.empty:
        return df

    q30 = df[col].quantile(0.3)
    q70 = df[col].quantile(0.7)

    def label_func(x):
        if x <= q30:
            return -1
        elif x >= q70:
            return 1
        else:
            return 0

    df["label"] = df[col].apply(label_func)
    return df

# -------------------
# Main
# -------------------

def main():
    df = load_transcripts(TRANSCRIPT_DIR)
    if df.empty:
        print("No transcripts found. Add .txt files to the transcripts folder.")
        return

    print("Loaded transcripts:")
    print(df[["ticker", "filename", "call_date"]])

    # Compute event-style excess returns
    print("\nComputing event excess returns...")
    df["target_excess_return"] = df.apply(
        lambda row: compute_event_excess_return(
            row["ticker"],
            row["call_date"],
            window_trading_days=EVENT_WINDOW_TRADING_DAYS,
            benchmark=BENCHMARK,
        ),
        axis=1,
    )

    df = add_return_labels(df, col="target_excess_return")
    if df.empty or "label" not in df.columns:
        print("No valid labeled rows after computing returns.")
        return

    print("\nSample targets:")
    print(df[["ticker", "call_date", "target_excess_return", "label"]].head())

        # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Compute embeddings
    print("Computing embeddings...")
    texts = df["text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    X = np.array(embeddings)

    # ------------------------
    # Decide: classification or regression
    # ------------------------
    from collections import Counter
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, r2_score

    # 3-class labels from add_return_labels
    labels = df["label"].values
    label_counts = Counter(labels)
    print("\nLabel counts:", label_counts)

    # Use classification only if we have at least 2 classes
    # and each class has at least 2 samples and we have enough rows.
    use_classification = (
        len(label_counts) >= 2
        and min(label_counts.values()) >= 2
        and len(df) >= 10
    )

    # Decide test size
    if len(df) < 5:
        test_size = 0.4
    else:
        test_size = 0.3

    if use_classification:
        # -------- Classification path --------
        print("\nUsing 3-class classification (-1, 0, 1).")
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )

        clf = LogisticRegression(
            multi_class="multinomial",
            max_iter=200,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print("\nClassification report (labels: -1 = worst, 0 = middle, 1 = best):")
        print(classification_report(y_test, y_pred, digits=3))

    else:
        # -------- Regression path --------
        print(
            "\nNot enough label variety for classification; "
            "falling back to regression on numeric excess returns."
        )

        y = df["target_excess_return"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
        )

        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)

        if len(y_test) > 1:
            r2 = r2_score(y_test, y_pred)
        else:
            r2 = float("nan")

        print(f"\nR^2 on test set: {r2:.4f}")
        print("\nExample predictions vs actual:")
        for true_val, pred_val in list(zip(y_test, y_pred))[:5]:
            print(f"Actual: {true_val:.4f}, Predicted: {pred_val:.4f}")



if __name__ == "__main__":
    main()
