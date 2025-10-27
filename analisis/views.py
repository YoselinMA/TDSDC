import os
import io
from functools import lru_cache

from django.shortcuts import render
from django.utils.safestring import mark_safe
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'datasets', 'datasets', 'TotalFeatures-ISCXFlowMeter.csv')

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y

def train_val_test_split_df(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train_set, val_set, test_set

@lru_cache(maxsize=1)
def load_df_cached(sample_n=10000):
    df = pd.read_csv(CSV_PATH)
    if sample_n and sample_n < len(df):
        df_sample = df.sample(n=sample_n, random_state=42)
    else:
        df_sample = df
    return df_sample

def df_info_to_text(df):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def analysis_view(request):
    try:
        df_sample = load_df_cached()
    except Exception as e:
        return render(request, 'analisis/analysis.html', {'error': str(e)})

    head_html = df_sample.head().to_html(classes='table table-striped table-center', index=False)
    describe_html = df_sample.describe().to_html(classes='table table-striped table-center')
    info_text = df_info_to_text(df_sample)

    strat_col = 'class' if 'class' in df_sample.columns else None
    train_set, val_set, test_set = train_val_test_split_df(df_sample, stratify=strat_col)

    train_head_html = train_set.head().to_html(classes='table table-striped table-center', index=False)

    context = {
        'head_html': mark_safe(head_html),
        'describe_html': mark_safe(describe_html),
        'info_text': info_text,
        'train_head_html': mark_safe(train_head_html),
    }
    return render(request, 'analisis/analysis.html', context)
