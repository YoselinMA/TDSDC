import os
import io
import requests
from functools import lru_cache

from django.shortcuts import render
from django.utils.safestring import mark_safe
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'datasets', 'TotalFeatures-ISCXFlowMeter.csv')
FILE_ID = '1kxHCT5asFDWdCrcLj-QMqwZaAFqL-hBu'  # reemplaza con el ID de tu archivo en Drive


def download_from_drive(file_id, destination):
    if os.path.exists(destination):
        return  
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y

def train_val_test_split_df(df, rstate=42, shuffle=True, stratify=None):
    """Divide un dataframe en train/val/test."""
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train_set, val_set, test_set

@lru_cache(maxsize=1)
def load_df_cached(sample_n=10000):
    """Carga el CSV y devuelve una muestra para velocidad."""
    download_from_drive(FILE_ID, CSV_PATH)
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
        df_sample = load_df_cached(sample_n=10000)
    except Exception as e:
        return render(request, 'analisis/analysis.html', {'error': str(e)})

    head_html = df_sample.head().to_html(classes='table table-striped', index=False)
    describe_html = df_sample.describe().to_html(classes='table table-striped')
    info_text = df_info_to_text(df_sample)

    strat_col = 'class' if 'class' in df_sample.columns else None
    train_set, val_set, test_set = train_val_test_split_df(df_sample, rstate=42, shuffle=True, stratify=strat_col)

    shapes = {
        'train_shape': train_set.shape,
        'val_shape': val_set.shape,
        'test_shape': test_set.shape,
    }

    train_head_html = train_set.head().to_html(classes='table table-sm', index=False)

    context = {
        'head_html': mark_safe(head_html),
        'describe_html': mark_safe(describe_html),
        'info_text': info_text,
        'shapes': shapes,
        'train_head_html': mark_safe(train_head_html),
        'stratify_used': strat_col,
    }
    return render(request, 'analisis/analysis.html', context)
