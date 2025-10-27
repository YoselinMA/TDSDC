from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from io import StringIO

def analysis_view(request):
  
    file_id = "1kxHCT5asFDWdCrcLj-QMqwZaAFqL-hBu"  
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    df = pd.read_csv(url)

   
    df_head_html = df.head().to_html(classes="table table-striped", index=False)

    
    df_describe_html = df.describe().to_html(classes="table table-striped", index=True)

  
    buffer = StringIO()
    df.info(buf=buffer)
    df_info_html = buffer.getvalue().replace("\n", "<br>")

    strat_col = None  
    train_set, val_set, test_set = train_val_test_split_df(df, rstate=42, shuffle=True, stratify=strat_col)

    train_html = train_set.head().to_html(classes="table table-striped", index=False)
    val_html = val_set.head().to_html(classes="table table-striped", index=False)
    test_html = test_set.head().to_html(classes="table table-striped", index=False)

    context = {
        "df_head_html": df_head_html,
        "df_describe_html": df_describe_html,
        "df_info_html": df_info_html,
        "train_html": train_html,
        "val_html": val_html,
        "test_html": test_html,
        "strat_col": strat_col,
    }

    return render(request, "analisis/analysis.html", context)

def train_val_test_split_df(df, rstate=42, shuffle=True, stratify=None):
    if stratify:
        strat = df[stratify]
    else:
        strat = None

    train, temp = train_test_split(df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    if stratify:
        strat_temp = temp[stratify]
    else:
        strat_temp = None
    val, test = train_test_split(temp, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat_temp)
    return train, val, test
