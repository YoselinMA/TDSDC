import pandas as pd
import io
import requests
from django.shortcuts import render

def analysis_view(request):
    
    file_id = "1kxHCT5asFDWdCrcLj-QMqwZaAFqL-hBu" 
   
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
      
        response = requests.get(url)
        response.raise_for_status()

       
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), on_bad_lines='skip')
        
   
        data_preview = df.head(5).to_html(classes="tabla-morada")

        return render(request, "analisis/analysis.html", {
            "data_preview": data_preview,
        })

    except Exception as e:
        return render(request, "analisis/analysis.html", {
            "error": str(e),
        })
