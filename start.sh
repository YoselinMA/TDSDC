
export PORT=${PORT:-8000}


gunicorn Tecnica.wsgi:application --bind 0.0.0.0:$PORT
