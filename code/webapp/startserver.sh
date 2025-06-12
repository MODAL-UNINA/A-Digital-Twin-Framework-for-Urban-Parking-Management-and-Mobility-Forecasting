#!/bin/bash

set -ex

export PORT=${PORT:-8080}
export HOST=${HOST:-0.0.0.0}
export DJANGO_ALLOWED_HOSTS=${DJANGO_ALLOWED_HOSTS:-"localhost,127.0.0.1,webapp"}
export DJANGO_CSRF_TRUSTED_ORIGINS=${DJANGO_CSRF_TRUSTED_ORIGINS:-"http://localhost:"$PORT",http://127.0.0.1:"$PORT",http://webapp:"$PORT}
export DJANGO_SECRET_KEY=${DJANGO_SECRET_KEY:-"your-secret-key"}
export GUNICORN_WORKERS=${GUNICORN_WORKERS:-8}

export DJANGO_DEBUG=${DJANGO_DEBUG:-0}
export CUDA_VISIBLE_DEVICES=${DJANGO_DEBUG:-0}
export CUDA_LAUNCH_BLOCKING=${DJANGO_DEBUG:-1}

python manage.py makemigrations
python -uB manage.py migrate --noinput
python manage.py collectstatic --noinput
gunicorn --config gunicorn_config.py DTMOB_webapp.wsgi:application