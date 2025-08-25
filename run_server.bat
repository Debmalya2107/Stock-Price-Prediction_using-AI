@echo off
REM Change to your Django project directory
cd /d C:\Users\230625\Desktop\stock_prediction_webapp

REM Run Django server
python manage.py runserver

REM Keep window open after server stops
pause
