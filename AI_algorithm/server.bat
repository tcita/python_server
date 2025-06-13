@echo off
title CardGame AI Server Console
echo ======================================================
echo           Starting AI Server...
echo           Working Directory: %cd%
echo           (Do not close this window)
echo ======================================================
echo.
python server.py
echo.
echo ======================================================
echo      Server has stopped running.
echo      Press any key to close this window...
echo ======================================================
pause