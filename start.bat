@echo off
echo +===============================+
echo |   Starting Sticker Service    |
echo |          Reload Off           |
echo +===============================+
uvicorn sticker_service.app:app
pause