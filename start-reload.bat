@echo off
echo +===============================+
echo |   Starting Sticker Service    |
echo |           Reload On           |
echo +===============================+
uvicorn sticker_service.app:app --reload
pause