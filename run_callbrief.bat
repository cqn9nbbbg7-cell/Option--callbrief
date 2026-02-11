@echo off
setlocal

REM --- Initialize Anaconda ---
call "C:\Users\eshof\anaconda3\Scripts\activate.bat" "C:\Users\eshof\anaconda3"

REM --- Activate project environment ---
call conda activate callbrief

REM --- Run CallBrief ---
python "C:\Users\eshof\OneDrive\Documents\MAsters FIN\project cool\callbrief.py" ^
 > "C:\Users\eshof\OneDrive\Documents\MAsters FIN\project cool\logs\callbrief_last_run.txt" 2>&1

endlocal
setlocal

call "C:\Users\eshof\anaconda3\Scripts\activate.bat" "C:\Users\eshof\anaconda3"
call conda activate callbrief

python "C:\Users\eshof\OneDrive\Documents\MAsters FIN\project cool\callbrief.py" ^
  > "C:\Users\eshof\OneDrive\Documents\MAsters FIN\project cool\logs\callbrief_last_run.txt" 2>&1

endlocal
