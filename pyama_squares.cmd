@echo off
call "C:\ProgramData\Anaconda3\condabin\conda.bat" activate "%~dp0\env"
doskey pyama_squares="%~dp0\env\python.exe" -X "pycache_prefix=%localappdata%\Temp\pyama_pycache" "%~dp0\pyama_squares.py" $*
cmd \k