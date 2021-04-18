@echo off

call %USERPROFILE%\Anaconda3\Scripts\activate.bat %USERPROFILE%\Anaconda3
call conda activate nyc-sc
call jupyter notebook --notebook-dir=".\src"
