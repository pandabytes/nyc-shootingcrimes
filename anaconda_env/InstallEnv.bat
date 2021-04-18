@echo off

call %USERPROFILE%\Anaconda3\Scripts\activate.bat %USERPROFILE%\Anaconda3
conda env list | find "nyc-sc" 2> nul 1> nul

if ("%errorlevel%" == 1) (
	echo Installing datascience environment
	conda env create -f datascience.yml
) else (
	echo Updating datascience environment
	conda activate datascience
	conda env update --file datascience.yml
)