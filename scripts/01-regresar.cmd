@echo off

call :RenameIfExists "origins\dataset001" "construction-ppe_ultralytics"
call :RenameIfExists "origins\dataset002" "Dataset-Object-identication-training-construction"
call :RenameIfExists "origins\dataset003" "EPP.v11i.yolov11_roboflow"
call :RenameIfExists "origins\dataset004" "Hard Hat Universe.v26-no_nulls_plain.yolov11"
call :RenameIfExists "origins\dataset005" "PPE Compliance Detection.v5-public-dataset-v4.yolov11"
call :RenameIfExists "origins\dataset006" "PPE.v9-pictor-v3-revised.yolov11_Art. Rahman"
exit /b

:RenameIfExists
if exist "%~1\" ren "%~1" "%~2"
exit /b