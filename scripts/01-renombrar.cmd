@echo off

call :RenameIfExists "origins\construction-ppe_ultralytics" "dataset001"
call :RenameIfExists "origins\Dataset-Object-identication-training-construction" "dataset002"
call :RenameIfExists "origins\EPP.v11i.yolov11_roboflow" "dataset003"
call :RenameIfExists "origins\Hard Hat Universe.v26-no_nulls_plain.yolov11" "dataset004"
call :RenameIfExists "origins\PPE Compliance Detection.v5-public-dataset-v4.yolov11" "dataset005"
call :RenameIfExists "origins\PPE.v9-pictor-v3-revised.yolov11_Art. Rahman" "dataset006"
exit /b

:RenameIfExists
if exist "%~1\" ren "%~1" "%~2"
exit /b