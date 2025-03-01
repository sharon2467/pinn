@echo off  
setlocal enabledelayedexpansion  

:: 设置根目录为当前目录  
set "root_dir=%cd%"  

:: 创建日志文件  
set "log_file=folder_cleaning_log_%date:~0,10%.txt"  

echo 开始清理文件夹：%date% %time% > "%log_file%"  
echo =============================================== >> "%log_file%"  

:: 遍历所有子文件夹  
for /f "delims=" %%a in ('dir /b /s /ad "%root_dir%"') do (  
    set "folder=%%a"  
    
    :: 检查文件夹是否包含PNG图片  
    if exist "%%a\*.png" (  
        echo [INFO] 文件夹 [!folder!] 中包含PNG图片，保留。>> "%log_file%"  
    ) else (  
        echo [INFO] 文件夹 [!folder!] 中不包含PNG图片，尝试删除...>> "%log_file%"  
        
        :: 尝试删除文件夹，忽略错误信息  
        rd /s /q "%%a" 2> nul  
        if !errorlevel! equ 0 (  
            echo [SUCCESS] 删除了文件夹 [!folder!]>> "%log_file%"  
        ) else (  
            echo [ERROR] 无法删除文件夹 [!folder!]，可能由于权限问题或文件夹被占用。>> "%log_file%"  
        )  
    )  
)  

echo =============================================== >> "%log_file%"  
echo 清理完成：%date% %time% >> "%log_file%"  

pause