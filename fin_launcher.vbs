Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c python -m streamlit run D:\Projects\Finance\trading-analytics\src\dashboard\0_Analisi.py", 0, False
