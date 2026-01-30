Set-Location "d:\USERFILES\GitHub\hswq"
Remove-Item -Force ".git\index.lock" -ErrorAction SilentlyContinue
Get-ChildItem ".git\objects\c6\tmp_obj_*" -ErrorAction SilentlyContinue | Remove-Item -Force
git add "fp8bench.py"
git commit -m "Add fp8bench.py benchmark script"
git push
