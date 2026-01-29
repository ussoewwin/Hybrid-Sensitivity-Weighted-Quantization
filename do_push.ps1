Set-Location "d:\USERFILES\GitHub\hswq"
Remove-Item -Force ".git\index.lock" -ErrorAction SilentlyContinue
Get-ChildItem ".git\objects\c6\tmp_obj_*" -ErrorAction SilentlyContinue | Remove-Item -Force
git add .
git status
git commit -m "Rewrite README in detail from HSWQ technical doc"
git push
