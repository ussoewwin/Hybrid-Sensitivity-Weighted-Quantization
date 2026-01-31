Set-Location "d:\USERFILES\GitHub\hswq"
Remove-Item -Force ".git\index.lock" -ErrorAction SilentlyContinue
Get-ChildItem ".git\objects\c6\tmp_obj_*" -ErrorAction SilentlyContinue | Remove-Item -Force
git add -A
git commit -m "Add ComfyUI-master, bulk file/folder changes, English comments"
git push
