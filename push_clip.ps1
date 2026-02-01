Set-Location "d:\USERFILES\GitHub\hswq"
Remove-Item -Force ".git\index.lock" -ErrorAction SilentlyContinue
git add "clip/"
git commit -F "push_clip_msg.txt"
git push
