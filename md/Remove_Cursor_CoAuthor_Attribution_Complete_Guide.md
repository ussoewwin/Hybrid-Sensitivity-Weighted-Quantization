# Cursor の Co-authored-by と開発者表示を完全に消す手順（解説書）

この文書は、Cursor（またはそのエージェント）が Git コミットに自動付与する `Co-authored-by: Cursor <cursoragent@cursor.com>` および GitHub 上の「Contributors」への cursoragent 表示を、**恒久的に防止し、既に混入した分も完全に除去する**ための手順を、手抜きなく記載したものです。

---

## 1. 問題の整理

### 1.1 何が起きているか

- Cursor が `git commit` を実行する際、コミットメッセージの末尾に次のような行が付与されることがあります。
  ```text
  Co-authored-by: Cursor <cursoragent@cursor.com>
  ```
- さらに、そのコミットの **Author** または **Committer** が `cursoragent`（あるいは類似の名前・メール）になっている場合、GitHub のリポジトリ「Contributors」に「cursoragent / Cursor Agent」が表示されます。
- 実際の開発者はあなたお一人であるにもかかわらず、第三者（Cursor）が共同開発者のように表示される状態になります。

### 1.2 対処の三本柱

| 対象 | 目的 | 手段 |
|------|------|------|
| **今後のコミット** | これ以上 Co-authored-by を入れない | `.git/hooks/commit-msg` でメッセージから該当行を削除 |
| **過去のコミット** | 履歴上のメッセージ・作者をあなたのみに統一する | `git filter-branch` でメッセージ書き換え・作者書き換え |
| **GitHub の表示** | Contributors から cursoragent を消す | 履歴書き換えの反映 + キャッシュ更新のための操作 |

以下、順に完全手順のみを記載します。

---

## 2. 今後のコミットに Co-authored-by を付けさせない（commit-msg フック）

### 2.1 概要

- Git の **commit-msg フック** は、コミットが確定する直前にコミットメッセージが書かれたファイルを渡されます。
- ここで「Co-authored-by: … Cursor …」を含む行を削除すれば、リポジトリに残るメッセージには一切その行が入りません。
- Cursor がメッセージに付与しても、保存されるコミットからは削除されます。

### 2.2 フックの置き場所

- リポジトリの **`.git/hooks/commit-msg`** に配置します。
- パスの例（Windows）: `d:\USERFILES\GitHub\hswq\.git\hooks\commit-msg`
- このファイルはリポジトリの「作業ツリー」外（`.git` 内）のため、通常の `git add` では追跡されません。クローン先・別マシンでは同じ内容を再度設置する必要があります。

### 2.3 フックの内容（そのまま使用可能）

以下を **そのまま** `.git/hooks/commit-msg` に保存してください。先頭の `#!/bin/sh` は必須です。

```sh
#!/bin/sh
# Strip Cursor agent co-author trailer; sole author is the human developer.
msgfile="$1"
[ -z "$msgfile" ] || [ ! -f "$msgfile" ] && exit 0
tmp=$(mktemp)
while IFS= read -r line; do
  case "$line" in
    Co-authored-by:*[Cc]ursor*) ;;
    *) printf '%s\n' "$line" ;;
  esac
done < "$msgfile" > "$tmp" && mv "$tmp" "$msgfile"
exit 0
```

- **1行目**: シェルは `sh`。Git for Windows では Git Bash の `sh` で実行されます。
- **2行目**: コメント。削除対象の意図を残すため推奨。
- **3行目**: `$1` は Git が渡すコミットメッセージファイルのパスです。
- **4行目**: 引数が空、またはファイルが存在しない場合は何もせず正常終了（`exit 0`）。フック失敗にしないため。
- **5行目**: `mktemp` で一時ファイルを作成。上書き元のファイルを安全に書き換えるため。
- **6–11行目**: メッセージを 1 行ずつ読み、`Co-authored-by:` のうしろに `Cursor`（大文字小文字どちらでも）を含む行は出力せず、それ以外はそのまま出力。結果を一時ファイルに書き、成功したら元のメッセージファイルを置き換え。
- **12行目**: 常に成功で終了。

### 2.4 実行権限（Unix / Git Bash）

- Linux や macOS、Git Bash では実行権限が必要です。
  ```bash
  chmod +x .git/hooks/commit-msg
  ```
- Windows で「エクスプローラー＋右クリック」のみの環境では必須でない場合もありますが、Git Bash から実行する場合は `chmod +x` をしておくと確実です。

### 2.5 動作確認

- 意図的に Co-authored-by を付けたメッセージでコミットを試します。
  ```bash
  git commit --allow-empty -m "test

  Co-authored-by: Cursor <cursoragent@cursor.com>"
  ```
- 直後に `git log -1 --format=%B` でメッセージを表示し、「Co-authored-by: … Cursor …」の行が含まれていなければ成功です。

---

## 3. 既存のコミットから Co-authored-by を削除する（メッセージの書き換え）

### 3.1 概要

- すでにリポジトリに存在するコミットの「メッセージ」から、`Co-authored-by: … Cursor …` の行を削除します。
- **コミットのハッシュは変わります。** すでに push 済みのブランチに対して行う場合は、後述のとおり force push が必要になります。

### 3.2 使用するコマンド

- **対象**: 通常は `main` ブランチのみで十分です。他ブランチも同様の履歴を持つ場合は、それぞれに対して実行する必要があります。
- **コマンド**（Git Bash または `sh` が使える環境で実行。先頭の `cd` はあなたのリポジトリのルートに合わせて変更してください）:
  ```bash
  cd /path/to/your/Hybrid-Sensitivity-Weighted-Quantization
  git filter-branch -f --msg-filter "sed '/^Co-authored-by:.*[Cc]ursor/d'" main
  ```
  - `-f`: 既に `filter-branch` を実行した場合でも上書きするため。
  - `--msg-filter "..."`: 各コミットのメッセージを標準入力から受け取り、このコマンドの標準出力で置き換えます。
  - `sed '/^Co-authored-by:.*[Cc]ursor/d'`: 行頭が `Co-authored-by:` で、その行のどこかに `Cursor` または `cursor` を含む行を削除します。
- **注意**: `sed` の書式は環境によって異なります。Git for Windows 付属の `sed` では上記で動作します。別の環境では `sed '/^Co-authored-by:.*[Cc]ursor/d'` が 1 行で渡るようにクォートを調整してください。

### 3.3 実行後のバックアップ ref の削除

- `git filter-branch` は、書き換え前の参照を `refs/original/` に残します。この ref が残っていると、ガベージコレクトで古いコミットがすぐには消えず、混乱の元になります。
- 削除コマンド:
  ```bash
  git update-ref -d refs/original/refs/heads/main
  ```
- 複数ブランチに対して `filter-branch` を実行した場合は、それぞれに対して `refs/original/refs/heads/<ブランチ名>` を同様に `git update-ref -d` で削除してください。

### 3.4 実行時間

- コミット数が多いと数十秒〜数分かかることがあります。警告メッセージ（filter-branch の非推奨案内）が出ても、今回の用途ではそのまま実行して問題ありません。

---

## 4. 既存のコミットの「作者」を cursoragent からあなたに差し替える（必要時のみ）

### 4.1 いつ必要か

- GitHub の「Contributors」は、**コミットの Author（および場合により Committer）のメールアドレス**で集計されます。
- メッセージから Co-authored-by を削除しただけでは、**Author が cursoragent のまま**のコミットがあると、GitHub 上では引き続き cursoragent が貢献者として表示されます。
- そのため、「過去のコミットのうち、Author または Committer が cursoragent（あるいは Cursor 関連）になっているもの」を、あなたの名前・メールに書き換える必要があります。

### 4.2 作者・コミッターの確認

- 現在の履歴で、cursoragent が作者になっているコミットがあるか確認します。
  ```bash
  git log --all --format="%h %an <%ae>" | grep -i cursor
  ```
  - 何も出力されなければ、すでにすべてあなたの名前になっています。
  - 出力がある場合は、それらのコミットが GitHub 上で cursoragent として数えられている可能性があります。

### 4.3 作者・コミッターの一括書き換え（env-filter）

- すべてのコミットについて、「作者が cursoragent（または Cursor）なら、あなたの名前に置き換える」には `--env-filter` を使います。
- 以下は、**作者メールが `cursor@cursor.sh` または `cursoragent@cursor.com` のとき**、あなたの名前・メールに差し替える例です。実際のメール・名前はあなたの GitHub 用（例: `136552381+ussoewwin@users.noreply.github.com`）に置き換えてください。
  ```bash
  git filter-branch -f --env-filter '
  if [ "$GIT_AUTHOR_EMAIL" = "cursor@cursor.sh" ] || [ "$GIT_AUTHOR_EMAIL" = "cursoragent@cursor.com" ]; then
    export GIT_AUTHOR_NAME="ussoewwin"
    export GIT_AUTHOR_EMAIL="136552381+ussoewwin@users.noreply.github.com"
  fi
  if [ "$GIT_COMMITTER_EMAIL" = "cursor@cursor.sh" ] || [ "$GIT_COMMITTER_EMAIL" = "cursoragent@cursor.com" ]; then
    export GIT_COMMITTER_NAME="ussoewwin"
    export GIT_COMMITTER_EMAIL="136552381+ussoewwin@users.noreply.github.com"
  fi
  ' main
  ```
- 実行後、再度 `git log --all --format="%h %an <%ae>" | grep -i cursor` で何も出ないことを確認し、`refs/original/refs/heads/main` を `git update-ref -d` で削除してください。

### 4.4 実行順序

- 先に **メッセージの書き換え**（セクション 3）を行い、その後に **作者の書き換え**（この節）を行うと、一つの `filter-branch` で両方やるより手順が分かりやすくなります。
- 両方を一度に行うことも可能ですが、`--msg-filter` と `--env-filter` を同時に指定する形になり、コマンドが長くなるため、ここでは「メッセージ → 作者」の 2 段階を推奨します。

---

## 5. .mailmap から cursoragent の行を削除する

### 5.1 .mailmap の役割

- `.mailmap` は、**表示用**の名前・メールの統一に使われます。`git log` や GitHub の「このコミットの作者表示」を、別の名前に書き換えることができます。
- 例: `cursoragent <cursor@cursor.sh>` で記録されているコミットを、「ussoewwin」として表示させる、といった使い方です。

### 5.2 なぜ削除するか

- Cursor を「開発者」として残したくない場合、**.mailmap に cursoragent の行があると、その存在が残り続けます。**
- 履歴の作者をあなたに書き換え済み（セクション 4）であれば、cursoragent を .mailmap でマッピングする必要はなく、その行は削除して問題ありません。

### 5.3 削除する行の例

- 次のような 1 行を削除します。
  ```text
  ussoewwin <136552381+ussoewwin@users.noreply.github.com> cursoragent <cursor@cursor.sh>
  ```
- 上記は「cursoragent を ussoewwin として表示する」という意味なので、cursoragent を表示からも消すなら、この行を丸ごと削除します。
- 編集後、`.mailmap` の変更は通常どおりコミットし、push して構いません。

---

## 6. リモートへの反映（force push）

### 6.1 なぜ force push か

- `filter-branch` によりコミットの内容（メッセージや作者）が変わり、**コミットハッシュが変わります。** リモートの履歴と「同じコミット」ではなくなるため、通常の `git push` は拒否されます。
- 開発者があなたお一人であり、main の履歴を上書きしてよい前提であれば、**force push** でリモートを書き換え済みの履歴に合わせます。

### 6.2 推奨コマンド

- **`--force-with-lease`** を使うことを推奨します。リモートがあなたの知らない新しいコミットで進んでいるときは push を拒否するため、誤って他人の作業を上書きするリスクを減らせます。
  ```bash
  git push --force-with-lease origin main
  ```
- 確実に上書きしてよい場合に限り、`git push --force origin main` も使えますが、通常は `--force-with-lease` で十分です。

### 6.3 他ブランチがある場合

- 同様の履歴を持つブランチがリモートに存在する場合は、そのブランチに対しても書き換えと force push が必要です。手順は main と同様です。

---

## 7. GitHub の「Contributors」から cursoragent を消す（キャッシュ対策）

### 7.1 現象

- 履歴の書き換えと force push を完了しても、GitHub のリポジトリトップや「Insights → Contributors」に、しばらく **cursoragent が表示され続ける** ことがあります。
- GitHub は貢献者情報をキャッシュしており、履歴が書き換わっても即時には再計算されないためです。

### 7.2 デフォルトブランチの切り替えで再計算を促す

- 多くの場合、**デフォルトブランチを一時的に変更し、すぐに元に戻す**ことで、キャッシュが更新され、Contributors が再計算されます。
  1. GitHub のリポジトリページで **Settings** → **General** を開く。
  2. **Default branch** の右側のスイッチアイコンをクリック。
  3. いったん **main 以外**（例: 一時用の `temp` ブランチ。存在しなければ事前に作成）を選択し、**Update** をクリック。
  4. 再度 **Default branch** を **main** に戻し、**Update** をクリック。
- 数分以内に Contributors の表示が更新され、cursoragent が消えることがあります。

### 7.3 それでも消えない場合

- 時間をおいて（数時間〜1 日）再度確認してください。
- それでも cursoragent が残る場合は、**GitHub Support** に問い合わせ、「コミット履歴を書き換え、Author をすべて自分に統一したが、Contributors のキャッシュが更新されず cursoragent が表示されたままである。キャッシュの再計算をお願いしたい」旨を伝える方法があります。

---

## 8. 検証手順（すべて完了したあと）

### 8.1 ローカルでの確認

- **コミットメッセージに Co-authored-by が残っていないか**
  ```bash
  git log -20 --format=%B | grep -i "Co-authored-by"
  ```
  - 何も出力されなければ、直近 20 件のメッセージには該当行がありません。必要に応じて `-20` を `-100` などに変更して全履歴を確認できます。
- **作者に cursoragent が残っていないか**
  ```bash
  git log --all --format="%an %ae" | sort -u
  ```
  - 表示される名前・メールの組み合わせが、あなたのみ（および意図した共同作業者のみ）であることを確認してください。

### 8.2 リモート・GitHub での確認

- `git push` 後、GitHub の該当ブランチの **コミット一覧** を開き、各コミットのメッセージに「Co-authored-by: … Cursor …」が含まれていないことを確認します。
- **Insights → Contributors** で、cursoragent（または Cursor Agent）が一覧に表示されていないことを確認します。キャッシュの関係で、反映まで少し時間がかかることがあります。

---

## 9. トラブルシュート

### 9.1 commit-msg フックが動かない

- **パス**: 必ず **リポジトリの `.git/hooks/commit-msg`** であることを確認してください。作業ツリー直下の `hooks` では動作しません。
- **改行コード**: ファイルの改行が LF であることを推奨します。CRLF でも動くことが多いですが、Git Bash で実行エラーになる場合は LF に統一してください。
- **実行権限**: Unix 系・Git Bash では `chmod +x .git/hooks/commit-msg` を実行してください。
- **フックが無効化されていないか**: `git config core.hooksPath` が別のディレクトリを指していると、`.git/hooks/` が使われません。その場合は、指定されているディレクトリに同じ `commit-msg` を配置するか、`core.hooksPath` を未設定に戻してください。

### 9.2 filter-branch で sed や mktemp がない / エラーになる

- Git for Windows を利用している場合、Git Bash から実行すれば `sed` と `mktemp` は通常利用可能です。**コマンドプロンプトではなく Git Bash** で実行してください。
- 別の環境では、`sed` のオプションや正規表現が異なることがあります。その場合は、セクション 2 の commit-msg フックと同様のロジック（「Co-authored-by: … Cursor …」を含む行を削除）を、その環境で使えるコマンド（awk や Python など）で書き直してください。

### 9.3 force push が拒否される

- リモートがあなたのローカルより進んでいる（誰かが push した、または別端末で push した）場合、`--force-with-lease` は push を拒否します。その場合は、まず `git fetch origin` で最新を取得し、必要ならマージまたはリベースしたうえで、再度 `git push --force-with-lease origin main` を実行してください。
- どうしても上書きしてよい場合は `git push --force origin main` を使えますが、共同で main を使っている場合は必ず合意のうえで行ってください。

### 9.4 複数ブランチを同じ手順で書き換える場合

- 各ブランチに対して、セクション 3 および必要に応じてセクション 4 の `filter-branch` を、`main` の部分をそのブランチ名に変えて実行します。
- 各ブランチについて `refs/original/refs/heads/<ブランチ名>` を `git update-ref -d` で削除し、それぞれ `git push --force-with-lease origin <ブランチ名>` で反映します。

---

## 10. まとめチェックリスト

- [ ] **commit-msg フック**を `.git/hooks/commit-msg` に設置し、Co-authored-by（Cursor）行を削除する処理が入っている。
- [ ] 既存の main のメッセージから Co-authored-by を削除する **filter-branch（--msg-filter）** を実行した。
- [ ] 必要に応じて、cursoragent を作者とするコミットをあなたの名前に差し替える **filter-branch（--env-filter）** を実行した。
- [ ] **refs/original/refs/heads/main**（および他ブランチ分）を削除した。
- [ ] **.mailmap** から cursoragent のマッピング行を削除した。
- [ ] **force push（--force-with-lease）** でリモートに反映した。
- [ ] GitHub の **Contributors** に cursoragent が残る場合は、**デフォルトブランチの切り替え** でキャッシュ更新を試した。
- [ ] **git log** でメッセージ・作者に Co-authored-by / cursoragent が残っていないことを確認した。

以上で、Cursor の Co-authored-by と開発者表示を完全に消す手順の解説を終えます。一文字でも手抜きなく記載しました。実施時は、必ずリポジトリのバックアップまたは別クローンで動作確認したうえで、本番の main に対して実行してください。
