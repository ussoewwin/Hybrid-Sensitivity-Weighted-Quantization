param(
  [string]$Base,                      # FP16 Flux1 DiT (the measurement baseline)
  [string]$Int8,                      # the complete native ConvRot INT8 pack (the build input)
  [string]$Impact,                    # impact json from Flux1/diag_impact.py
  [string]$ClipPath,                  # t5xxl
  [string]$ClipLPath,                 # clip_l
  [string]$ComfyPath = "D:\USERFILES\GitHub\hswq\ComfyUI-master",
  [string]$OutDir = ".",              # where the packs are written
  [string]$LogDir = "D:\USERFILES\GitHub\hswq\log",
  [string]$Ks = "100,130,160,170,190,220",   # the K values to sweep
  [int]$Steps = 12,
  [string]$Seeds = ""
)
$ErrorActionPreference = "Stop"
$py   = "D:\USERFILES\ComfyUI\python_embeded\python.exe"
$repo = "D:\USERFILES\GitHub\hswq"
$env:PYTHONIOENCODING = "utf-8"

if (-not $Seeds) {
  $Seeds = "42,137,849,2024,7391,18429,53082,149206,382715,826401,1938502,4710928,8391642," +
           "15820493,36192847,71058294,128491703,285039184,491730285,762019483,938174026," +
           "1409285713,2683910547,3851729406,4195820371"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stem = [IO.Path]::GetFileNameWithoutExtension($Base)
$rows = @()

foreach ($k in ($Ks -split ",")) {
  $k = $k.Trim()
  if (-not $k) { continue }
  $pack = Join-Path $OutDir "${stem}_hybrid_w${k}_convrot_w4a8.safetensors"
  $log  = Join-Path $LogDir "flux1_w4a8_k${k}.log"

  Write-Host "[sweep] K=$k -> building $pack"
  & $py -u "$repo\Flux1\gen_reverse_w4a8.py" $k $pack $Int8 $Impact --out-dir $OutDir *> $log
  if (-not (Test-Path $pack)) { throw "build failed for K=$k (see $log)" }
  $size = (Get-Item $pack).Length

  Write-Host "[sweep] K=$k -> gate (steps=$Steps)"
  & $py -u "$repo\benchmark\flux1_convrot_w4a8_traj_compare.py" `
      --fp16 $Base --w4a8 $pack `
      --clip_path $ClipPath --clip_l_path $ClipLPath --comfy_path $ComfyPath `
      --steps $Steps --seeds $Seeds *>> $log

  # The comparator prints the summary the sweep needs; read it back from the log.
  $mean = $null; $min = $null; $max = $null; $bif = $null; $same = $null
  foreach ($line in (Get-Content $log)) {
    if ($line -match 'final-cosine:\s+min=([0-9.]+)\s+mean=([0-9.]+)\s+max=([0-9.]+)') {
      $min = $Matches[1]; $mean = $Matches[2]; $max = $Matches[3]
    }
    if ($line -match 'bifurcated seeds\s*:\s*(\d+)/(\d+)') { $bif = $Matches[1] }
    if ($line -match 'same-image seeds\s*:\s*(\d+)/(\d+)') { $same = $Matches[1] }
  }
  $rows += [pscustomobject]@{
    K = $k; Bytes = $size; Mean = $mean; Min = $min; Max = $max
    Bifurcated = $bif; SameImage = $same; Pack = $pack
  }
  Write-Host ("[sweep] K={0} size={1} mean={2} min={3} bifurcated={4}" -f $k,$size,$mean,$min,$bif)
}

$csv = Join-Path $LogDir "flux1_w4a8_sweep.csv"
$rows | Export-Csv -Path $csv -NoTypeInformation -Encoding utf8
Write-Host "[sweep] csv -> $csv"

& $py -u "$repo\benchmark\build_flux1_w4a8_sweep_md.py" $csv `
    (Join-Path $LogDir "flux1_w4a8_sweep.md")
Write-Host "[sweep] md  -> $(Join-Path $LogDir 'flux1_w4a8_sweep.md')"
