# One-command laptop run (Windows PowerShell 5+). Resumable: re-run the same command after any interruption.
#   powershell -ExecutionPolicy Bypass -File code\laptop\run_laptop_e2e.ps1            # everything
#   ... -Stage gate | bench | e2e | summary                                              # one stage
# Stages: 0 preflight -> 1 speed gate CPU vs Vulkan -> 2 blueprint indexing benchmark (10K-file repo) ->
#         3 50-task guarded repair with Gemma 4 E2B + official SWE-bench scoring -> 4 summary.
# A background sampler logs RAM every ~5 s (llama-server working set, Docker VM, system) to results\laptop\ram_samples.csv.
param([ValidateSet('all','gate','bench','e2e','rescore','score','summary')][string]$Stage = 'all', [int]$Limit = 0, [string]$Tasks = 'e2e_tasks50.json')
# 'Continue': Windows PowerShell 5.1 turns *redirected* native stderr (docker progress) into terminating errors under
# 'Stop'. Failures are still caught: every native call checks $LASTEXITCODE and cmdlets use -ErrorAction Stop.
$ErrorActionPreference = 'Continue'
# Windows PowerShell 5.1 defaults to TLS 1.0/1.1, which GitHub refuses ("unexpected error occurred on a send").
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$Final = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$SwePaper = Join-Path (Split-Path $Final -Parent) 'swe_paper'
$Out = Join-Path $Final 'results\laptop'; New-Item -ItemType Directory -Force $Out | Out-Null
$Tools = Join-Path $SwePaper 'local_runs\tools'; New-Item -ItemType Directory -Force $Tools | Out-Null
$Log = Join-Path $Out 'run.log'
function Log($m) { $s = "$(Get-Date -Format s) $m"; Write-Host $s; Add-Content $Log $s }
function Sha($p) { (Get-FileHash $p -Algorithm SHA256).Hash.ToLower() }

$Model = Join-Path $Tools 'gemma-4-E2B_q4_0-it.gguf'
$ModelUrl = 'https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/675cff42a74c774d6cb76f76d8eacb49b48c9b93/gemma-4-E2B_q4_0-it.gguf'
$ModelSha = 'fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634'
$Builds = @{
  cpu    = @{ zip = 'llama-b12211-bin-win-cpu-x64.zip';    sha = '23729000408c6968faf5e322afc99319d3b8c576ba872ef7663d9b89810090f3' }
  vulkan = @{ zip = 'llama-b12211-bin-win-vulkan-x64.zip'; sha = '20954cbad1a4626436c6ebaed5e48477f68427b075934689f360301ac426a2f1' }
}
$Port = 8080
$Threads = [int](Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfCores
$ServerArgs = @('-c','8192','--parallel','1','-fa','on','--jinja','--seed','42','--cache-type-k','q4_0','--cache-type-v','q4_0','--swa-full','--host','0.0.0.0','--port',"$Port")

function Get-Asset($url, $path, $sha) {
  if ((Test-Path $path) -and ((Sha $path) -eq $sha)) { return }
  Log "downloading $url"
  try { Invoke-WebRequest $url -OutFile "$path.part" -UseBasicParsing -ErrorAction Stop; Move-Item -Force "$path.part" $path -ErrorAction Stop }
  catch { throw "Download failed ($url). Download it in your browser and save it as $path, then re-run. $_" }
  if ((Sha $path) -ne $sha) { throw "SHA-256 mismatch for $path" }
}
function Get-Server($backend) {
  $b = $Builds[$backend]; $zip = Join-Path $Tools $b.zip
  Get-Asset "https://github.com/ggml-org/llama.cpp/releases/download/b12211/$($b.zip)" $zip $b.sha
  $dir = Join-Path $Tools "llama-$backend"
  if (!(Test-Path (Join-Path $dir 'llama-server.exe'))) { Expand-Archive -Force $zip $dir -ErrorAction Stop }
  return (Get-ChildItem $dir -Recurse -Filter llama-server.exe | Select-Object -First 1).FullName
}
function Stop-Server { Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2 }
function Start-Server($backend) {
  Stop-Server
  $exe = Get-Server $backend
  $a = @('-m', $Model, '-t', "$Threads") + $ServerArgs
  if ($backend -eq 'vulkan') { $a += @('-ngl','999') } else { $a += @('-ngl','0') }
  $p = Start-Process $exe -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $Out "llama_$backend.err.log") -RedirectStandardOutput (Join-Path $Out "llama_$backend.out.log")
  $t0 = Get-Date
  for ($i = 0; $i -lt 600; $i++) {
    Start-Sleep 1
    try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5 -ErrorAction Stop).status -eq 'ok') { break } } catch {}
    if ($p.HasExited) { throw "llama-server ($backend) exited; see llama_$backend.err.log" }
  }
  Log "llama-server $backend up in $([math]::Round(((Get-Date)-$t0).TotalSeconds,1)) s (pid $($p.Id))"
  return $p
}
function Invoke-Chat($user, $n) {
  $body = @{ messages = @(@{role='system'; content='You are a helpful assistant.'}, @{role='user'; content=$user}); temperature = 0; seed = 42;
             max_tokens = $n; cache_prompt = $false; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 5
  return Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 3600 -ErrorAction Stop
}

# ---------------------------------------------------------------- RAM sampler (background job)
function Start-Sampler {
  $csv = Join-Path $Out 'ram_samples.csv'
  return Start-Job -ArgumentList $csv -ScriptBlock {
    param($csv)
    if (!(Test-Path $csv)) { 'utc_epoch,sys_used_mib,sys_total_mib,llama_ws_mib,llama_peak_ws_mib,vmmem_ws_mib,docker_containers_mib' | Out-File $csv -Encoding ascii }
    while ($true) {
      $os = Get-CimInstance Win32_OperatingSystem
      $tot = [math]::Round($os.TotalVisibleMemorySize / 1024); $used = $tot - [math]::Round($os.FreePhysicalMemory / 1024)
      $l = Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object -First 1
      $lw = if ($l) { [math]::Round($l.WorkingSet64 / 1MB) } else { 0 }; $lp = if ($l) { [math]::Round($l.PeakWorkingSet64 / 1MB) } else { 0 }
      $vm = (Get-Process vmmem, vmmemWSL -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum
      $vmm = if ($vm) { [math]::Round($vm / 1MB) } else { 0 }
      $dc = 0
      try { docker stats --no-stream --format '{{.MemUsage}}' 2>$null | ForEach-Object { $u = ($_ -split '/')[0].Trim()
            if ($u -match '([\d.]+)\s*GiB') { $dc += [double]$matches[1] * 1024 } elseif ($u -match '([\d.]+)\s*MiB') { $dc += [double]$matches[1] } } } catch {}
      "$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()),$used,$tot,$lw,$lp,$vmm,$([math]::Round($dc))" | Out-File $csv -Append -Encoding ascii
      Start-Sleep 5
    }
  }
}

# ---------------------------------------------------------------- 0 preflight
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$os = Get-CimInstance Win32_OperatingSystem
$gpu = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '
$disk = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
$hw = [ordered]@{ cpu = $cpu.Name.Trim(); cores = $cpu.NumberOfCores; threads = $cpu.NumberOfLogicalProcessors;
  ram_total_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2); gpu = $gpu; c_free_gb = $disk; os = $os.Caption;
  docker = (docker version --format '{{.Server.Version}}' 2>$null); wslconfig = (Get-Content "$env:USERPROFILE\.wslconfig" -ErrorAction SilentlyContinue) -join ' | ' }
$hw | ConvertTo-Json | Out-File (Join-Path $Out 'hardware.json') -Encoding utf8
Log "preflight: $($hw | ConvertTo-Json -Compress)"
if (!$hw.docker) { throw 'Docker Desktop is not running.' }
if ($disk -lt 25 -or $disk -eq $null) { throw "Only $disk GB free on C: (need >= 25 GB: one task image at a time + env images)." }
Get-Asset $ModelUrl $Model $ModelSha
$sampler = Start-Sampler

try {
# ---------------------------------------------------------------- 1 speed gate CPU vs Vulkan
$gateFile = Join-Path $Out 'speed_gate.json'
if ($Stage -in 'all','gate' -or !(Test-Path $gateFile)) {
  $gate = [ordered]@{}
  $prompt = ('def f(x):\n    return x + 1\n' * 120) + 'Explain what this code does in two sentences.'
  foreach ($b in 'cpu','vulkan') {
    try {
      $p = Start-Server $b
      $null = Invoke-Chat 'warm up' 8
      $r = Invoke-Chat $prompt 128
      $p.Refresh()
      $gate[$b] = [ordered]@{ prompt_n = $r.timings.prompt_n; prompt_tok_s = [math]::Round($r.timings.prompt_per_second, 2);
        gen_n = $r.timings.predicted_n; gen_tok_s = [math]::Round($r.timings.predicted_per_second, 2);
        peak_ws_mib = [math]::Round($p.PeakWorkingSet64 / 1MB); threads = $Threads;
        # cost of a typical agent turn: 1,500 new prompt tokens + 200 generated tokens
        turn_s = [math]::Round(1500 / $r.timings.prompt_per_second + 200 / $r.timings.predicted_per_second, 1) }
      Log "gate $b : $($gate[$b] | ConvertTo-Json -Compress)"
    } catch { $gate[$b] = @{ error = "$_" }; Log "gate $b failed: $_" }
    Stop-Server
  }
  $ok = @($gate.Keys | Where-Object { $gate[$_].turn_s })
  if ($ok.Count -eq 0) { throw 'no llama.cpp backend works' }
  $gate['chosen'] = ($ok | Sort-Object { $gate[$_].turn_s } | Select-Object -First 1)
  $gate | ConvertTo-Json -Depth 4 | Out-File $gateFile -Encoding utf8
}
$backend = (Get-Content $gateFile | ConvertFrom-Json).chosen
Log "backend: $backend"

# ---------------------------------------------------------------- agent image
Copy-Item (Join-Path $Tools 'windows_ca.pem') (Join-Path $PSScriptRoot 'windows_ca.pem') -Force -ErrorAction SilentlyContinue
if (!(Test-Path (Join-Path $PSScriptRoot 'windows_ca.pem'))) { '' | Out-File (Join-Path $PSScriptRoot 'windows_ca.pem') -Encoding ascii }
$null = cmd /c "docker image inspect gemnav-agent:1 >nul 2>&1"
if ($LASTEXITCODE -ne 0) { Log 'building gemnav-agent:1 (one time)'; docker build -t gemnav-agent:1 -f (Join-Path $PSScriptRoot 'Dockerfile.agent') $PSScriptRoot; if ($LASTEXITCODE) { throw 'agent image build failed' } }
$common = @('run','--rm','-v','/var/run/docker.sock:/var/run/docker.sock','-v',"${Final}:/work",'-v','gemnav-cache:/cache','-v','gemnav-scratch:/scratch',
            '-v','swe-paper-swebench:/swebench','--add-host','host.docker.internal:host-gateway','-e','EMB_THREADS=4','gemnav-agent:1')

# ---------------------------------------------------------------- 2 blueprint indexing benchmark (llama-server stopped: CPU is not shared)
$benchFile = Join-Path $Out 'bench_index.json'
if ($Stage -in 'all','bench' -and !(Test-Path $benchFile)) {
  Stop-Server
  Log 'bench: cloning home-assistant/core (shallow) into the scratch volume and indexing it'
  docker @common bash -c "set -e; cd /scratch; [ -d core ] || git clone -q --depth 1 https://github.com/home-assistant/core.git core; python /work/code/experiments/bench_index.py --repo /scratch/core --embedder minilm --threads $Threads --out /work/results/laptop/bench_index.json"
  if ($LASTEXITCODE) { Log 'bench failed (see output above)' }
}

# ---------------------------------------------------------------- 3 end-to-end repair + scoring
if ($Stage -in 'all','e2e') {
  $p = Start-Server $backend
  $e4b = '/work/results/kaggle/predictions_gemma4-e4b-qat-q4_0-gemnav.jsonl'
  $lim = if ($Limit) { @('--limit', "$Limit") } else { @() }
  for ($attempt = 1; $attempt -le 5; $attempt++) {
    docker @common python /work/code/experiments/run_laptop_e2e.py --tasks /work/results/e2e_tasks50.json `
      --server "http://host.docker.internal:$Port" --out /work/results/laptop --embedder minilm --e4b-preds $e4b @lim
    if ($LASTEXITCODE -eq 0) { break }
    Log "agent container exited with $LASTEXITCODE (attempt $attempt); restarting llama-server and resuming"
    $p = Start-Server $backend
  }
}
if ($Stage -eq 'score') {
  Stop-Server
  docker @common python /work/code/experiments/run_laptop_e2e.py --tasks "/work/results/$Tasks" --out /work/results/laptop `
    --e4b-preds '/work/results/kaggle/predictions_gemma4-e4b-qat-q4_0-gemnav.jsonl' --score-only
}
if ($Stage -eq 'rescore') {
  docker @common python /work/code/experiments/run_laptop_e2e.py --tasks /work/results/e2e_tasks50.json --out /work/results/laptop `
    --e4b-preds '/work/results/kaggle/predictions_gemma4-e4b-qat-q4_0-gemnav.jsonl' --rescore-e4b
}
} finally {
  Stop-Server
  if ($sampler) { Stop-Job $sampler; Remove-Job $sampler -Force }
}

# ---------------------------------------------------------------- 4 summary
docker run --rm -v "${Final}:/work" gemnav-agent:1 python /work/code/experiments/summarize_e2e.py /work/results/laptop
Log 'done. Results: results\laptop\  (summary_e2e.json, e2e_laptop_tasks.jsonl, predictions_*.jsonl, reports\, ram_samples.csv, speed_gate.json, bench_index.json)'
