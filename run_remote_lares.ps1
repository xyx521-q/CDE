param(
    # 这里传的是 configs/ 下的配置基名，不带 .toml
    [string[]]$Configs715 = @(),
    [string]$SessionPrefix = "lares"
)

$ErrorActionPreference = "Stop"

if ($Configs715.Count -eq 0) {
    throw "请至少提供一个配置名（即 configs/ 下的配置基名，不带 .toml）。示例: .\run_remote_lares.ps1 -Configs715 pro_max_slow,pro_max_fast"
}

$targets = @(
    @{
        Alias = "715"
        RemoteDir = "/home/bless/projects/CDE"
        Configs = $Configs715
    }
)

function Normalize-ConfigName {
    param(
        [string]$Config
    )

    $normalized = $Config -replace "\\", "/"

    if ($normalized.StartsWith("./")) {
        $normalized = $normalized.Substring(2)
    }

    if ($normalized.StartsWith("configs/")) {
        $normalized = $normalized.Substring(8)
    }

    if ($normalized.EndsWith(".toml")) {
        $normalized = $normalized.Substring(0, $normalized.Length - 5)
    }

    return $normalized
}

function Invoke-RemoteLaunch {
    param(
        [string]$Alias,
        [string]$RemoteDir,
        [string[]]$Configs,
        [string]$SessionPrefix
    )

    if ($Configs.Count -eq 0) {
        Write-Host "[$Alias] 没有分配配置，跳过。"
        return
    }

    $normalizedConfigs = @($Configs | ForEach-Object { Normalize-ConfigName -Config $_ })
    $sessionName = "$SessionPrefix-$Alias"
    Write-Host "[$Alias] git pull -> tmux session: $sessionName -> configs: $($normalizedConfigs -join ', ')"

    $remoteScript = @'
set -euo pipefail

remote_dir="$1"
session_name="$2"
shift 2
configs=("$@")

if [ "${#configs[@]}" -eq 0 ]; then
  echo "No configs provided." >&2
  exit 1
fi

cd "$remote_dir"
git pull

tmux kill-session -t "$session_name" 2>/dev/null || true

first_config="${configs[0]}"
first_safe="${first_config//[^[:alnum:]_.-]/_}"
first_window="lares-${first_safe}"

tmux new-session -d -s "$session_name" -n "$first_window"
tmux send-keys -t "$session_name:$first_window" \
  "cd \"$remote_dir\" && uv run python src/direct_main.py --config ${first_config} --env-config vmas_transport &> ./${first_config}.txt" C-m

for config in "${configs[@]:1}"; do
  safe_config="${config//[^[:alnum:]_.-]/_}"
  window_name="lares-${safe_config}"
  tmux new-window -t "$session_name" -n "$window_name"
  tmux send-keys -t "$session_name:$window_name" \
    "cd \"$remote_dir\" && uv run python src/direct_main.py --config ${config} --env-config vmas_transport &> ./${config}.txt" C-m
done

sleep 1
tmux has-session -t "$session_name"
tmux list-panes -t "$session_name" -a -F '#{session_name} #{window_name} #{pane_current_command} #{pane_dead}' || true
'@

    ($remoteScript -replace "`r`n", "`n") | & ssh $Alias bash -s -- $RemoteDir $sessionName @normalizedConfigs

    if ($LASTEXITCODE -ne 0) {
        throw "[$Alias] 远程启动失败。"
    }
}

foreach ($target in $targets) {
    Invoke-RemoteLaunch `
        -Alias $target.Alias `
        -RemoteDir $target.RemoteDir `
        -Configs $target.Configs `
        -SessionPrefix $SessionPrefix
}
