$ErrorActionPreference = "Stop"

$hostAlias = "715"
Write-Host "[$hostAlias] killing all remote tmux sessions..."
& ssh $hostAlias "tmux kill-server 2>/dev/null || true"

if ($LASTEXITCODE -ne 0) {
    throw "[$hostAlias] 远程 tmux 清理失败。"
}
