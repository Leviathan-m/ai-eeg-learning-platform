Param(
    [string]$Message = "chore: automated update",
    [switch]$All,
    [string]$Branch = "master",
    [string]$UserName = $env:GIT_USER_NAME,
    [string]$UserEmail = $env:GIT_USER_EMAIL
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Move to repo root regardless of current directory
$repoRoot = (git rev-parse --show-toplevel) 2>$null
if (-not $repoRoot) { throw "Not inside a git repository." }
Set-Location $repoRoot

# Ensure git user is configured
if (-not (git config user.name)) {
    if ($UserName) { git config user.name "$UserName" } else { git config user.name "automation-bot" }
}
if (-not (git config user.email)) {
    if ($UserEmail) { git config user.email "$UserEmail" } else { git config user.email "automation@example.com" }
}

# Stage changes
if ($All) {
    git add -A
} else {
    git add -A
}

# Commit if there is anything to commit
if ((git status --porcelain) -ne "") {
    git commit -m "$Message"
}

# Optionally use tokenized remote for CI/automation
$token = $env:GITHUB_PUSH_TOKEN
$repoUrl = $env:GITHUB_REPO_URL
if ($token -and $repoUrl) {
    $authedUrl = $repoUrl -replace "https://", "https://$token@"
    git remote remove authed 2>$null
    git remote add authed $authedUrl
    git push authed $Branch
} else {
    git push origin $Branch
}

Write-Host "Pushed to branch '$Branch' successfully."


