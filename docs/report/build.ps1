param(
    [switch]$Clean,
    [switch]$Watch
)

$ErrorActionPreference = "Stop"
$Main = "main"

if ($Clean) {
    Remove-Item -ErrorAction SilentlyContinue "$Main.aux", "$Main.bbl", "$Main.blg", "$Main.log", "$Main.out", "$Main.toc", "$Main.pdf"
    Write-Host "Cleaned LaTeX build artifacts."
    exit 0
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64",
        "C:\Program Files\MiKTeX\miktex\bin\x64"
    )
    foreach ($dir in $candidates) {
        if (Test-Path (Join-Path $dir "pdflatex.exe")) {
            $env:PATH = "$dir;$env:PATH"
            break
        }
    }
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Error "pdflatex not found. Please install MiKTeX (https://miktex.org/download) or TeX Live first."
}

function Invoke-Build {
    pdflatex -interaction=nonstopmode "$Main.tex" | Out-Null
    bibtex "$Main" | Out-Null
    pdflatex -interaction=nonstopmode "$Main.tex" | Out-Null
    pdflatex -interaction=nonstopmode "$Main.tex"
    Write-Host "Built $Main.pdf successfully at $(Get-Date -Format 'HH:mm:ss')."
}

Invoke-Build

if ($Watch) {
    Write-Host "Watching *.tex and references.bib for changes (Press Ctrl+C to stop)..."
    $lastHash = ""
    while ($true) {
        Start-Sleep -Milliseconds 700
        $files = Get-ChildItem -File -Include *.tex, references.bib
        $curHash = ($files | ForEach-Object { "$($_.Name):$($_.LastWriteTimeUtc.Ticks)" }) -join "|"
        if ($lastHash -eq "") {
            $lastHash = $curHash
            continue
        }
        if ($curHash -ne $lastHash) {
            $lastHash = $curHash
            Write-Host "Change detected, rebuilding..."
            try {
                Invoke-Build
            } catch {
                Write-Warning $_
            }
        }
    }
}
