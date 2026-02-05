# PowerShell version of Ollama Health Check
# Quick health check for Ollama

$OLLAMA_HOST = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "http://localhost:11434" }
$OLLAMA_MODEL_ENV = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "" }

Write-Host "🔍 Quick Ollama Health Check" -ForegroundColor Cyan

try {
    # Check service
    $response = Invoke-WebRequest -Uri "$OLLAMA_HOST/api/tags" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Ollama service is running" -ForegroundColor Green

        # Parse models from JSON response
        $modelsData = $response.Content | ConvertFrom-Json
        $modelCount = if ($modelsData.models) { $modelsData.models.Count } else { 0 }
        Write-Host "📊 Available models: $modelCount" -ForegroundColor Yellow

        if ($modelCount -gt 0) {
            Write-Host "📋 Model list:" -ForegroundColor Yellow
            foreach ($model in $modelsData.models) {
                Write-Host "  - $($model.name)" -ForegroundColor Gray
            }

            # Determine model to test
            if ($OLLAMA_MODEL_ENV) {
                $testModel = $OLLAMA_MODEL_ENV
            } else {
                $testModel = $modelsData.models[0].name
            }

            Write-Host "🧭 Using model: $testModel" -ForegroundColor Cyan

            # Test basic functionality
            Write-Host "🧪 Testing basic inference..." -ForegroundColor Yellow
            $testPayload = @{
                model = $testModel
                prompt = "Hello"
                stream = $false
            } | ConvertTo-Json

            $testResponse = Invoke-WebRequest -Uri "$OLLAMA_HOST/api/generate" -Method POST -Body $testPayload -ContentType "application/json" -UseBasicParsing -TimeoutSec 30
            $testData = $testResponse.Content | ConvertFrom-Json

            if ($testData.response -and -not $testData.error) {
                Write-Host "✅ Basic inference working" -ForegroundColor Green
            } else {
                Write-Host "❌ Basic inference failed" -ForegroundColor Red
                Write-Host "↪ Response: $($testResponse.Content.Substring(0, [Math]::Min(300, $testResponse.Content.Length)))" -ForegroundColor Gray
                Write-Host "💡 Tip: set OLLAMA_MODEL to a specific installed model, e.g.:" -ForegroundColor Yellow
                Write-Host "   `$env:OLLAMA_MODEL='llama3:instruct'; .\bash\ollama_healthcheck.ps1" -ForegroundColor Gray
            }
        } else {
            Write-Host "⚠️  No models available. Install models with: ollama pull <model-name>" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "❌ Ollama service not available" -ForegroundColor Red
    Write-Host "💡 Start with: ollama serve" -ForegroundColor Yellow
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
