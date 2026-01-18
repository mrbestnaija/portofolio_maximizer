# Script to remove memory files from git index by hash
$files = @(
    @{hash="09044ac3b0ef9c07e199942199326a8181571c7a"; path=":memory:"},
    @{hash="09044ac3b0ef9c07e199942199326a8181571c7a"; path="\357\200\272memory\357\200\272"},
    @{hash="fe9ac2845eca6fe6da8a63cd096d9cf9e24ece10"; path="\357\200\272memory\357\200\272-shm"},
    @{hash="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"; path="\357\200\272memory\357\200\272-wal"}
)

# Get all index entries
$indexEntries = git ls-files --stage

# Filter out memory files
$filtered = $indexEntries | Where-Object {
    $entry = $_
    $shouldKeep = $true
    foreach ($file in $files) {
        if ($entry -match $file.hash) {
            $shouldKeep = $false
            break
        }
    }
    $shouldKeep
}

# Write filtered index
$filtered | git update-index --index-info
