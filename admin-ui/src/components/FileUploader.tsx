import React, { useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { toast } from 'sonner'

type FileIngestOptions = {
  chunk_size: number
  chunk_overlap: number
  splitter: 'tokens' | 'chars' | 'lines' | 'paragraphs'
  max_chunks?: number
  delimiter: string
  sheet?: string
  normalize_whitespace: boolean
  lowercase: boolean
  metadata?: Record<string, any>
}

type FileIngestResult = {
  file_name: string
  num_chunks: number
  num_embeddings: number
  metadata?: Record<string, any>
  warnings: string[]
}

type FileIngestResponse = {
  object: string
  results: FileIngestResult[]
  total_files: number
  total_chunks: number
  total_embeddings: number
}

interface FileUploaderProps {
  vectorStoreId: string
  apiKey: string
  apiBase?: string
  onSuccess?: () => void
}

export function FileUploader({ vectorStoreId, apiKey, apiBase = '', onSuccess }: FileUploaderProps) {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [results, setResults] = useState<FileIngestResponse | null>(null)
  
  // Options state
  const [options, setOptions] = useState<FileIngestOptions>({
    chunk_size: 1000,
    chunk_overlap: 200,
    splitter: 'chars',
    max_chunks: undefined,
    delimiter: '\n',
    sheet: undefined,
    normalize_whitespace: true,
    lowercase: false,
    metadata: undefined
  })
  
  const [showAdvanced, setShowAdvanced] = useState(false)

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(e.target.files || [])
    
    // Validate file types
    const allowedTypes = ['.pdf', '.docx', '.xlsx', '.csv', '.txt']
    const validFiles = selectedFiles.filter(file => {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase()
      return allowedTypes.includes(ext)
    })
    
    if (validFiles.length !== selectedFiles.length) {
      toast.error('Some files were skipped. Only PDF, DOCX, XLSX, CSV, and TXT files are supported.')
    }
    
    setFiles(prev => [...prev, ...validFiles])
    e.target.value = '' // Reset input
  }

  function removeFile(index: number) {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  async function handleUpload() {
    if (!vectorStoreId.trim()) {
      toast.error('Please enter a vector store ID')
      return
    }

    if (files.length === 0) {
      toast.error('Please select at least one file')
      return
    }

    setUploading(true)
    setResults(null)

    try {
      const formData = new FormData()
      
      // Add files
      files.forEach(file => {
        formData.append('files', file)
      })
      
      // Add options as JSON
      const optionsJson = JSON.stringify(options)
      formData.append('options', optionsJson)

      const response = await fetch(`${apiBase}/v1/vector_stores/${vectorStoreId}/files`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`
        },
        body: formData
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `HTTP ${response.status}`)
      }

      const result: FileIngestResponse = await response.json()
      setResults(result)
      
      // Show success message
      toast.success(
        `Successfully processed ${result.total_files} file(s). Created ${result.total_embeddings} embeddings.`
      )
      
      // Show warnings if any
      const warnings = result.results.flatMap(r => r.warnings)
      if (warnings.length > 0) {
        warnings.forEach(warning => {
          toast.warning(`${warning}`)
        })
      }
      
      // Reset files
      setFiles([])
      
      // Call success callback
      if (onSuccess) {
        onSuccess()
      }
    } catch (error: any) {
      toast.error(`Upload failed: ${error.message}`)
      console.error('Upload error:', error)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>File Upload</CardTitle>
          <p className="text-sm text-muted-foreground">
            Upload PDF, DOCX, XLSX, CSV, or TXT files to create embeddings
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File selection */}
          <div className="grid gap-2">
            <Label htmlFor="files">Select Files</Label>
            <Input
              id="files"
              type="file"
              multiple
              accept=".pdf,.docx,.xlsx,.csv,.txt"
              onChange={handleFileSelect}
              disabled={uploading}
            />
            <p className="text-xs text-muted-foreground">
              Supported formats: PDF, DOCX, XLSX, CSV, TXT
            </p>
          </div>

          {/* Selected files */}
          {files.length > 0 && (
            <div className="space-y-2">
              <Label>Selected Files ({files.length})</Label>
              <div className="space-y-2 max-h-48 overflow-y-auto border rounded-md p-2">
                {files.map((file, index) => (
                  <div key={index} className="flex items-center justify-between text-sm p-2 hover:bg-muted rounded">
                    <span className="truncate flex-1">{file.name}</span>
                    <span className="text-muted-foreground mx-2">
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => removeFile(index)}
                      disabled={uploading}
                    >
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Basic options */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="grid gap-2">
              <Label htmlFor="chunk-size">Chunk Size</Label>
              <Input
                id="chunk-size"
                type="number"
                min="1"
                value={options.chunk_size}
                onChange={e => setOptions({ ...options, chunk_size: parseInt(e.target.value) || 1000 })}
                disabled={uploading}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="chunk-overlap">Chunk Overlap</Label>
              <Input
                id="chunk-overlap"
                type="number"
                min="0"
                value={options.chunk_overlap}
                onChange={e => setOptions({ ...options, chunk_overlap: parseInt(e.target.value) || 200 })}
                disabled={uploading}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="splitter">Splitter</Label>
              <select
                id="splitter"
                value={options.splitter}
                onChange={e => setOptions({ ...options, splitter: e.target.value as any })}
                disabled={uploading}
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="chars">Characters</option>
                <option value="tokens">Tokens</option>
                <option value="lines">Lines</option>
                <option value="paragraphs">Paragraphs</option>
              </select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="max-chunks">Max Chunks (optional)</Label>
              <Input
                id="max-chunks"
                type="number"
                min="1"
                placeholder="No limit"
                value={options.max_chunks || ''}
                onChange={e => setOptions({ 
                  ...options, 
                  max_chunks: e.target.value ? parseInt(e.target.value) : undefined 
                })}
                disabled={uploading}
              />
            </div>
          </div>

          {/* Advanced options toggle */}
          <Button
            variant="outline"
            onClick={() => setShowAdvanced(!showAdvanced)}
            disabled={uploading}
          >
            {showAdvanced ? 'Hide' : 'Show'} Advanced Options
          </Button>

          {/* Advanced options */}
          {showAdvanced && (
            <div className="grid gap-4 border rounded-md p-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="grid gap-2">
                  <Label htmlFor="delimiter">Delimiter (for CSV/XLSX)</Label>
                  <Input
                    id="delimiter"
                    value={options.delimiter}
                    onChange={e => setOptions({ ...options, delimiter: e.target.value })}
                    disabled={uploading}
                    placeholder="\n"
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="sheet">Sheet Name (for XLSX, optional)</Label>
                  <Input
                    id="sheet"
                    value={options.sheet || ''}
                    onChange={e => setOptions({ ...options, sheet: e.target.value || undefined })}
                    disabled={uploading}
                    placeholder="First sheet"
                  />
                </div>
              </div>

              <div className="flex gap-4">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="normalize-whitespace"
                    checked={options.normalize_whitespace}
                    onChange={e => setOptions({ ...options, normalize_whitespace: e.target.checked })}
                    disabled={uploading}
                  />
                  <Label htmlFor="normalize-whitespace">Normalize Whitespace</Label>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="lowercase"
                    checked={options.lowercase}
                    onChange={e => setOptions({ ...options, lowercase: e.target.checked })}
                    disabled={uploading}
                  />
                  <Label htmlFor="lowercase">Lowercase</Label>
                </div>
              </div>
            </div>
          )}

          {/* Upload button */}
          <Button
            onClick={handleUpload}
            disabled={uploading || files.length === 0 || !vectorStoreId.trim()}
            className="w-full"
          >
            {uploading ? 'Uploading...' : `Upload ${files.length} file(s)`}
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <Card>
          <CardHeader>
            <CardTitle>Upload Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="p-3 border rounded-md">
                  <div className="text-sm text-muted-foreground">Total Files</div>
                  <div className="text-2xl font-bold">{results.total_files}</div>
                </div>
                <div className="p-3 border rounded-md">
                  <div className="text-sm text-muted-foreground">Total Chunks</div>
                  <div className="text-2xl font-bold">{results.total_chunks}</div>
                </div>
                <div className="p-3 border rounded-md">
                  <div className="text-sm text-muted-foreground">Total Embeddings</div>
                  <div className="text-2xl font-bold">{results.total_embeddings}</div>
                </div>
              </div>

              <div className="space-y-2">
                <Label>File Results</Label>
                <div className="space-y-2 max-h-64 overflow-y-auto border rounded-md p-2">
                  {results.results.map((result, index) => (
                    <div key={index} className="p-3 border rounded-md">
                      <div className="font-medium">{result.file_name}</div>
                      <div className="text-sm text-muted-foreground mt-1">
                        {result.num_chunks} chunks • {result.num_embeddings} embeddings
                      </div>
                      {result.warnings.length > 0 && (
                        <div className="mt-2 text-sm text-yellow-600">
                          {result.warnings.map((warning, i) => (
                            <div key={i}>⚠️ {warning}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

