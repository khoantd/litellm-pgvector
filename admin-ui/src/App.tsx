import React, { useEffect, useMemo, useState } from 'react'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Label } from './components/ui/label'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './components/ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './components/ui/tabs'
import { Toaster, toast } from 'sonner'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './components/ui/dialog'

type Settings = {
  server: { host: string; port: number }
  auth: { server_api_key: string | null }
  embedding: { model: string; base_url: string; api_key: string | null; dimensions: number }
  db_fields: Record<string, string>
}

type VectorStore = {
  id: string
  name: string
  usage_bytes: number
  file_counts: { completed: number; total: number; [key: string]: number }
  status: string
  created_at: number
  metadata?: Record<string, any>
}

type Embedding = {
  id: string
  vector_store_id: string
  content: string
  metadata?: Record<string, any>
  created_at: number
}

type VectorStoreStats = {
  vector_store_id: string
  period: string
  total_requests: number
  search_queries: number
  embeddings_created: number
  embeddings_deleted: number
  storage_bytes: number
  avg_response_time_ms: number
  error_count: number
  error_rate: number
  endpoint_stats?: Record<string, { count: number; avg_response_time_ms: number; error_count: number }>
}

type GlobalStats = {
  total_requests: number
  total_vector_stores: number
  total_embeddings: number
  total_storage_bytes: number
  avg_response_time_ms: number
  error_count: number
  error_rate: number
  endpoint_stats?: Record<string, { count: number; avg_response_time_ms: number; error_count: number }>
  top_vector_stores?: Array<{ vector_store_id: string; request_count: number; avg_response_time_ms: number }>
}

// Use VITE_API_BASE_URL if provided (for separated services), otherwise use relative URLs (same domain)
const apiBase = import.meta.env.VITE_API_BASE_URL || ''

// Authorization is handled in-memory inside App; no localStorage usage

function AuthGate({ onReady }: { onReady: (key: string) => void }) {
  const [key, setKey] = useState('')
  const [open, setOpen] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)

  async function save() {
    const trimmed = key.trim()
    if (!trimmed) return
    setSubmitting(true)
    setAuthError(null)
    try {
      const res = await fetch(`${apiBase}/v1/admin/settings`, {
        headers: { 'Authorization': `Bearer ${trimmed}` }
      })
      if (!res.ok) throw new Error('unauthorized')
      setOpen(false)
      onReady(trimmed)
    } catch (e: any) {
      setAuthError('Invalid or unauthorized API key')
    } finally {
      setSubmitting(false)
    }
  }

  useEffect(() => {
    function handleEnter(e: KeyboardEvent) {
      if (e.key === 'Enter') save()
    }
    window.addEventListener('keydown', handleEnter)
    return () => window.removeEventListener('keydown', handleEnter)
  }, [key])

  return (
    <Dialog open={open}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Enter Admin API Key</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">Paste the server's <code>SERVER_API_KEY</code> to access admin settings.</p>
        <div className="mt-3 grid gap-2">
          <Label htmlFor="admin-key">API Key</Label>
          <Input id="admin-key" autoFocus type="password" placeholder="SERVER_API_KEY" value={key} onChange={e => setKey(e.target.value)} />
        </div>
        {authError && <div className="mt-2 text-sm text-red-600">{authError}</div>}
        <div className="mt-4 flex justify-end gap-2">
          <Button onClick={save} disabled={submitting}>{submitting ? 'Checking…' : 'Continue'}</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function App() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiKey, setApiKey] = useState<string>('')
  const [initialSnapshot, setInitialSnapshot] = useState<string | null>(null)
  const [health, setHealth] = useState<'unknown' | 'ok' | 'error'>('unknown')
  const [testingEmbeddings, setTestingEmbeddings] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState('settings')
  
  // Vector Stores state
  const [vectorStores, setVectorStores] = useState<VectorStore[]>([])
  const [loadingStores, setLoadingStores] = useState(false)
  const [selectedStore, setSelectedStore] = useState<string | null>(null)
  const [newStoreName, setNewStoreName] = useState('')
  const [creatingStore, setCreatingStore] = useState(false)
  
  // Embeddings state
  const [embeddings, setEmbeddings] = useState<Embedding[]>([])
  const [loadingEmbeddings, setLoadingEmbeddings] = useState(false)
  const [selectedEmbedding, setSelectedEmbedding] = useState<string | null>(null)
  
  // Analytics state
  const [vectorStoreStats, setVectorStoreStats] = useState<VectorStoreStats | null>(null)
  const [globalStats, setGlobalStats] = useState<GlobalStats | null>(null)
  const [loadingStats, setLoadingStats] = useState(false)
  const [statsPeriod, setStatsPeriod] = useState<'daily' | 'weekly' | 'monthly' | 'all'>('daily')
  
  // Search Testing state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMode, setSearchMode] = useState<'hybrid' | 'vector_only' | 'keyword_only'>('hybrid')
  const [vectorWeight, setVectorWeight] = useState(0.7)
  const [keywordWeight, setKeywordWeight] = useState(0.3)
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [searching, setSearching] = useState(false)
  const [searchStoreId, setSearchStoreId] = useState('')

  const hasSettings = !!settings
  const validationErrors = useMemo(() => {
    if (!settings) return { port: false, dimensions: false }
    const portBad = !Number.isInteger(settings.server.port) || settings.server.port <= 0 || settings.server.port > 65535
    const dimsBad = !Number.isInteger(settings.embedding.dimensions) || settings.embedding.dimensions <= 0
    return { port: portBad, dimensions: dimsBad }
  }, [settings])

  function api<T>(path: string, init?: RequestInit): Promise<T> {
    return fetch(`${apiBase}/v1/admin${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        ...(init?.headers || {})
      }
    }).then(async (res) => {
      if (!res.ok) {
        if (res.status === 401) {
          setApiKey('')
        }
        const errorText = await res.text()
        throw new Error(errorText || `HTTP ${res.status}`)
      }
      return res.json()
    })
  }

  function apiPublic<T>(path: string, init?: RequestInit): Promise<T> {
    return fetch(`${apiBase}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        ...(init?.headers || {})
      }
    }).then(async (res) => {
      if (!res.ok) {
        if (res.status === 401) {
          setApiKey('')
        }
        const errorText = await res.text()
        throw new Error(errorText || `HTTP ${res.status}`)
      }
      return res.json()
    })
  }

  useEffect(() => {
    if (!apiKey) return
    api<Settings>('/settings')
      .then(s => {
        setSettings(s)
        setInitialSnapshot(JSON.stringify(s))
      })
      .catch(e => setError(String(e)))
  }, [apiKey])

  // Health check
  useEffect(() => {
    let cancelled = false
    fetch(`${apiBase}/health`).then(r => {
      if (cancelled) return
      setHealth(r.ok ? 'ok' : 'error')
    }).catch(() => {
      if (!cancelled) setHealth('error')
    })
    return () => { cancelled = true }
  }, [])

  // Load vector stores
  useEffect(() => {
    if (!apiKey || activeTab !== 'stores') return
    loadVectorStores()
  }, [apiKey, activeTab])

  // Load embeddings when store selected
  useEffect(() => {
    if (!apiKey || !selectedStore || activeTab !== 'embeddings') return
    loadEmbeddings(selectedStore)
  }, [apiKey, selectedStore, activeTab])

  // Load stats when tab active
  useEffect(() => {
    if (!apiKey || activeTab !== 'analytics') return
    loadGlobalStats()
    if (selectedStore) {
      loadVectorStoreStats(selectedStore)
    }
  }, [apiKey, activeTab, statsPeriod, selectedStore])

  async function loadVectorStores() {
    setLoadingStores(true)
    try {
      const response = await apiPublic<{ data: VectorStore[] }>('/v1/vector_stores?limit=100')
      setVectorStores(response.data || [])
    } catch (e: any) {
      toast.error('Failed to load vector stores')
    } finally {
      setLoadingStores(false)
    }
  }

  async function createVectorStore() {
    if (!newStoreName.trim()) {
      toast.error('Store name is required')
      return
    }
    setCreatingStore(true)
    try {
      const store = await apiPublic<VectorStore>('/v1/vector_stores', {
        method: 'POST',
        body: JSON.stringify({ name: newStoreName.trim() })
      })
      toast.success('Vector store created')
      setNewStoreName('')
      await loadVectorStores()
      setSelectedStore(store.id)
    } catch (e: any) {
      toast.error(`Failed to create store: ${e.message}`)
    } finally {
      setCreatingStore(false)
    }
  }

  async function deleteVectorStore(storeId: string) {
    if (!confirm('Delete this vector store and all its embeddings?')) return
    try {
      await apiPublic(`/v1/vector_stores/${storeId}?cascade=true`, { method: 'DELETE' })
      toast.success('Vector store deleted')
      if (selectedStore === storeId) {
        setSelectedStore(null)
        setEmbeddings([])
      }
      await loadVectorStores()
    } catch (e: any) {
      toast.error(`Failed to delete store: ${e.message}`)
    }
  }

  async function loadEmbeddings(storeId: string) {
    setLoadingEmbeddings(true)
    try {
      // Note: There's no list endpoint yet, so we'll use search with empty query as workaround
      // Or we can show a message that this requires search
      setEmbeddings([])
      toast.info('Use Search Testing tab to find embeddings')
    } catch (e: any) {
      toast.error('Failed to load embeddings')
    } finally {
      setLoadingEmbeddings(false)
    }
  }

  async function deleteEmbedding(storeId: string, embeddingId: string) {
    if (!confirm('Delete this embedding?')) return
    try {
      await apiPublic(`/v1/vector_stores/${storeId}/embeddings/${embeddingId}`, { method: 'DELETE' })
      toast.success('Embedding deleted')
      if (selectedStore === storeId) {
        await loadEmbeddings(storeId)
      }
      await loadVectorStores() // Refresh store stats
    } catch (e: any) {
      toast.error(`Failed to delete embedding: ${e.message}`)
    }
  }

  async function loadGlobalStats() {
    setLoadingStats(true)
    try {
      const stats = await apiPublic<GlobalStats>(`/v1/stats?period=${statsPeriod}`)
      setGlobalStats(stats)
    } catch (e: any) {
      toast.error('Failed to load global stats')
    } finally {
      setLoadingStats(false)
    }
  }

  async function loadVectorStoreStats(storeId: string) {
    try {
      const stats = await apiPublic<VectorStoreStats>(`/v1/vector_stores/${storeId}/stats?period=${statsPeriod}`)
      setVectorStoreStats(stats)
    } catch (e: any) {
      // Silently fail - stats might not be available
    }
  }

  async function performSearch() {
    if (!searchStoreId || !searchQuery.trim()) {
      toast.error('Please select a store and enter a search query')
      return
    }
    setSearching(true)
    try {
      const response = await apiPublic<{ data: any[] }>(`/v1/vector_stores/${searchStoreId}/search`, {
        method: 'POST',
        body: JSON.stringify({
          query: searchQuery,
          limit: 20,
          search_mode: searchMode,
          vector_weight: vectorWeight,
          keyword_weight: keywordWeight
        })
      })
      setSearchResults(response.data || [])
      toast.success(`Found ${response.data?.length || 0} results`)
    } catch (e: any) {
      toast.error(`Search failed: ${e.message}`)
      setSearchResults([])
    } finally {
      setSearching(false)
    }
  }

  // Ctrl+S save shortcut
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const isMac = navigator.platform.toUpperCase().includes('MAC')
      if ((isMac ? e.metaKey : e.ctrlKey) && e.key.toLowerCase() === 's' && activeTab === 'settings') {
        e.preventDefault()
        save()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activeTab])

  // Warn on unload if dirty
  const dirty = useMemo(() => initialSnapshot !== null && settings && JSON.stringify(settings) !== initialSnapshot, [initialSnapshot, settings])
  useEffect(() => {
    function beforeUnload(e: BeforeUnloadEvent) {
      if (!dirty) return
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', beforeUnload)
    return () => window.removeEventListener('beforeunload', beforeUnload)
  }, [dirty])

  function setField(group: keyof Settings, key: string, value: any) {
    setSettings(prev => prev ? { ...prev, [group]: { ...prev[group], [key]: value } } as Settings : prev)
  }

  async function save() {
    if (!settings) return
    setSaving(true)
    setError(null)
    try {
      const payload = JSON.parse(JSON.stringify(settings)) as Settings
      if (payload.auth && (payload.auth.server_api_key === '***' || !payload.auth.server_api_key)) {
        delete (payload as any).auth.server_api_key
      }
      if (payload.embedding && (payload.embedding.api_key === '***' || !payload.embedding.api_key)) {
        delete (payload as any).embedding.api_key
      }
      await api('/settings', { method: 'PUT', body: JSON.stringify(payload) })
      const fresh = await api<Settings>('/settings')
      setSettings(fresh)
      setInitialSnapshot(JSON.stringify(fresh))
      toast.success('Settings saved')
    } catch (e: any) {
      setError(String(e))
      toast.error('Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="relative p-6 font-sans">
      <Toaster richColors position="top-right" />
      {apiKey ? null : <AuthGate onReady={(key) => setApiKey(key)} />}
      <header className="relative z-10 flex items-center justify-between rounded-lg border bg-white/80 px-4 py-3 shadow-sm backdrop-blur mb-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-md bg-primary" />
          <div>
            <h1 className="text-xl font-semibold leading-tight">Vector Store Admin</h1>
            <div className="mt-0.5 text-xs text-muted-foreground">FastAPI + PGVector</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={"inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs " + (health === 'ok' ? 'bg-green-100 text-green-700' : health === 'error' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700')}>
            <span className={"h-1.5 w-1.5 rounded-full " + (health === 'ok' ? 'bg-green-600' : health === 'error' ? 'bg-red-600' : 'bg-gray-500')} />
            {health === 'ok' ? 'Healthy' : health === 'error' ? 'Unreachable' : 'Checking...'}
          </span>
          <Button variant="outline" onClick={() => {
            setHealth('unknown')
            fetch(`${apiBase}/health`).then(r => setHealth(r.ok ? 'ok' : 'error')).catch(() => setHealth('error'))
          }}>Test connection</Button>
          <Button variant="outline" onClick={async () => {
            if (!apiKey) { toast.error('Please authenticate first'); return }
            setTestingEmbeddings(true)
            try {
              const res = await api<{ database: { ok: boolean; error?: string }, embedding: { ok: boolean; error?: string } }>(`/settings/test`, { method: 'POST' })
              if (res.embedding.ok) {
                toast.success('Embedding provider OK')
              } else {
                toast.error(`Embedding check failed${res.embedding.error ? `: ${res.embedding.error}` : ''}`)
              }
              if (!res.database.ok) {
                toast.error(`Database check failed${res.database.error ? `: ${res.database.error}` : ''}`)
              }
            } catch (e: any) {
              toast.error('Test failed: unauthorized or server error')
            } finally {
              setTestingEmbeddings(false)
            }
          }} disabled={testingEmbeddings}>{testingEmbeddings ? 'Testing…' : 'Test embeddings'}</Button>
          <Button variant="secondary" onClick={() => { setApiKey('') }}>Change API Key</Button>
        </div>
      </header>

      {apiKey && (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full mt-6">
          <div className="mb-6">
            <TabsList className="w-full flex gap-1">
              <TabsTrigger value="settings" className="flex-1">Settings</TabsTrigger>
              <TabsTrigger value="stores" className="flex-1">Vector Stores</TabsTrigger>
              <TabsTrigger value="embeddings" className="flex-1">Embeddings</TabsTrigger>
              <TabsTrigger value="analytics" className="flex-1">Analytics</TabsTrigger>
              <TabsTrigger value="search" className="flex-1">Search Testing</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="settings" className="space-y-6">
            {!settings ? (
              <div className="grid gap-6 md:grid-cols-2">
                {[1,2,3].map(i => (
                  <div key={i} className="rounded-lg border bg-white p-6 shadow-sm">
                    <div className="h-5 w-40 animate-pulse rounded bg-gray-200" />
                    <div className="mt-4 space-y-3">
                      {[1,2,3].map(j => (
                        <div key={j} className="space-y-2">
                          <div className="h-4 w-16 animate-pulse rounded bg-gray-200" />
                          <div className="h-9 w-full animate-pulse rounded-md bg-gray-100" />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
                {error && <div className="col-span-full text-sm text-red-600">{error}</div>}
              </div>
            ) : (
              <div className="grid gap-6 md:grid-cols-2">
                <Card>
                <CardHeader>
                  <CardTitle>Server</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-3">
                  <div className="grid gap-2">
                    <Label htmlFor="host">Host</Label>
                    <Input id="host" value={settings.server.host} onChange={e => setField('server', 'host', e.target.value)} />
                    <p className="text-xs text-muted-foreground">Hostname or IP where the API is served.</p>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="port">Port</Label>
                    <Input id="port" type="number" className={validationErrors.port ? 'border-red-300' : ''} value={settings.server.port} onChange={e => setField('server', 'port', Number(e.target.value))} />
                    {validationErrors.port && <p className="text-xs text-red-600">Enter a valid port between 1 and 65535.</p>}
                    <p className="text-xs text-muted-foreground">Public port exposed by the API.</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Embedding</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-3">
                  <div className="grid gap-2">
                    <Label htmlFor="model">Model</Label>
                    <Input id="model" value={settings.embedding.model} onChange={e => setField('embedding', 'model', e.target.value)} />
                    <p className="text-xs text-muted-foreground">Provider-specific embedding model name.</p>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="base">Base URL</Label>
                    <Input id="base" value={settings.embedding.base_url} onChange={e => setField('embedding', 'base_url', e.target.value)} />
                    <p className="text-xs text-muted-foreground">LiteLLM proxy or direct embedding API endpoint.</p>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="ekey">API Key (write-only)</Label>
                    <Input id="ekey" type="password" placeholder="sk-..." onChange={e => setField('embedding', 'api_key', e.target.value)} />
                    <p className="text-xs text-muted-foreground">Used for outbound embedding requests. Not returned by the server.</p>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="dims">Dimensions</Label>
                    <Input id="dims" type="number" className={validationErrors.dimensions ? 'border-red-300' : ''} value={settings.embedding.dimensions} onChange={e => setField('embedding', 'dimensions', Number(e.target.value))} />
                    {validationErrors.dimensions && <p className="text-xs text-red-600">Dimensions must be a positive integer.</p>}
                    <p className="text-xs text-muted-foreground">Must match pgvector column dimensions.</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>DB Fields</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-3 md:grid-cols-3">
                  {Object.entries(settings.db_fields).map(([k, v]) => (
                    <div className="grid gap-2" key={k}>
                      <Label htmlFor={`dbf-${k}`}>{k}</Label>
                      <Input id={`dbf-${k}`} value={v as any} onChange={e => setField('db_fields', k, e.target.value)} />
                      <p className="text-xs text-muted-foreground">Database column for <span className="font-mono">{k}</span>.</p>
                    </div>
                  ))}
                </CardContent>
                <CardFooter className="justify-end">
                  <Button onClick={save} disabled={saving || validationErrors.port || validationErrors.dimensions}>{saving ? 'Saving...' : 'Save changes'}</Button>
                </CardFooter>
              </Card>
            </div>
            )}
          </TabsContent>

          <TabsContent value="stores" className="space-y-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Vector Stores</CardTitle>
                <div className="flex gap-2">
                  <Input
                    placeholder="New store name"
                    value={newStoreName}
                    onChange={e => setNewStoreName(e.target.value)}
                    className="w-48"
                    onKeyDown={e => e.key === 'Enter' && createVectorStore()}
                  />
                  <Button onClick={createVectorStore} disabled={creatingStore || !newStoreName.trim()}>
                    {creatingStore ? 'Creating...' : 'Create Store'}
                  </Button>
                  <Button variant="outline" onClick={loadVectorStores} disabled={loadingStores}>
                    {loadingStores ? 'Loading...' : 'Refresh'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {loadingStores ? (
                  <div className="text-center py-8 text-muted-foreground">Loading stores...</div>
                ) : vectorStores.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">No vector stores found. Create one to get started.</div>
                ) : (
                  <div className="space-y-2">
                    {vectorStores.map(store => (
                      <div
                        key={store.id}
                        className={`flex items-center justify-between p-4 border rounded-lg cursor-pointer transition-colors ${
                          selectedStore === store.id ? 'border-primary bg-primary/5' : 'hover:bg-muted/50'
                        }`}
                        onClick={() => setSelectedStore(store.id === selectedStore ? null : store.id)}
                      >
                        <div className="flex-1">
                          <div className="font-medium">{store.name}</div>
                          <div className="text-sm text-muted-foreground">
                            ID: {store.id} • {(store.usage_bytes / 1024 / 1024).toFixed(2)} MB • {store.file_counts?.total || 0} embeddings
                          </div>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation()
                            deleteVectorStore(store.id)
                          }}
                        >
                          Delete
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="embeddings" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Embeddings</CardTitle>
                <p className="text-sm text-muted-foreground">Select a vector store to view embeddings</p>
              </CardHeader>
              <CardContent>
                {!selectedStore ? (
                  <div className="text-center py-8 text-muted-foreground">
                    Select a vector store from the Vector Stores tab to manage embeddings.
                  </div>
                ) : (
                  <div>
                    <div className="mb-4 p-3 bg-muted rounded-lg">
                      <div className="font-medium">Selected Store: {vectorStores.find(s => s.id === selectedStore)?.name || selectedStore}</div>
                      <Button variant="outline" size="sm" className="mt-2" onClick={() => setSelectedStore(null)}>Change Store</Button>
                    </div>
                    <div className="text-center py-8 text-muted-foreground">
                      Use the Search Testing tab to find and manage embeddings. List view coming soon.
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Analytics & Monitoring</CardTitle>
                <div className="flex gap-2">
                  <select
                    value={statsPeriod}
                    onChange={e => setStatsPeriod(e.target.value as typeof statsPeriod)}
                    className="rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                    <option value="all">All Time</option>
                  </select>
                  <Button variant="outline" onClick={loadGlobalStats} disabled={loadingStats}>
                    {loadingStats ? 'Loading...' : 'Refresh'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {loadingStats ? (
                  <div className="text-center py-8 text-muted-foreground">Loading analytics...</div>
                ) : (
                  <>
                    <div>
                      <h3 className="text-lg font-semibold mb-4">Global Statistics</h3>
                      {globalStats ? (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="p-4 border rounded-lg">
                            <div className="text-sm text-muted-foreground">Total Requests</div>
                            <div className="text-2xl font-bold">{globalStats.total_requests}</div>
                          </div>
                          <div className="p-4 border rounded-lg">
                            <div className="text-sm text-muted-foreground">Vector Stores</div>
                            <div className="text-2xl font-bold">{globalStats.total_vector_stores}</div>
                          </div>
                          <div className="p-4 border rounded-lg">
                            <div className="text-sm text-muted-foreground">Embeddings</div>
                            <div className="text-2xl font-bold">{globalStats.total_embeddings}</div>
                          </div>
                          <div className="p-4 border rounded-lg">
                            <div className="text-sm text-muted-foreground">Storage</div>
                            <div className="text-2xl font-bold">{(globalStats.total_storage_bytes / 1024 / 1024).toFixed(2)} MB</div>
                          </div>
                          <div className="p-4 border rounded-lg">
                            <div className="text-sm text-muted-foreground">Avg Response Time</div>
                            <div className="text-2xl font-bold">{globalStats.avg_response_time_ms.toFixed(1)} ms</div>
                          </div>
                          <div className="p-4 border rounded-lg">
                            <div className="text-sm text-muted-foreground">Error Rate</div>
                            <div className="text-2xl font-bold">{(globalStats.error_rate * 100).toFixed(2)}%</div>
                          </div>
                          <div className="p-4 border rounded-lg">
                            <div className="text-sm text-muted-foreground">Error Count</div>
                            <div className="text-2xl font-bold">{globalStats.error_count}</div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-8 text-muted-foreground">No statistics available</div>
                      )}
                    </div>

                    {selectedStore && (
                      <div>
                        <h3 className="text-lg font-semibold mb-4">Vector Store Statistics</h3>
                        {vectorStoreStats ? (
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="p-4 border rounded-lg">
                              <div className="text-sm text-muted-foreground">Total Requests</div>
                              <div className="text-2xl font-bold">{vectorStoreStats.total_requests}</div>
                            </div>
                            <div className="p-4 border rounded-lg">
                              <div className="text-sm text-muted-foreground">Search Queries</div>
                              <div className="text-2xl font-bold">{vectorStoreStats.search_queries}</div>
                            </div>
                            <div className="p-4 border rounded-lg">
                              <div className="text-sm text-muted-foreground">Embeddings Created</div>
                              <div className="text-2xl font-bold">{vectorStoreStats.embeddings_created}</div>
                            </div>
                            <div className="p-4 border rounded-lg">
                              <div className="text-sm text-muted-foreground">Embeddings Deleted</div>
                              <div className="text-2xl font-bold">{vectorStoreStats.embeddings_deleted}</div>
                            </div>
                            <div className="p-4 border rounded-lg">
                              <div className="text-sm text-muted-foreground">Storage</div>
                              <div className="text-2xl font-bold">{(vectorStoreStats.storage_bytes / 1024 / 1024).toFixed(2)} MB</div>
                            </div>
                            <div className="p-4 border rounded-lg">
                              <div className="text-sm text-muted-foreground">Avg Response Time</div>
                              <div className="text-2xl font-bold">{vectorStoreStats.avg_response_time_ms.toFixed(1)} ms</div>
                            </div>
                            <div className="p-4 border rounded-lg">
                              <div className="text-sm text-muted-foreground">Error Rate</div>
                              <div className="text-2xl font-bold">{(vectorStoreStats.error_rate * 100).toFixed(2)}%</div>
                            </div>
                          </div>
                        ) : (
                          <div className="text-center py-4 text-muted-foreground text-sm">Select a vector store to view its statistics</div>
                        )}
                      </div>
                    )}

                    {globalStats?.endpoint_stats && (
                      <div>
                        <h3 className="text-lg font-semibold mb-4">Endpoint Breakdown</h3>
                        <div className="space-y-2">
                          {Object.entries(globalStats.endpoint_stats).map(([endpoint, stats]) => (
                            <div key={endpoint} className="p-4 border rounded-lg">
                              <div className="font-medium mb-2">{endpoint}</div>
                              <div className="grid grid-cols-3 gap-4 text-sm">
                                <div>
                                  <span className="text-muted-foreground">Requests: </span>
                                  <span className="font-medium">{stats.count}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Avg Time: </span>
                                  <span className="font-medium">{stats.avg_response_time_ms.toFixed(1)} ms</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Errors: </span>
                                  <span className="font-medium">{stats.error_count}</span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="search" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Search Testing</CardTitle>
                <p className="text-sm text-muted-foreground">Test hybrid search with different modes and weights</p>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4">
                  <div className="grid gap-2">
                    <Label htmlFor="search-store">Vector Store</Label>
                    <select
                      id="search-store"
                      value={searchStoreId}
                      onChange={e => setSearchStoreId(e.target.value)}
                      className="rounded-md border border-input bg-background px-3 py-2 text-sm"
                    >
                      <option value="">Select a vector store</option>
                      {vectorStores.map(store => (
                        <option key={store.id} value={store.id}>{store.name}</option>
                      ))}
                    </select>
                  </div>

                  <div className="grid gap-2">
                    <Label htmlFor="search-query">Search Query</Label>
                    <Input
                      id="search-query"
                      placeholder="Enter your search query..."
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && performSearch()}
                    />
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="grid gap-2">
                      <Label htmlFor="search-mode">Search Mode</Label>
                      <select
                        id="search-mode"
                        value={searchMode}
                        onChange={e => setSearchMode(e.target.value as typeof searchMode)}
                        className="rounded-md border border-input bg-background px-3 py-2 text-sm"
                      >
                        <option value="hybrid">Hybrid (Vector + Keyword)</option>
                        <option value="vector_only">Vector Only</option>
                        <option value="keyword_only">Keyword Only</option>
                      </select>
                    </div>

                    {searchMode === 'hybrid' && (
                      <>
                        <div className="grid gap-2">
                          <Label htmlFor="vector-weight">Vector Weight (0.0 - 1.0)</Label>
                          <Input
                            id="vector-weight"
                            type="number"
                            min="0"
                            max="1"
                            step="0.1"
                            value={vectorWeight}
                            onChange={e => {
                              const val = parseFloat(e.target.value)
                              if (!isNaN(val) && val >= 0 && val <= 1) {
                                setVectorWeight(val)
                                setKeywordWeight(1 - val)
                              }
                            }}
                          />
                        </div>
                        <div className="grid gap-2">
                          <Label htmlFor="keyword-weight">Keyword Weight (0.0 - 1.0)</Label>
                          <Input
                            id="keyword-weight"
                            type="number"
                            min="0"
                            max="1"
                            step="0.1"
                            value={keywordWeight}
                            onChange={e => {
                              const val = parseFloat(e.target.value)
                              if (!isNaN(val) && val >= 0 && val <= 1) {
                                setKeywordWeight(val)
                                setVectorWeight(1 - val)
                              }
                            }}
                          />
                        </div>
                      </>
                    )}
                  </div>

                  <Button onClick={performSearch} disabled={searching || !searchStoreId || !searchQuery.trim()}>
                    {searching ? 'Searching...' : 'Search'}
                  </Button>
                </div>

                {searchResults.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold mb-4">Search Results ({searchResults.length})</h3>
                    <div className="space-y-3">
                      {searchResults.map((result, idx) => (
                        <div key={idx} className="p-4 border rounded-lg">
                          <div className="flex items-start justify-between mb-2">
                            <div>
                              <div className="font-medium">{result.filename || `Result ${idx + 1}`}</div>
                              <div className="text-sm text-muted-foreground">
                                Score: {result.score?.toFixed(4) || 'N/A'} • ID: {result.file_id}
                              </div>
                            </div>
                            {selectedStore && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  if (confirm('Delete this embedding?')) {
                                    deleteEmbedding(selectedStore, result.file_id)
                                    setSearchResults(searchResults.filter((_, i) => i !== idx))
                                  }
                                }}
                              >
                                Delete
                              </Button>
                            )}
                          </div>
                          <div className="text-sm mt-2">
                            <div className="font-medium mb-1">Content:</div>
                            <div className="text-muted-foreground line-clamp-3">
                              {result.content?.[0]?.text || 'No content'}
                            </div>
                          </div>
                          {result.attributes && Object.keys(result.attributes).length > 0 && (
                            <div className="text-sm mt-2">
                              <div className="font-medium mb-1">Metadata:</div>
                              <div className="text-muted-foreground">
                                {JSON.stringify(result.attributes, null, 2)}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      {/* Sticky save bar */}
      {dirty && activeTab === 'settings' && (
        <div className="fixed inset-x-0 bottom-4 z-20 mx-auto w-full max-w-6xl">
          <div className="mx-6 rounded-lg border bg-white/95 px-4 py-3 shadow-lg backdrop-blur">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm">You have unsaved changes.</div>
              <div className="flex items-center gap-2">
                <Button variant="secondary" onClick={() => settings && setSettings(JSON.parse(initialSnapshot!))} disabled={saving}>Discard</Button>
                <Button onClick={save} disabled={saving || validationErrors.port || validationErrors.dimensions}>{saving ? 'Saving...' : 'Save'}</Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
