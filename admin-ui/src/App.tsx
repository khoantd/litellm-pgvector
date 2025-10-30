import React, { useEffect, useMemo, useState } from 'react'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Label } from './components/ui/label'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './components/ui/card'
import { Toaster, toast } from 'sonner'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './components/ui/dialog'

type Settings = {
  server: { host: string; port: number }
  auth: { server_api_key: string | null }
  embedding: { model: string; base_url: string; api_key: string | null; dimensions: number }
  db_fields: Record<string, string>
}

const apiBase = ''

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
      // Validate immediately before closing using provided key
      const res = await fetch(`${apiBase}/v1/admin/settings`, {
        headers: { 'Authorization': `Bearer ${trimmed}` }
      })
      if (!res.ok) throw new Error('unauthorized')
      setOpen(false)
      onReady(trimmed)
    } catch (e: any) {
      // Keep dialog open and report error
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
        throw new Error(await res.text())
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

  // Ctrl+S save shortcut
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const isMac = navigator.platform.toUpperCase().includes('MAC')
      if ((isMac ? e.metaKey : e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault()
        save()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

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
      // Build payload that omits unchanged/redacted secrets
      const payload = JSON.parse(JSON.stringify(settings)) as Settings
      if (payload.auth && (payload.auth.server_api_key === '***' || !payload.auth.server_api_key)) {
        // Do not update auth key when placeholder or empty
        delete (payload as any).auth.server_api_key
      }
      if (payload.embedding && (payload.embedding.api_key === '***' || !payload.embedding.api_key)) {
        // Do not update embedding key when placeholder or empty
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
      <header className="relative z-10 flex items-center justify-between rounded-lg border bg-white/80 px-4 py-3 shadow-sm backdrop-blur">
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

      {!settings ? (
        <div className="mt-6 grid gap-6 md:grid-cols-2">
          {/* Loading skeletons */}
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
        <div className="mt-6 grid gap-6 md:grid-cols-2">
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

      {/* Sticky save bar */}
      {dirty && (
        <div className="fixed inset-x-0 bottom-4 z-20 mx-auto w-full max-w-4xl">
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
