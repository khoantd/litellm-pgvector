import React, { useEffect, useState } from 'react'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Label } from './components/ui/label'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './components/ui/card'
import { Toaster, toast } from 'sonner'

type Settings = {
  server: { host: string; port: number }
  auth: { server_api_key: string | null }
  embedding: { model: string; base_url: string; api_key: string | null; dimensions: number }
  db_fields: Record<string, string>
}

const apiBase = ''

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${apiBase}/v1/admin${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${localStorage.getItem('SERVER_API_KEY') || ''}`,
      ...(init?.headers || {})
    }
  })
  if (!res.ok) {
    if (res.status === 401) {
      // Clear bad key and force re-auth prompt
      localStorage.removeItem('SERVER_API_KEY')
    }
    throw new Error(await res.text())
  }
  return res.json()
}

function AuthGate({ onReady }: { onReady: () => void }) {
  const [key, setKey] = useState('')
  function save() {
    if (!key) return
    localStorage.setItem('SERVER_API_KEY', key)
    onReady()
  }
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
      display: 'flex', alignItems: 'center', justifyContent: 'center'
    }}>
      <div style={{ background: '#fff', padding: 24, borderRadius: 8, minWidth: 360 }}>
        <h2>Enter Admin API Key</h2>
        <p>Paste the server's SERVER_API_KEY to access admin settings.</p>
        <div className="mt-3 grid gap-2">
          <Label htmlFor="admin-key">API Key</Label>
          <Input id="admin-key" type="password" placeholder="SERVER_API_KEY" value={key} onChange={e => setKey(e.target.value)} />
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <Button onClick={save}>Continue</Button>
        </div>
      </div>
    </div>
  )
}

export function App() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiKeyPresent, setApiKeyPresent] = useState<boolean>(() => !!localStorage.getItem('SERVER_API_KEY'))

  useEffect(() => {
    if (!apiKeyPresent) return
    api<Settings>('/settings').then(setSettings).catch(e => setError(String(e)))
  }, [apiKeyPresent])

  function setField(group: keyof Settings, key: string, value: any) {
    setSettings(prev => prev ? { ...prev, [group]: { ...prev[group], [key]: value } } as Settings : prev)
  }

  async function save() {
    if (!settings) return
    setSaving(true)
    setError(null)
    try {
      await api('/settings', { method: 'PUT', body: JSON.stringify(settings) })
      const fresh = await api<Settings>('/settings')
      setSettings(fresh)
      toast.success('Settings saved')
    } catch (e: any) {
      setError(String(e))
      toast.error('Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="p-6 font-sans">
      <Toaster richColors position="top-right" />
      {apiKeyPresent ? null : <AuthGate onReady={() => setApiKeyPresent(true)} />}
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Vector Store Admin</h1>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => { localStorage.removeItem('SERVER_API_KEY'); setApiKeyPresent(false); }}>Change API Key</Button>
        </div>
      </header>

      {!settings ? (
        <div className="p-5">Loading... {error && <div>{error}</div>}</div>
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
              </div>
              <div className="grid gap-2">
                <Label htmlFor="port">Port</Label>
                <Input id="port" type="number" value={settings.server.port} onChange={e => setField('server', 'port', Number(e.target.value))} />
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
              </div>
              <div className="grid gap-2">
                <Label htmlFor="base">Base URL</Label>
                <Input id="base" value={settings.embedding.base_url} onChange={e => setField('embedding', 'base_url', e.target.value)} />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="ekey">API Key (write-only)</Label>
                <Input id="ekey" type="password" placeholder="sk-..." onChange={e => setField('embedding', 'api_key', e.target.value)} />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="dims">Dimensions</Label>
                <Input id="dims" type="number" value={settings.embedding.dimensions} onChange={e => setField('embedding', 'dimensions', Number(e.target.value))} />
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
                </div>
              ))}
            </CardContent>
            <CardFooter className="justify-end">
              <Button onClick={save} disabled={saving}>{saving ? 'Saving...' : 'Save changes'}</Button>
            </CardFooter>
          </Card>
        </div>
      )}
    </div>
  )
}
