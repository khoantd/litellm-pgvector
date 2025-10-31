import React from 'react'
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts'

type EndpointStats = {
  count: number
  avg_response_time_ms: number
  error_count: number
}

type TopVectorStore = {
  vector_store_id: string
  request_count: number
  avg_response_time_ms: number
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#ec4899']

interface EndpointBreakdownChartProps {
  endpointStats: Record<string, EndpointStats>
  title?: string
}

export function EndpointBreakdownChart({ endpointStats, title }: EndpointBreakdownChartProps) {
  const data = Object.entries(endpointStats).map(([endpoint, stats]) => ({
    endpoint: endpoint.length > 30 ? endpoint.substring(0, 30) + '...' : endpoint,
    fullEndpoint: endpoint,
    requests: stats.count,
    avgTime: Number(stats.avg_response_time_ms.toFixed(1)),
    errors: stats.error_count
  }))

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground border rounded-lg">
        No endpoint data available
      </div>
    )
  }

  return (
    <div className="w-full">
      {title && <h4 className="text-sm font-semibold mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="endpoint"
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
            fontSize={12}
          />
          <YAxis />
          <Tooltip
            formatter={(value: number, name: string) => {
              if (name === 'requests') return [value, 'Requests']
              if (name === 'avgTime') return [`${value} ms`, 'Avg Response Time']
              if (name === 'errors') return [value, 'Errors']
              return [value, name]
            }}
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px' }}
          />
          <Legend />
          <Bar dataKey="requests" fill="#3b82f6" name="Requests" />
          <Bar dataKey="errors" fill="#ef4444" name="Errors" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface EndpointPieChartProps {
  endpointStats: Record<string, EndpointStats>
  title?: string
}

export function EndpointPieChart({ endpointStats, title }: EndpointPieChartProps) {
  const data = Object.entries(endpointStats)
    .map(([endpoint, stats]) => ({
      name: endpoint.length > 20 ? endpoint.substring(0, 20) + '...' : endpoint,
      fullName: endpoint,
      value: stats.count
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8) // Top 8 endpoints

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground border rounded-lg">
        No endpoint data available
      </div>
    )
  }

  return (
    <div className="w-full">
      {title && <h4 className="text-sm font-semibold mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value: number) => [value, 'Requests']}
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px' }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

interface TopVectorStoresChartProps {
  topStores: TopVectorStore[]
  storeNames: Record<string, string>
  title?: string
}

export function TopVectorStoresChart({ topStores, storeNames, title }: TopVectorStoresChartProps) {
  const data = topStores
    .map(store => ({
      name: storeNames[store.vector_store_id] || store.vector_store_id.substring(0, 20),
      fullName: storeNames[store.vector_store_id] || store.vector_store_id,
      requests: store.request_count,
      avgTime: Number(store.avg_response_time_ms.toFixed(1))
    }))
    .slice(0, 10) // Top 10

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground border rounded-lg">
        No vector store data available
      </div>
    )
  }

  return (
    <div className="w-full">
      {title && <h4 className="text-sm font-semibold mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="name"
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
            fontSize={12}
          />
          <YAxis />
          <Tooltip
            formatter={(value: number, name: string) => {
              if (name === 'requests') return [value, 'Requests']
              if (name === 'avgTime') return [`${value} ms`, 'Avg Response Time']
              return [value, name]
            }}
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px' }}
          />
          <Legend />
          <Bar dataKey="requests" fill="#3b82f6" name="Requests" />
          <Bar dataKey="avgTime" fill="#10b981" name="Avg Response Time (ms)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface ResponseTimeChartProps {
  endpointStats: Record<string, EndpointStats>
  title?: string
}

export function ResponseTimeChart({ endpointStats, title }: ResponseTimeChartProps) {
  const data = Object.entries(endpointStats)
    .map(([endpoint, stats]) => ({
      endpoint: endpoint.length > 30 ? endpoint.substring(0, 30) + '...' : endpoint,
      fullEndpoint: endpoint,
      avgTime: Number(stats.avg_response_time_ms.toFixed(1))
    }))
    .sort((a, b) => b.avgTime - a.avgTime)
    .slice(0, 10) // Top 10 by response time

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground border rounded-lg">
        No endpoint data available
      </div>
    )
  }

  return (
    <div className="w-full">
      {title && <h4 className="text-sm font-semibold mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="endpoint"
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
            fontSize={12}
          />
          <YAxis />
          <Tooltip
            formatter={(value: number) => [`${value} ms`, 'Avg Response Time']}
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px' }}
          />
          <Bar dataKey="avgTime" fill="#f59e0b" name="Avg Response Time (ms)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface ErrorRateChartProps {
  endpointStats: Record<string, EndpointStats>
  title?: string
}

export function ErrorRateChart({ endpointStats, title }: ErrorRateChartProps) {
  const data = Object.entries(endpointStats)
    .map(([endpoint, stats]) => {
      const errorRate = stats.count > 0 ? (stats.error_count / stats.count) * 100 : 0
      return {
        endpoint: endpoint.length > 30 ? endpoint.substring(0, 30) + '...' : endpoint,
        fullEndpoint: endpoint,
        errorRate: Number(errorRate.toFixed(2)),
        errors: stats.error_count,
        requests: stats.count
      }
    })
    .filter(item => item.requests > 0)
    .sort((a, b) => b.errorRate - a.errorRate)
    .slice(0, 10) // Top 10 by error rate

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground border rounded-lg">
        No endpoint data available
      </div>
    )
  }

  return (
    <div className="w-full">
      {title && <h4 className="text-sm font-semibold mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="endpoint"
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
            fontSize={12}
          />
          <YAxis />
          <Tooltip
            formatter={(value: number, name: string) => {
              if (name === 'errorRate') return [`${value.toFixed(2)}%`, 'Error Rate']
              if (name === 'errors') return [value, 'Errors']
              if (name === 'requests') return [value, 'Total Requests']
              return [value, name]
            }}
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px' }}
          />
          <Bar dataKey="errorRate" fill="#ef4444" name="Error Rate (%)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface MetricsComparisonProps {
  globalStats?: {
    total_requests: number
    total_vector_stores: number
    total_embeddings: number
    total_storage_bytes: number
    avg_response_time_ms: number
    error_rate: number
  }
  vectorStoreStats?: {
    total_requests: number
    search_queries: number
    embeddings_created: number
    embeddings_deleted: number
    storage_bytes: number
    avg_response_time_ms: number
    error_rate: number
  }
  title?: string
}

export function MetricsComparison({ globalStats, vectorStoreStats, title }: MetricsComparisonProps) {
  if (!globalStats && !vectorStoreStats) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground border rounded-lg">
        No metrics data available
      </div>
    )
  }

  const data = []
  
  if (globalStats) {
    data.push(
      { metric: 'Total Requests', value: globalStats.total_requests },
      { metric: 'Vector Stores', value: globalStats.total_vector_stores },
      { metric: 'Embeddings', value: globalStats.total_embeddings },
      { metric: 'Storage (MB)', value: Number((globalStats.total_storage_bytes / 1024 / 1024).toFixed(2)) }
    )
  }

  if (vectorStoreStats) {
    data.push(
      { metric: 'Search Queries', value: vectorStoreStats.search_queries },
      { metric: 'Embeddings Created', value: vectorStoreStats.embeddings_created },
      { metric: 'Embeddings Deleted', value: vectorStoreStats.embeddings_deleted }
    )
  }

  return (
    <div className="w-full">
      {title && <h4 className="text-sm font-semibold mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="metric"
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
            fontSize={12}
          />
          <YAxis />
          <Tooltip
            formatter={(value: number) => [value, 'Value']}
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px' }}
          />
          <Bar dataKey="value" fill="#3b82f6" name="Value" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface PeriodComparisonData {
  period: string
  total_requests: number
  avg_response_time_ms: number
  error_count: number
  error_rate: number
}

interface PeriodComparisonChartProps {
  daily?: PeriodComparisonData
  weekly?: PeriodComparisonData
  monthly?: PeriodComparisonData
  allTime?: PeriodComparisonData
  title?: string
}

export function PeriodComparisonChart({ daily, weekly, monthly, allTime, title }: PeriodComparisonChartProps) {
  const periods = [
    { key: 'daily', label: 'Daily', data: daily },
    { key: 'weekly', label: 'Weekly', data: weekly },
    { key: 'monthly', label: 'Monthly', data: monthly },
    { key: 'all', label: 'All Time', data: allTime }
  ].filter(p => p.data)

  if (periods.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground border rounded-lg">
        No comparison data available
      </div>
    )
  }

  const requestsData = periods.map(p => ({
    period: p.label,
    value: p.data!.total_requests
  }))

  const responseTimeData = periods.map(p => ({
    period: p.label,
    value: Number(p.data!.avg_response_time_ms.toFixed(1))
  }))

  const errorRateData = periods.map(p => ({
    period: p.label,
    value: Number((p.data!.error_rate * 100).toFixed(2))
  }))

  return (
    <div className="w-full space-y-6">
      {title && <h4 className="text-sm font-semibold mb-4">{title}</h4>}
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <h5 className="text-sm font-medium mb-2 text-center">Requests</h5>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={requestsData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip
                formatter={(value: number) => [value, 'Requests']}
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px', fontSize: '12px' }}
              />
              <Bar dataKey="value" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <h5 className="text-sm font-medium mb-2 text-center">Response Time (ms)</h5>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={responseTimeData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip
                formatter={(value: number) => [`${value} ms`, 'Avg Response Time']}
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px', fontSize: '12px' }}
              />
              <Bar dataKey="value" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <h5 className="text-sm font-medium mb-2 text-center">Error Rate (%)</h5>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={errorRateData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip
                formatter={(value: number) => [`${value}%`, 'Error Rate']}
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '6px', fontSize: '12px' }}
              />
              <Bar dataKey="value" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

