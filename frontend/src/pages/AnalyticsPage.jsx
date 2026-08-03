import React, { useEffect, useState } from 'react'
import {
  getOverview, getModelUsage, getMetrics, getAdminStats,
  getMonthlyUsage, getTopKeywords, getVerdictsOverTime
} from '../api/analytics'
import { useAuth } from '../context/AuthContext'

function StatCard({ label, value, color }) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className={`text-3xl font-bold ${color || 'text-slate-800'}`}>{value ?? '—'}</p>
    </div>
  )
}

function ModelBar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-500 mb-1">
        <span>{label}</span>
        <span>{value} predictions</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function MonthlyChart({ data }) {
  if (!data || data.length === 0) return <p className="text-sm text-slate-400">No monthly data yet.</p>
  const max = Math.max(...data.map(d => d.count), 1)
  return (
    <div className="flex items-end gap-1 h-24">
      {data.map((d) => (
        <div key={d.month} className="flex-1 flex flex-col items-center gap-1">
          <div
            className="w-full bg-indigo-400 rounded-t"
            style={{ height: `${Math.round((d.count / max) * 80)}px`, minHeight: d.count > 0 ? '4px' : '0' }}
          />
          <span className="text-xs text-slate-400 rotate-45 origin-left hidden sm:block"
            style={{ fontSize: '9px' }}>{d.month.slice(5)}</span>
        </div>
      ))}
    </div>
  )
}

function VerdictsChart({ data }) {
  if (!data || data.length === 0) return <p className="text-sm text-slate-400">No trend data yet.</p>
  const max = Math.max(...data.map(d => d.fake_count + d.real_count), 1)
  return (
    <div className="flex items-end gap-1 h-24">
      {data.map((d) => {
        const total = d.fake_count + d.real_count
        const h = Math.round((total / max) * 80)
        const fakeH = total > 0 ? Math.round((d.fake_count / total) * h) : 0
        return (
          <div key={d.month} className="flex-1 flex flex-col items-center gap-0.5">
            <div className="w-full flex flex-col" style={{ height: `${h}px`, minHeight: total > 0 ? '4px' : '0' }}>
              <div className="w-full bg-red-400 rounded-t" style={{ height: `${fakeH}px` }} />
              <div className="w-full bg-emerald-400 flex-1" />
            </div>
            <span className="text-slate-400 hidden sm:block" style={{ fontSize: '9px' }}>{d.month.slice(5)}</span>
          </div>
        )
      })}
    </div>
  )
}

function ConfidenceBar({ bucket, count, max }) {
  const pct = max > 0 ? Math.round((count / max) * 100) : 0
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 w-14 shrink-0">{bucket}%</span>
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-2 bg-indigo-400 rounded-full" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-400 w-8 text-right">{count}</span>
    </div>
  )
}

export default function AnalyticsPage() {
  const { user } = useAuth()
  const [overview, setOverview] = useState(null)
  const [modelUsage, setModelUsage] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [monthly, setMonthly] = useState([])
  const [keywords, setKeywords] = useState([])
  const [verdicts, setVerdicts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getOverview(), getModelUsage(), getMetrics(),
      getMonthlyUsage(), getTopKeywords(), getVerdictsOverTime()
    ])
      .then(([ov, mu, me, mo, kw, vt]) => {
        setOverview(ov.data)
        setModelUsage(mu.data)
        setMetrics(me.data)
        setMonthly(mo.data)
        setKeywords(kw.data)
        setVerdicts(vt.data)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex justify-center mt-20">
      <div className="w-8 h-8 border-4 border-indigo-400 border-t-transparent rounded-full animate-spin" />
    </div>
  )

  const maxUsage = modelUsage ? Math.max(...Object.values(modelUsage), 1) : 1
  const modelColors = { logistic: 'bg-indigo-400', lstm: 'bg-violet-400', bert: 'bg-pink-400' }

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-6">
      <h1 className="text-2xl font-bold text-slate-800">Analytics Dashboard</h1>

      {/* Overview */}
      <section>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Your Predictions</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatCard label="Total" value={overview?.total_predictions} />
          <StatCard label="Fake" value={overview?.fake_predictions} color="text-red-500" />
          <StatCard label="Real" value={overview?.real_predictions} color="text-emerald-600" />
          <StatCard label="Avg Confidence" value={overview ? `${Math.round((overview.average_confidence || 0) * 100)}%` : null} color="text-indigo-500" />
        </div>
      </section>

      {/* Fake vs Real ratio */}
      {overview?.total_predictions > 0 && (
        <section className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-sm font-semibold text-slate-600 mb-3">Fake vs Real Ratio</p>
          <div className="flex h-3 rounded-full overflow-hidden">
            <div className="bg-red-400 transition-all"
              style={{ width: `${(overview.fake_predictions / overview.total_predictions) * 100}%` }} />
            <div className="bg-emerald-400 flex-1" />
          </div>
          <div className="flex gap-4 mt-2 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-red-400 rounded-full inline-block" />
              Fake: {Math.round((overview.fake_predictions / overview.total_predictions) * 100)}%
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-emerald-400 rounded-full inline-block" />
              Real: {Math.round((overview.real_predictions / overview.total_predictions) * 100)}%
            </span>
          </div>
        </section>
      )}

      {/* Monthly + Verdicts charts side by side */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-sm font-semibold text-slate-600 mb-3">Monthly Usage</p>
          <MonthlyChart data={monthly} />
        </div>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-sm font-semibold text-slate-600 mb-2">Fake vs Real Over Time</p>
          <div className="flex gap-3 mb-2 text-xs text-slate-400">
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-red-400 rounded-sm inline-block" />Fake</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-emerald-400 rounded-sm inline-block" />Real</span>
          </div>
          <VerdictsChart data={verdicts} />
        </div>
      </div>

      {/* Model usage */}
      {modelUsage && Object.keys(modelUsage).length > 0 && (
        <section className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-sm font-semibold text-slate-600 mb-4">Model Usage</p>
          <div className="flex flex-col gap-3">
            {Object.entries(modelUsage).map(([model, count]) => (
              <ModelBar key={model} label={model.charAt(0).toUpperCase() + model.slice(1)}
                value={count} max={maxUsage} color={modelColors[model] || 'bg-indigo-400'} />
            ))}
          </div>
        </section>
      )}

      {/* Top keywords */}
      {keywords.length > 0 && (
        <section className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-sm font-semibold text-slate-600 mb-3">Top Keywords</p>
          <div className="flex flex-wrap gap-2">
            {keywords.slice(0, 20).map((kw) => (
              <span key={kw.keyword}
                className="bg-indigo-50 text-indigo-600 text-xs px-2.5 py-1 rounded-full border border-indigo-100 flex items-center gap-1">
                {kw.keyword}
                <span className="text-indigo-300 font-semibold">{kw.count}</span>
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Model accuracy */}
      {metrics && Object.keys(metrics).length > 0 && (
        <section>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Model Accuracy (from feedback)</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {Object.entries(metrics).map(([model, data]) => (
              <div key={model} className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <p className="text-xs text-slate-400 capitalize mb-1">{model}</p>
                <p className="text-2xl font-bold text-indigo-500">{data.accuracy}%</p>
                <p className="text-xs text-slate-400 mt-1">{data.correct_predictions}/{data.total_feedback} correct</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {user?.role === 'admin' && <AdminStats />}
    </div>
  )
}

function AdminStats() {
  const [stats, setStats] = useState(null)
  useEffect(() => { getAdminStats().then((res) => setStats(res.data)).catch(() => {}) }, [])
  if (!stats) return null
  return (
    <section>
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Admin — Global Stats</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Total Users', value: stats.users },
          { label: 'Total Predictions', value: stats.predictions },
          { label: 'Fake', value: stats.fake_predictions, color: 'text-red-500' },
          { label: 'Real', value: stats.real_predictions, color: 'text-emerald-600' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
            <p className="text-xs text-slate-400 mb-1">{label}</p>
            <p className={`text-3xl font-bold ${color || 'text-slate-800'}`}>{value}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
