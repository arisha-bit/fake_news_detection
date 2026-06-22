import React, { useEffect, useState } from 'react'
import { getOverview, getModelUsage, getMetrics, getAdminStats } from '../api/analytics'
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

export default function AnalyticsPage() {
  const { user } = useAuth()
  const [overview, setOverview] = useState(null)
  const [modelUsage, setModelUsage] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getOverview(), getModelUsage(), getMetrics()])
      .then(([ov, mu, me]) => {
        setOverview(ov.data)
        setModelUsage(mu.data)
        setMetrics(me.data)
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
    <div className="max-w-3xl mx-auto flex flex-col gap-6">
      <h1 className="text-2xl font-bold text-slate-800">Analytics Dashboard</h1>

      <section>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Your Predictions</p>
        <div className="grid grid-cols-3 gap-4">
          <StatCard label="Total Predictions" value={overview?.total_predictions} />
          <StatCard label="Fake News" value={overview?.fake_predictions} color="text-red-500" />
          <StatCard label="Real News" value={overview?.real_predictions} color="text-emerald-600" />
        </div>
      </section>

      {overview && overview.total_predictions > 0 && (
        <section className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-sm font-semibold text-slate-600 mb-3">Fake vs Real Ratio</p>
          <div className="flex h-3 rounded-full overflow-hidden">
            <div
              className="bg-red-400 transition-all"
              style={{ width: `${(overview.fake_predictions / overview.total_predictions) * 100}%` }}
            />
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

      {modelUsage && Object.keys(modelUsage).length > 0 && (
        <section className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-sm font-semibold text-slate-600 mb-4">Model Usage</p>
          <div className="flex flex-col gap-3">
            {Object.entries(modelUsage).map(([model, count]) => (
              <ModelBar
                key={model}
                label={model.charAt(0).toUpperCase() + model.slice(1)}
                value={count}
                max={maxUsage}
                color={modelColors[model] || 'bg-indigo-400'}
              />
            ))}
          </div>
        </section>
      )}

      {metrics && Object.keys(metrics).length > 0 && (
        <section>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Model Accuracy (from feedback)</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {Object.entries(metrics).map(([model, data]) => (
              <div key={model} className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <p className="text-xs text-slate-400 capitalize mb-1">{model}</p>
                <p className="text-2xl font-bold text-indigo-500">{data.accuracy}%</p>
                <p className="text-xs text-slate-400 mt-1">
                  {data.correct_predictions}/{data.total_feedback} correct
                </p>
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

  useEffect(() => {
    getAdminStats().then((res) => setStats(res.data)).catch(() => {})
  }, [])

  if (!stats) return null

  return (
    <section>
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Admin — Global Stats</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-xs text-slate-400 mb-1">Total Users</p>
          <p className="text-3xl font-bold text-slate-800">{stats.users}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-xs text-slate-400 mb-1">Total Predictions</p>
          <p className="text-3xl font-bold text-slate-800">{stats.predictions}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-xs text-slate-400 mb-1">Fake</p>
          <p className="text-3xl font-bold text-red-500">{stats.fake_predictions}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <p className="text-xs text-slate-400 mb-1">Real</p>
          <p className="text-3xl font-bold text-emerald-600">{stats.real_predictions}</p>
        </div>
      </div>
    </section>
  )
}
