import React, { useEffect, useState } from 'react'
import { getHistory } from '../api/prediction'

export default function HistoryPage() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [filterModel, setFilterModel] = useState('all')
  const [filterLabel, setFilterLabel] = useState('all')

  useEffect(() => {
    getHistory().then((res) => setRecords(res.data)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const filtered = records.filter((r) => {
    const matchSearch = r.text.toLowerCase().includes(search.toLowerCase())
    const matchModel = filterModel === 'all' || r.model_name === filterModel
    const matchLabel = filterLabel === 'all' || r.prediction === filterLabel
    return matchSearch && matchModel && matchLabel
  })

  if (loading) return (
    <div className="flex justify-center mt-20">
      <div className="w-8 h-8 border-4 border-indigo-400 border-t-transparent rounded-full animate-spin" />
    </div>
  )

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-slate-800 mb-5">Prediction History</h1>

      <div className="flex flex-wrap gap-3 mb-5">
        <input
          type="text"
          placeholder="Search articles..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 min-w-48 bg-white border border-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-indigo-400 text-slate-700"
        />
        <select
          value={filterModel}
          onChange={(e) => setFilterModel(e.target.value)}
          className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-400 text-slate-600"
        >
          <option value="all">All Models</option>
          <option value="logistic">Logistic</option>
          <option value="lstm">LSTM</option>
          <option value="bert">BERT</option>
        </select>
        <select
          value={filterLabel}
          onChange={(e) => setFilterLabel(e.target.value)}
          className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-400 text-slate-600"
        >
          <option value="all">All Labels</option>
          <option value="FAKE">FAKE</option>
          <option value="REAL">REAL</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <div className="text-center text-slate-400 mt-16">No predictions found.</div>
      ) : (
        <div className="flex flex-col gap-3">
          {filtered.map((r) => (
            <div key={r.id} className="bg-white border border-slate-200 rounded-xl p-4 flex items-start gap-4 shadow-sm">
              <span className={`mt-0.5 px-2 py-0.5 rounded text-xs font-bold shrink-0 ${
                r.prediction === 'FAKE'
                  ? 'bg-red-50 text-red-500 border border-red-100'
                  : 'bg-emerald-50 text-emerald-600 border border-emerald-100'
              }`}>
                {r.prediction}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-slate-700 truncate">{r.text}</p>
                <div className="flex gap-3 mt-1 text-xs text-slate-400">
                  <span className="capitalize">{r.model_name}</span>
                  <span>{Math.round(r.confidence * 100)}% confidence</span>
                  <span>{new Date(r.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
