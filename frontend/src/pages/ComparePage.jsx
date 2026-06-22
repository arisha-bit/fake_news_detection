import React, { useState } from 'react'
import { compareModels } from '../api/prediction'

const MODEL_LABELS = { logistic: 'Logistic Regression', lstm: 'LSTM', bert: 'DistilBERT' }

export default function ComparePage() {
  const [text, setText] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleCompare = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setError('')
    setLoading(true)
    try {
      const res = await compareModels({ text })
      setResults(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-slate-800 mb-1">Model Comparison</h1>
      <p className="text-slate-400 text-sm mb-5">Run the same article through all three models side by side.</p>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">{error}</div>
      )}

      <form onSubmit={handleCompare} className="flex flex-col gap-4 mb-6">
        <textarea
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your news article or headline here..."
          required
          className="w-full bg-white border border-slate-200 rounded-lg px-4 py-3 text-sm resize-none focus:outline-none focus:border-indigo-400 text-slate-700 placeholder-slate-300"
        />
        <button
          type="submit"
          disabled={loading || !text.trim()}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg py-2.5 text-sm font-semibold text-white transition"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Comparing...
            </span>
          ) : (
            'Compare Models'
          )}
        </button>
      </form>

      {results && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {Object.entries(results).map(([model, data]) => {
            const isFake = data.prediction === 'FAKE'
            return (
              <div key={model} className={`rounded-2xl p-5 border shadow-sm ${
                isFake ? 'bg-red-50 border-red-100' : 'bg-emerald-50 border-emerald-100'
              }`}>
                <p className="text-xs text-slate-400 mb-1">{MODEL_LABELS[model]}</p>
                <p className={`text-xl font-bold ${isFake ? 'text-red-500' : 'text-emerald-600'}`}>
                  {isFake ? 'FAKE' : 'REAL'}
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  {Math.round(data.confidence * 100)}% confidence
                </p>
                <div className="mt-3 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className={`h-1.5 rounded-full ${isFake ? 'bg-red-400' : 'bg-emerald-400'}`}
                    style={{ width: `${Math.round(data.confidence * 100)}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
