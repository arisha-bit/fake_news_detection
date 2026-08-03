import React, { useState } from 'react'
import { extractClaims } from '../api/upload'

const MODELS = [
  { value: 'logistic', label: 'Logistic' },
  { value: 'lstm', label: 'LSTM' },
  { value: 'bert', label: 'DistilBERT' },
]

export default function ClaimsPage() {
  const [text, setText] = useState('')
  const [model, setModel] = useState('logistic')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setError('')
    setLoading(true)
    setResult(null)
    try {
      const res = await extractClaims({ text, model })
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Claim extraction failed.')
    } finally {
      setLoading(false)
    }
  }

  const verdictColor = (v) => v === 'FAKE' ? 'text-red-500' : 'text-emerald-600'
  const verdictBg = (v) => v === 'FAKE' ? 'bg-red-50 border-red-100' : 'bg-emerald-50 border-emerald-100'

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Claim Extraction</h1>
        <p className="text-slate-400 text-sm mt-1">Split an article into factual claims and verify each one independently.</p>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">{error}</div>
      )}

      <form onSubmit={handleSubmit} className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm flex flex-col gap-4 mb-6">
        <textarea
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste article text here..."
          required
          className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 text-sm resize-none focus:outline-none focus:border-indigo-400 text-slate-700 placeholder-slate-300"
        />
        <div className="flex items-center gap-3">
          <div className="flex gap-2">
            {MODELS.map((m) => (
              <button key={m.value} type="button" onClick={() => setModel(m.value)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                  model === m.value ? 'bg-indigo-50 border-indigo-300 text-indigo-700' : 'bg-white border-slate-200 text-slate-500'
                }`}>
                {m.label}
              </button>
            ))}
          </div>
          <button type="submit" disabled={loading || !text.trim()}
            className="ml-auto bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg px-5 py-2 text-sm font-semibold text-white transition">
            {loading ? 'Extracting...' : 'Extract Claims'}
          </button>
        </div>
      </form>

      {result && (
        <div className="flex flex-col gap-4">
          {/* Overall verdict */}
          <div className={`rounded-2xl p-5 border ${verdictBg(result.overall_verdict)}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-400 mb-0.5">Overall Verdict</p>
                <p className={`text-2xl font-bold ${verdictColor(result.overall_verdict)}`}>{result.overall_verdict}</p>
              </div>
              <div className="text-right">
                <p className="text-xs text-slate-400">Confidence</p>
                <p className="text-xl font-bold text-slate-700">{Math.round(result.overall_confidence * 100)}%</p>
              </div>
            </div>
            <div className="flex gap-4 mt-3 text-xs text-slate-500">
              <span>{result.total_claims} claims extracted</span>
              <span className="text-red-500">{result.fake_claims} FAKE</span>
              <span className="text-emerald-600">{result.real_claims} REAL</span>
            </div>
          </div>

          {/* Per-claim results */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-5 py-3 border-b border-slate-100">
              <p className="text-sm font-semibold text-slate-600">Individual Claims</p>
            </div>
            <div className="divide-y divide-slate-50">
              {result.claims.map((claim, i) => (
                <div key={i} className="flex items-start gap-3 px-5 py-4">
                  <span className={`mt-0.5 shrink-0 px-2 py-0.5 rounded text-xs font-bold border ${
                    claim.prediction === 'FAKE' ? 'bg-red-50 text-red-500 border-red-100' : 'bg-emerald-50 text-emerald-600 border-emerald-100'
                  }`}>
                    {claim.prediction}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-slate-700">{claim.text}</p>
                    <p className="text-xs text-slate-400 mt-0.5">{Math.round(claim.confidence * 100)}% confidence</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
