import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { predict, ensemblePredict } from '../api/prediction'
import { useAuth } from '../context/AuthContext'

const MODELS = [
  { value: 'logistic', label: 'Logistic Regression' },
  { value: 'lstm', label: 'LSTM' },
  { value: 'bert', label: 'DistilBERT' },
  { value: 'ensemble', label: 'Ensemble' },
]

export default function DashboardPage() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [text, setText] = useState('')
  const [model, setModel] = useState('logistic')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handlePredict = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setError('')
    setLoading(true)
    try {
      let result
      if (model === 'ensemble') {
        const res = await ensemblePredict({ text })
        result = { ...res.data, model: 'ensemble', text }
      } else {
        const res = await predict({ text, model })
        result = { ...res.data, model, text }
      }
      navigate('/result', { state: { result } })
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Welcome back, {user?.username}</h1>
        <p className="text-slate-400 text-sm mt-1">Paste a news article or headline to check its authenticity.</p>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">
          {error}
        </div>
      )}

      <form onSubmit={handlePredict} className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm flex flex-col gap-5">
        <div>
          <label className="block text-sm font-medium text-slate-600 mb-2">News Article / Headline</label>
          <textarea
            rows={6}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your news article or headline here..."
            required
            className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 text-sm resize-none focus:outline-none focus:border-indigo-400 text-slate-700 placeholder-slate-300"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-600 mb-2">Select Model</label>
          <div className="grid grid-cols-2 gap-2">
            {MODELS.map((m) => (
              <button
                key={m.value}
                type="button"
                onClick={() => setModel(m.value)}
                className={`px-4 py-2.5 rounded-lg text-sm font-medium border transition ${
                  model === m.value
                    ? 'bg-indigo-50 border-indigo-300 text-indigo-700'
                    : 'bg-white border-slate-200 text-slate-500 hover:border-indigo-200 hover:text-slate-700'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !text.trim()}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg py-3 font-semibold text-sm text-white transition"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Analyzing...
            </span>
          ) : (
            'Analyze Article'
          )}
        </button>
      </form>
    </div>
  )
}
