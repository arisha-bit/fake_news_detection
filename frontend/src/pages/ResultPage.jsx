import React, { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { submitFeedback } from '../api/feedback'

export default function ResultPage() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const [feedbackSent, setFeedbackSent] = useState(false)
  const [feedbackLoading, setFeedbackLoading] = useState(false)

  if (!state?.result) {
    navigate('/')
    return null
  }

  const r = state.result
  const isFake = r.prediction === 'FAKE'
  const isEnsemble = r.model === 'ensemble'
  const confidencePct = Math.round((r.confidence || 0) * 100)

  const handleFeedback = async (is_correct) => {
    if (!r.prediction_id || feedbackSent) return
    setFeedbackLoading(true)
    try {
      await submitFeedback({ prediction_id: r.prediction_id, is_correct })
      setFeedbackSent(true)
    } catch {
      // silently ignore
    } finally {
      setFeedbackLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-slate-800">Analysis Result</h1>
        <button onClick={() => navigate('/')} className="text-sm text-indigo-500 hover:underline">
          New Analysis
        </button>
      </div>

      {/* Verdict */}
      <div className={`rounded-2xl p-6 border text-center ${
        isFake ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'
      }`}>
        <div className={`text-4xl font-extrabold mb-1 tracking-tight ${isFake ? 'text-red-500' : 'text-emerald-600'}`}>
          {isFake ? 'FAKE NEWS' : 'REAL NEWS'}
        </div>
        <p className="text-slate-500 text-sm mt-1">
          Confidence: <span className="font-semibold text-slate-700">{confidencePct}%</span>
          {' · '}
          Model: <span className="font-semibold text-slate-700 capitalize">{r.model}</span>
        </p>
        <div className="mt-4 h-2 bg-slate-200 rounded-full overflow-hidden">
          <div
            className={`h-2 rounded-full transition-all ${isFake ? 'bg-red-400' : 'bg-emerald-400'}`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {/* Ensemble votes */}
      {isEnsemble && r.votes && (
        <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-600 mb-3">Model Votes</h2>
          <div className="grid grid-cols-3 gap-3">
            {Object.entries(r.votes).map(([modelName, vote]) => (
              <div key={modelName} className={`rounded-xl p-3 text-center border ${
                vote.prediction === 'FAKE' ? 'bg-red-50 border-red-100' : 'bg-emerald-50 border-emerald-100'
              }`}>
                <p className="text-xs text-slate-400 capitalize mb-1">{modelName}</p>
                <p className={`text-sm font-bold ${vote.prediction === 'FAKE' ? 'text-red-500' : 'text-emerald-600'}`}>
                  {vote.prediction}
                </p>
                <p className="text-xs text-slate-400">{Math.round(vote.confidence * 100)}%</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Single model details */}
      {!isEnsemble && (
        <>
          {r.explanation && (
            <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
              <h2 className="text-sm font-semibold text-slate-600 mb-2">Explanation</h2>
              <p className="text-slate-500 text-sm leading-relaxed">{r.explanation}</p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
            {r.keywords?.length > 0 && (
              <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
                <h2 className="text-sm font-semibold text-slate-600 mb-3">Keywords</h2>
                <div className="flex flex-wrap gap-2">
                  {r.keywords.map((kw, i) => (
                    <span key={i} className="bg-indigo-50 text-indigo-600 text-xs px-2 py-1 rounded-md border border-indigo-100">
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
              <h2 className="text-sm font-semibold text-slate-600 mb-2">Clickbait Score</h2>
              <div className="flex items-end gap-1">
                <span className="text-3xl font-bold text-amber-500">{r.clickbait_score}</span>
                <span className="text-slate-400 text-sm mb-1">/100</span>
              </div>
              <div className="mt-2 h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-2 bg-amber-400 rounded-full"
                  style={{ width: `${r.clickbait_score}%` }}
                />
              </div>
            </div>
          </div>
        </>
      )}

      {/* Text preview */}
      <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-600 mb-2">Analyzed Text</h2>
        <p className="text-slate-500 text-sm leading-relaxed line-clamp-4">{r.text}</p>
      </div>

      {/* Feedback */}
      {!isEnsemble && r.prediction_id && (
        <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-600 mb-3">Was this prediction correct?</h2>
          {feedbackSent ? (
            <p className="text-emerald-600 text-sm">Thanks for your feedback!</p>
          ) : (
            <div className="flex gap-3">
              <button
                onClick={() => handleFeedback(true)}
                disabled={feedbackLoading}
                className="flex-1 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 text-emerald-700 disabled:opacity-50 rounded-lg py-2 text-sm font-medium transition"
              >
                Correct
              </button>
              <button
                onClick={() => handleFeedback(false)}
                disabled={feedbackLoading}
                className="flex-1 bg-red-50 hover:bg-red-100 border border-red-200 text-red-600 disabled:opacity-50 rounded-lg py-2 text-sm font-medium transition"
              >
                Incorrect
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
