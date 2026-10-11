import React, { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { submitFeedback } from '../api/feedback'
import { explainPrediction } from '../api/prediction'

export default function ResultPage() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const [feedbackSent, setFeedbackSent] = useState(false)
  const [feedbackLoading, setFeedbackLoading] = useState(false)

  const [activeMethod, setActiveMethod] = useState('shap')
  const [loadedExplanations, setLoadedExplanations] = useState({})
  const [explainingLoading, setExplainingLoading] = useState(false)
  const [explainingError, setExplainingError] = useState('')

  if (!state?.result) {
    navigate('/')
    return null
  }

  const r = state.result
  const isFake = r.prediction === 'FAKE'
  const isEnsemble = r.model === 'ensemble'
  const confidencePct = Math.round((r.confidence || 0) * 100)
  const modelName = r.model || 'logistic'

  // Initialize with any pre-loaded SHAP data
  useEffect(() => {
    if (r.xai?.shap && !loadedExplanations[`${modelName}-shap`]) {
      setLoadedExplanations(prev => ({
        ...prev,
        [`${modelName}-shap`]: {
          explanations: r.xai.shap.top_features || [],
          fake_features: r.xai.shap.fake_features || [],
          real_features: r.xai.shap.real_features || [],
        }
      }))
    }
  }, [r, modelName])

  const handleFetchExplanation = async (method) => {
    setActiveMethod(method)
    const key = `${modelName}-${method}`
    if (loadedExplanations[key] || isEnsemble) return

    setExplainingLoading(true)
    setExplainingError('')
    try {
      const res = await explainPrediction({
        text: r.text,
        model: modelName,
        method: method,
        top_k: 10,
      })
      setLoadedExplanations(prev => ({
        ...prev,
        [key]: {
          explanations: res.data.explanations || [],
          fake_features: res.data.fake_features || [],
          real_features: res.data.real_features || [],
        }
      }))
    } catch (err) {
      setExplainingError(err.response?.data?.detail || `Failed to generate ${method.toUpperCase()} explanation.`)
    } finally {
      setExplainingLoading(false)
    }
  }

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

  // Active explanation data
  const currentExpKey = `${modelName}-${activeMethod}`
  const currentExp = loadedExplanations[currentExpKey] || null
  const xai = r.xai || null

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
            {Object.entries(r.votes).map(([vModel, vote]) => (
              <div key={vModel} className={`rounded-xl p-3 text-center border ${
                vote.prediction === 'FAKE' ? 'bg-red-50 border-red-100' : 'bg-emerald-50 border-emerald-100'
              }`}>
                <p className="text-xs text-slate-400 capitalize mb-1">{vModel}</p>
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
          {/* Explanation text */}
          {(r.explanation || xai?.explanation) && (
            <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
              <h2 className="text-sm font-semibold text-slate-600 mb-2">Explanation</h2>
              <p className="text-slate-500 text-sm leading-relaxed">
                {xai?.explanation || r.explanation}
              </p>
            </div>
          )}

          {/* Model Explainability: Both SHAP and LIME */}
          <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <div>
                <h2 className="text-sm font-semibold text-slate-800">
                  Model Explainability ({modelName.toUpperCase()})
                </h2>
                <p className="text-xs text-slate-400">
                  Inspect word contributions using SHAP or LIME
                </p>
              </div>

              {/* Method Selector Tabs */}
              <div className="flex gap-1.5 bg-slate-100 p-1 rounded-lg shrink-0">
                <button
                  type="button"
                  onClick={() => handleFetchExplanation('shap')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                    activeMethod === 'shap'
                      ? 'bg-white text-indigo-700 shadow-sm'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  SHAP
                </button>
                <button
                  type="button"
                  onClick={() => handleFetchExplanation('lime')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                    activeMethod === 'lime'
                      ? 'bg-white text-indigo-700 shadow-sm'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  LIME
                </button>
              </div>
            </div>

            {explainingLoading && (
              <div className="py-8 flex flex-col items-center justify-center gap-2">
                <span className="w-5 h-5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                <p className="text-xs text-slate-400">Computing {activeMethod.toUpperCase()} explanation for {modelName}...</p>
              </div>
            )}

            {explainingError && (
              <div className="p-3 bg-red-50 text-red-600 rounded-lg text-xs mb-3">
                {explainingError}
              </div>
            )}

            {!explainingLoading && !currentExp && (
              <div className="text-center py-6">
                <p className="text-xs text-slate-400 mb-3">
                  Click below to generate {activeMethod.toUpperCase()} word attributions for this prediction.
                </p>
                <button
                  type="button"
                  onClick={() => handleFetchExplanation(activeMethod)}
                  className="px-4 py-2 bg-indigo-50 border border-indigo-200 text-indigo-700 hover:bg-indigo-100 rounded-lg text-xs font-medium transition"
                >
                  Compute {activeMethod.toUpperCase()} Explanation
                </button>
              </div>
            )}

            {!explainingLoading && currentExp && currentExp.explanations?.length > 0 && (
              <>
                <div className="flex flex-col gap-2">
                  {currentExp.explanations.slice(0, 10).map((feat, i) => {
                    const isFakeWord = feat.direction === 'FAKE'
                    const barW = Math.min(100, Math.round(feat.impact * 1000)) || Math.round(feat.impact * 100) || 10
                    return (
                      <div key={i} className="flex items-center gap-2">
                        <span className="text-xs text-slate-600 font-medium w-28 shrink-0 truncate">{feat.word}</span>
                        <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className={`h-2 rounded-full ${isFakeWord ? 'bg-red-400' : 'bg-emerald-400'}`}
                            style={{ width: `${Math.max(8, barW)}%` }}
                          />
                        </div>
                        <span className={`text-xs font-semibold w-12 text-right shrink-0 ${
                          isFakeWord ? 'text-red-500' : 'text-emerald-600'
                        }`}>
                          {feat.direction}
                        </span>
                      </div>
                    )
                  })}
                </div>

                {currentExp.fake_features?.length > 0 && (
                  <p className="text-xs text-slate-500 mt-4">
                    <span className="text-red-500 font-medium">Fake indicators:</span>{' '}
                    {currentExp.fake_features.slice(0, 5).join(', ')}
                  </p>
                )}
                {currentExp.real_features?.length > 0 && (
                  <p className="text-xs text-slate-500 mt-1">
                    <span className="text-emerald-600 font-medium">Real indicators:</span>{' '}
                    {currentExp.real_features.slice(0, 5).join(', ')}
                  </p>
                )}

                <div className="mt-4 pt-3 border-t border-slate-100">
                  <p className="text-[11px] text-slate-400">
                    ℹ️ Feature attributions reflect words that most heavily influenced this model's prediction estimate. They do not constitute proof that an event factually occurred.
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Keywords + Clickbait */}
          <div className="grid grid-cols-2 gap-4">
            {(r.keywords?.length > 0 || xai?.keywords?.length > 0) && (
              <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
                <h2 className="text-sm font-semibold text-slate-600 mb-3">Keywords</h2>
                <div className="flex flex-wrap gap-2">
                  {(xai?.keywords || r.keywords || []).map((kw, i) => (
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
                <span className="text-3xl font-bold text-amber-500">
                  {xai?.clickbait?.score ?? r.clickbait_score ?? 0}
                </span>
                <span className="text-slate-400 text-sm mb-1">/100</span>
              </div>
              <div className="mt-2 h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-2 bg-amber-400 rounded-full"
                  style={{ width: `${xai?.clickbait?.score ?? r.clickbait_score ?? 0}%` }}
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
