import React, { useState, useRef, useEffect } from 'react'
import api from '../api/client'

const MODELS = [
  { value: 'logistic', label: 'Logistic' },
  { value: 'lstm', label: 'LSTM' },
  { value: 'bert', label: 'DistilBERT' },
]

const VERDICT_STYLES = {
  'LIKELY FAKE':       { bg: 'bg-red-50',     border: 'border-red-200',     text: 'text-red-600',     icon: '🚫' },
  'LIKELY REAL':       { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', icon: '✅' },
  'LIKELY MISLEADING': { bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-700',   icon: '⚠️' },
  'UNCERTAIN':         { bg: 'bg-slate-50',   border: 'border-slate-200',   text: 'text-slate-600',   icon: '❓' },
}

function ConfidenceBadge({ label, prediction, confidence }) {
  const isFake = prediction === 'FAKE'
  const isUnavailable = prediction === 'UNAVAILABLE' || prediction == null
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      {isUnavailable ? (
        <p className="text-sm text-slate-400">Not available</p>
      ) : (
        <>
          <p className={`text-lg font-bold ${isFake ? 'text-red-500' : 'text-emerald-600'}`}>
            {prediction}
          </p>
          <div className="mt-2 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className={`h-1.5 rounded-full ${isFake ? 'bg-red-400' : 'bg-emerald-400'}`}
              style={{ width: `${Math.round((confidence || 0) * 100)}%` }}
            />
          </div>
          <p className="text-xs text-slate-400 mt-1">{Math.round((confidence || 0) * 100)}% confidence</p>
        </>
      )}
    </div>
  )
}

export default function ImageVerifyPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState('logistic')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const inputRef = useRef()

  // Revoke old blob URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  const handleFile = (f) => {
    if (!f) return
    if (preview) URL.revokeObjectURL(preview)
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError('')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return
    setError('')
    setLoading(true)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await api.post(`/verify/image?model=${model}&top_k=5`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      })
      setResult(res.data)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(
        typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
          ? detail.map(d => d.msg || d).join('; ')
          : 'Verification failed. Please try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  const verdictStyle = result
    ? VERDICT_STYLES[result.overall_verdict] || VERDICT_STYLES['UNCERTAIN']
    : null

  const xai = result?.xai || null

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Image Verification</h1>
        <p className="text-slate-400 text-sm mt-1">
          Full multimodal analysis: OCR text detection + image classification + reverse image search + XAI.
        </p>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">{error}</div>
      )}

      <form onSubmit={handleSubmit} className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm flex flex-col gap-5 mb-6">
        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition ${
            drag ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />

          {preview ? (
            <div className="flex flex-col items-center gap-2">
              <img src={preview} alt="Preview" className="max-h-48 rounded-lg object-contain" />
              <p className="text-xs text-slate-400">{file?.name}</p>
            </div>
          ) : (
            <div>
              <p className="text-2xl mb-2">🖼️</p>
              <p className="text-slate-500 text-sm">Drop your image here or click to browse</p>
              <p className="text-slate-300 text-xs mt-1">JPG, JPEG, PNG — max 10 MB</p>
            </div>
          )}
        </div>

        {/* Model selector */}
        <div className="flex items-center gap-3">
          <p className="text-sm text-slate-500 shrink-0">Text model:</p>
          <div className="flex gap-2">
            {MODELS.map((m) => (
              <button
                key={m.value}
                type="button"
                onClick={() => setModel(m.value)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                  model === m.value
                    ? 'bg-indigo-50 border-indigo-300 text-indigo-700'
                    : 'bg-white border-slate-200 text-slate-500'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !file}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg py-3 font-semibold text-sm text-white transition"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Analyzing image...
            </span>
          ) : (
            'Verify Image'
          )}
        </button>
      </form>

      {/* Results */}
      {result && (
        <div className="flex flex-col gap-4">
          {/* Overall verdict */}
          <div className={`rounded-2xl p-6 border ${verdictStyle.bg} ${verdictStyle.border}`}>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-3xl">{verdictStyle.icon}</span>
              <div>
                <p className="text-xs text-slate-400">Overall Verdict</p>
                <p className={`text-2xl font-bold ${verdictStyle.text}`}>{result.overall_verdict}</p>
              </div>
            </div>
            {result.reasoning && result.reasoning.length > 0 && (
              <ul className="mt-3 space-y-1">
                {result.reasoning.map((r, i) => (
                  <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                    <span className="text-slate-400 mt-0.5">•</span>{r}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Branch results */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <ConfidenceBadge
              label="Text Analysis (OCR)"
              prediction={result.text_prediction}
              confidence={result.text_confidence}
            />
            <ConfidenceBadge
              label="Image Classification (ResNet18)"
              prediction={result.image_prediction}
              confidence={result.image_confidence}
            />
          </div>

          {/* Grad-CAM Heatmap */}
          {xai?.heatmap_url ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm font-semibold text-slate-600">Grad-CAM Heatmap</p>
                <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                  ResNet18 · layer4
                </span>
              </div>
              <p className="text-xs text-slate-400 mb-3">
                Highlighted regions show which areas of the image most influenced the{' '}
                <span className={xai.gradcam_prediction === 'FAKE' ? 'text-red-500 font-medium' : 'text-emerald-600 font-medium'}>
                  {xai.gradcam_prediction}
                </span>{' '}
                prediction ({xai.gradcam_confidence != null ? Math.round(xai.gradcam_confidence * 100) : '—'}% confidence).
                Red = high influence, blue = low influence.
              </p>
              <img
                src={xai.heatmap_url}
                alt="Grad-CAM heatmap"
                className="rounded-lg max-h-64 object-contain w-full border border-slate-100"
                onError={(e) => {
                  // Hide if the image fails to load
                  e.currentTarget.parentElement.style.display = 'none'
                }}
              />
            </div>
          ) : (
            xai !== null && (
              <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                <p className="text-xs text-slate-400">
                  <span className="font-medium text-slate-500">Grad-CAM:</span>{' '}
                  Heatmap generation unavailable — model weights may be missing or pytorch_grad_cam is not installed.
                </p>
              </div>
            )
          )}

          {/* LIME Explanation (lstm / bert only) */}
          {xai?.lime_explanations && xai.lime_explanations.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm font-semibold text-slate-600">
                  LIME Explanation ({model.toUpperCase()})
                </p>
                <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                  OCR text · top features
                </span>
              </div>
              <div className="flex flex-col gap-2">
                {xai.lime_explanations.slice(0, 10).map((feat, i) => {
                  const isFakeWord = feat.direction === 'FAKE'
                  const barW = Math.min(100, Math.round(Math.abs(feat.impact) * 1000))
                  return (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-xs text-slate-500 w-28 shrink-0 truncate">{feat.word}</span>
                      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-2 rounded-full ${isFakeWord ? 'bg-red-400' : 'bg-emerald-400'}`}
                          style={{ width: `${barW}%` }}
                        />
                      </div>
                      <span
                        className={`text-xs font-medium w-10 text-right shrink-0 ${
                          isFakeWord ? 'text-red-500' : 'text-emerald-600'
                        }`}
                      >
                        {feat.direction}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Note: logistic model uses SHAP (available on main predict page) */}
          {model === 'logistic' && (
            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
              <p className="text-xs text-slate-400">
                SHAP word-level explanation is available for Logistic Regression on the{' '}
                <a href="/" className="text-indigo-500 hover:underline">main analysis page</a>.
              </p>
            </div>
          )}

          {/* OCR text */}
          {result.ocr_text && (
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
              <p className="text-sm font-semibold text-slate-600 mb-2">Extracted Text (OCR)</p>
              <p className="text-sm text-slate-500 leading-relaxed line-clamp-4">{result.ocr_text}</p>
            </div>
          )}

          {/* Image class probabilities — now shows both FAKE and REAL */}
          {result.image_class_probabilities &&
            Object.keys(result.image_class_probabilities).length > 0 && (
              <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
                <p className="text-sm font-semibold text-slate-600 mb-3">
                  Image Classification Probabilities
                </p>
                {Object.entries(result.image_class_probabilities).map(([cls, prob]) => (
                  <div key={cls} className="mb-2">
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span className="font-medium">{cls}</span>
                      <span>{Math.round(prob * 100)}%</span>
                    </div>
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-2 rounded-full ${
                          cls === 'FAKE' ? 'bg-red-400' : 'bg-emerald-400'
                        }`}
                        style={{ width: `${Math.round(prob * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

          {/* CLIP similar articles */}
          {result.similar_articles && result.similar_articles.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm">
              <div className="px-5 py-3 border-b border-slate-100 flex items-center justify-between">
                <p className="text-sm font-semibold text-slate-600">Similar News Images Found</p>
                {result.clip_reuse_detected && (
                  <span className="text-xs font-medium text-amber-600 bg-amber-50 border border-amber-100 px-2 py-0.5 rounded-full">
                    ⚠️ Possible Reuse Detected
                  </span>
                )}
              </div>
              <div className="divide-y divide-slate-50">
                {result.similar_articles.map((art) => (
                  <div key={art.rank} className="px-5 py-4">
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-sm font-medium text-slate-700 flex-1">{art.title}</p>
                      <span
                        className={`shrink-0 text-xs font-bold px-2 py-0.5 rounded border ${
                          art.label === 'FAKE'
                            ? 'bg-red-50 text-red-500 border-red-100'
                            : 'bg-emerald-50 text-emerald-600 border-emerald-100'
                        }`}
                      >
                        {art.label}
                      </span>
                    </div>
                    {art.snippet && (
                      <p className="text-xs text-slate-400 mt-1">{art.snippet}</p>
                    )}
                    <div className="flex gap-3 mt-1 text-xs text-slate-400">
                      <span>{Math.round(art.similarity * 100)}% similarity</span>
                      {art.date && <span>{art.date}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
