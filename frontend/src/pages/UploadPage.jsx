import React, { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'

export default function UploadPage() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [drag, setDrag] = useState(false)
  const inputRef = useRef()

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
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
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
      formData.append('model', 'logistic')  // Default text model
      const res = await api.post('/upload/image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      })
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Image classification and analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const isFake = result?.image_prediction === 'FAKE'
  const confidencePct = result ? Math.round((result.image_confidence || 0) * 100) : 0

  return (
    <div className="max-w-2xl mx-auto flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Image Classification & Explainability</h1>
        <p className="text-slate-500 text-sm mt-1">
          Upload an image to classify authenticity with ResNet18 and generate a Grad-CAM explanation heatmap.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm">
          {error}
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm flex flex-col gap-5"
      >
        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition ${
            drag
              ? 'border-indigo-400 bg-indigo-50'
              : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'
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
              <img src={preview} alt="Upload Preview" className="max-h-56 rounded-lg object-contain" />
              <p className="text-xs text-slate-500 font-medium">{file?.name} ({(file.size / 1024).toFixed(1)} KB)</p>
            </div>
          ) : (
            <div>
              <p className="text-3xl mb-2">🖼️</p>
              <p className="text-slate-600 font-medium text-sm">Drop your image here or click to browse</p>
              <p className="text-slate-400 text-xs mt-1">Supported: JPG, JPEG, PNG — max 10 MB</p>
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={loading || !file}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-xl py-3 font-semibold text-sm text-white transition flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Running ResNet18 & Grad-CAM...
            </>
          ) : (
            'Classify Image & Explain'
          )}
        </button>
      </form>

      {/* Result Section */}
      {result && (
        <div className="flex flex-col gap-5">
          {/* Verdict Banner */}
          <div className={`rounded-2xl p-6 border text-center ${
            isFake ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'
          }`}>
            <span className="text-xs font-semibold tracking-wider uppercase text-slate-400 mb-1 block">
              ResNet18 Image Classification
            </span>
            <div className={`text-4xl font-extrabold mb-1 tracking-tight ${isFake ? 'text-red-500' : 'text-emerald-600'}`}>
              {isFake ? 'MANIPULATED / FAKE IMAGE' : 'AUTHENTIC / REAL IMAGE'}
            </div>
            <p className="text-slate-600 text-sm mt-1">
              Model Confidence: <span className="font-bold">{confidencePct}%</span>
            </p>
            <div className="mt-4 h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className={`h-2 rounded-full transition-all ${isFake ? 'bg-red-400' : 'bg-emerald-400'}`}
                style={{ width: `${confidencePct}%` }}
              />
            </div>
          </div>

          {/* Text Explainability (XAI) */}
          {result.xai && (result.xai.explanations || []).length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="text-sm font-semibold text-slate-800">{result.xai.method?.toUpperCase()} Text Explainability</h2>
                  <p className="text-xs text-slate-400">Model: {result.xai.model}</p>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-4">
                <div>
                  <h3 className="text-xs font-bold text-slate-700 mb-2">Words Pushing Toward REAL:</h3>
                  <div className="flex flex-wrap gap-2">
                    {(result.xai.real_features || []).slice(0, 5).map((word, i) => (
                      <span key={i} className="bg-emerald-50 text-emerald-700 px-2 py-1 rounded text-xs border border-emerald-100">
                        {word}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-700 mb-2">Words Pushing Toward FAKE:</h3>
                  <div className="flex flex-wrap gap-2">
                    {(result.xai.fake_features || []).slice(0, 5).map((word, i) => (
                      <span key={i} className="bg-red-50 text-red-700 px-2 py-1 rounded text-xs border border-red-100">
                        {word}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Keywords & Clickbait */}
          {result.keywords && result.keywords.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-slate-800 mb-3">Extracted Keywords</h2>
              <div className="flex flex-wrap gap-2">
                {result.keywords.map((kw, i) => (
                  <span key={i} className="bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded text-xs border border-indigo-100 font-medium">
                    {kw}
                  </span>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-slate-100">
                <p className="text-xs text-slate-600">
                  Clickbait Score: <span className="font-bold">{result.clickbait_score || 0}%</span>
                </p>
              </div>
            </div>
          )}

          {/* Grad-CAM Heatmap Visualization */}
          {result.heatmap_url ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h2 className="text-base font-bold text-slate-800">Grad-CAM Visual Explanation</h2>
                  <p className="text-xs text-slate-400">ResNet18 · Target Layer: layer4[-1]</p>
                </div>
                <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${
                  isFake ? 'bg-red-50 text-red-600 border border-red-100' : 'bg-emerald-50 text-emerald-700 border border-emerald-100'
                }`}>
                  Explaining: {result.prediction}
                </span>
              </div>
              <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                The heatmap highlights regions in the image that strongly drove the classifier toward the{' '}
                <span className="font-semibold">{result.prediction}</span> prediction. Warm colors (red/yellow) indicate maximum activation influence; cool colors (blue) indicate minimal influence.
              </p>
              <div className="flex justify-center bg-slate-50 rounded-xl p-2 border border-slate-100">
                <img
                  src={result.heatmap_url}
                  alt="Grad-CAM heatmap"
                  className="rounded-lg max-h-80 object-contain w-full"
                />
              </div>
            </div>
          ) : (
            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 text-xs text-slate-500">
              Grad-CAM heatmap generation was unavailable for this image.
            </div>
          )}

          {/* OCR Extracted Text (if >20 meaningful characters extracted) */}
          {result.has_meaningful_text && result.ocr_text && (
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-sm font-semibold text-slate-800">Extracted Text (OCR Active)</h2>
                {result.text_prediction && (
                  <span className={`text-xs font-bold px-2 py-0.5 rounded border ${
                    result.text_prediction === 'FAKE'
                      ? 'bg-red-50 text-red-600 border-red-100'
                      : 'bg-emerald-50 text-emerald-700 border-emerald-100'
                  }`}>
                    Text: {result.text_prediction} ({Math.round((result.text_confidence || 0) * 100)}%)
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-600 bg-slate-50 rounded-lg p-3 border border-slate-100 leading-relaxed">
                {result.ocr_text}
              </p>
              <div className="mt-3 flex justify-end">
                <button
                  type="button"
                  onClick={() => navigate('/', { state: { initialText: result.ocr_text } })}
                  className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 hover:underline"
                >
                  Analyze Extracted Text in Full NLP Pipeline →
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
