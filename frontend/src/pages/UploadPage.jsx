import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadImage } from '../api/upload'

const MODELS = [
  { value: 'logistic', label: 'Logistic Regression' },
  { value: 'lstm', label: 'LSTM' },
  { value: 'bert', label: 'DistilBERT' },
]

export default function UploadPage() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [model, setModel] = useState('logistic')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)

  const handleFile = (f) => {
    if (!f) return
    setFile(f)
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
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await uploadImage(formData, model)
      navigate('/result', {
        state: {
          result: {
            ...res.data,
            model,
            text: res.data.extracted_text,
            source: 'Image OCR',
          },
        },
      })
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Upload Image</h1>
        <p className="text-slate-400 text-sm mt-1">
          Upload an image containing news text. OCR will extract the text and run fake-news detection.
        </p>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">
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
          onClick={() => document.getElementById('file-input').click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition ${
            drag
              ? 'border-indigo-400 bg-indigo-50'
              : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'
          }`}
        >
          <input
            id="file-input"
            type="file"
            accept=".jpg,.jpeg,.png"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {file ? (
            <div>
              <p className="text-indigo-600 font-medium text-sm">{file.name}</p>
              <p className="text-slate-400 text-xs mt-1">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
          ) : (
            <div>
              <p className="text-2xl mb-2">🖼️</p>
              <p className="text-slate-400 text-sm">Drop your image here or click to browse</p>
              <p className="text-slate-300 text-xs mt-1">JPG, JPEG, PNG — max 10 MB</p>
            </div>
          )}
        </div>

        {/* Model selector */}
        <div>
          <label className="block text-sm font-medium text-slate-600 mb-2">
            Text detection model
          </label>
          <div className="grid grid-cols-3 gap-2">
            {MODELS.map((m) => (
              <button
                key={m.value}
                type="button"
                onClick={() => setModel(m.value)}
                className={`px-3 py-2 rounded-lg text-xs font-medium border transition ${
                  model === m.value
                    ? 'bg-indigo-50 border-indigo-300 text-indigo-700'
                    : 'bg-white border-slate-200 text-slate-500 hover:border-indigo-200'
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
              Extracting & Analyzing...
            </span>
          ) : (
            'Analyze Image'
          )}
        </button>
      </form>
    </div>
  )
}
