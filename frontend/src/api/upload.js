import api from './client'

export const uploadImage = (formData, model = 'logistic') => {
  const params = new URLSearchParams({ model })
  return api.post(`/upload/image?${params}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
}

export const extractClaims = (data) => api.post('/claims/extract', data)

export const searchEvidence = (data) => api.post('/evidence/search', data)

export const reverseImageSearch = (formData, topK = 5) => {
  return api.post(`/images/reverse-search?top_k=${topK}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
}

export const checkCredibility = (data) => api.post('/credibility/check', data)

export const analysePropaganda = (data) => api.post('/propaganda/analyse', data)

export const generateReport = (data) =>
  api.post('/report/generate', data, { responseType: 'blob', timeout: 180000 })

export const extractKnowledgeGraph = (data) => api.post('/knowledge-graph/extract', data)
