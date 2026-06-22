import api from './client'

export const predict = (data) => api.post('/predict', data)
export const getHistory = () => api.get('/predict/history')
export const compareModels = (data) => api.post('/predict/compare', data)
export const ensemblePredict = (data) => api.post('/predict/ensemble', data)
