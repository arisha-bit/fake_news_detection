import api from './client'

export const submitFeedback = (data) => api.post('/feedback', data)
