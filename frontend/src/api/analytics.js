import api from './client'

export const getOverview = () => api.get('/analytics/overview')
export const getModelUsage = () => api.get('/analytics/models')
export const getMetrics = () => api.get('/metrics')
export const getAdminStats = () => api.get('/admin/stats')
