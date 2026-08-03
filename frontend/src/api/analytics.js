import api from './client'

export const getOverview = () => api.get('/analytics/overview')
export const getModelUsage = () => api.get('/analytics/models')
export const getMetrics = () => api.get('/metrics')
export const getAdminStats = () => api.get('/admin/stats')
export const getAnalyticsSummary = () => api.get('/analytics/summary')
export const getMonthlyUsage = () => api.get('/analytics/monthly')
export const getTopKeywords = () => api.get('/analytics/keywords')
export const getVerdictsOverTime = () => api.get('/analytics/verdicts-over-time')
