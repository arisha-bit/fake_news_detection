import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import Layout from './components/Layout'

import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import DashboardPage from './pages/DashboardPage'
import ResultPage from './pages/ResultPage'
import HistoryPage from './pages/HistoryPage'
import ComparePage from './pages/ComparePage'
import AnalyticsPage from './pages/AnalyticsPage'
import UploadPage from './pages/UploadPage'
import ClaimsPage from './pages/ClaimsPage'
import ToolsPage from './pages/ToolsPage'

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<Layout />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/result" element={<ResultPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/claims" element={<ClaimsPage />} />
            <Route path="/tools" element={<ToolsPage />} />
          </Route>
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  )
}
