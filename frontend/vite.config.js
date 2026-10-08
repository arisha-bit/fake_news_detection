import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      // Proxy /static/ to the backend so Grad-CAM heatmaps are served
      // from FastAPI's StaticFiles mount during development
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
