/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#6366f1',
        danger: '#ef4444',
        success: '#22c55e',
      }
    }
  },
  plugins: []
}
