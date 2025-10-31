import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/admin/',
  server: {
    proxy: {
      '/v1': 'http://localhost:8003',
      '/health': 'http://localhost:8003'
    }
  },
  build: {
    outDir: 'dist'
  }
})
