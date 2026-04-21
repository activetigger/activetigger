import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    __BUILD_DATE__: JSON.stringify(new Date().toLocaleDateString('en-GB')), // Format: DD-MM-YYYY
  },
  server: {
    allowedHosts: [
      'localhost',
      '127.0.0.1',
      'css.activetigger.com',
      'demo1.activetigger.com',
      'demo2.activetigger.com',
      'demo3.activetigger.com',
    ],
  },
});
