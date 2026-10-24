/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        edu: {
          950: '#060913',
          900: '#0B1021',
          850: '#111833',
          800: '#192347',
          700: '#253466',
          600: '#384E94',
          500: '#526FC2',
          400: '#7B96E0',
          300: '#ABC0F2',
          200: '#D6E2FC',
          100: '#EEF4FF',
        },
        grade: {
          a: '#22C55E',
          b: '#3B82F6',
          c: '#EAB308',
          d: '#F97316',
          f: '#EF4444',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
