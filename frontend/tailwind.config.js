/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Monopoly property colors
      colors: {
        'property-brown': '#8B4513',
        'property-lightblue': '#87CEEB',
        'property-magenta': '#FF00FF',
        'property-orange': '#FFA500',
        'property-red': '#FF0000',
        'property-yellow': '#FFFF00',
        'property-green': '#008000',
        'property-blue': '#0000FF',
        'property-railroad': '#000000',
        'property-utility': '#808080',

        // UI colors
        'board-bg': '#C8E6C9',
        'board-border': '#2E7D32',
        'space-bg': '#FAFAFA',

        // Player colors
        'player-0': '#E53935', // Red
        'player-1': '#1E88E5', // Blue
        'player-2': '#43A047', // Green
        'player-3': '#FDD835', // Yellow
        'player-4': '#8E24AA', // Purple
        'player-5': '#FB8C00', // Orange
        'player-6': '#00ACC1', // Cyan
        'player-7': '#6D4C41', // Brown
      },

      // Board grid sizing
      spacing: {
        'space-width': '70px',
        'space-height': '100px',
        'corner-size': '100px',
      },

      // Animations
      animation: {
        'dice-roll': 'diceRoll 0.5s ease-out',
        'token-move': 'tokenMove 0.3s ease-in-out',
        'pulse-highlight': 'pulseHighlight 1s infinite',
      },
      keyframes: {
        diceRoll: {
          '0%': { transform: 'rotate(0deg) scale(1)' },
          '50%': { transform: 'rotate(180deg) scale(1.2)' },
          '100%': { transform: 'rotate(360deg) scale(1)' },
        },
        tokenMove: {
          '0%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
          '100%': { transform: 'translateY(0)' },
        },
        pulseHighlight: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(66, 153, 225, 0.7)' },
          '50%': { boxShadow: '0 0 0 10px rgba(66, 153, 225, 0)' },
        },
      },
    },
  },
  plugins: [],
};
