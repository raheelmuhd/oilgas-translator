import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Oil & Gas inspired color palette
        petroleum: {
          50: "#f0f7ff",
          100: "#e0effe",
          200: "#bae0fd",
          300: "#7cc8fb",
          400: "#36aaf6",
          500: "#0c8ee7",
          600: "#0070c4",
          700: "#01599f",
          800: "#064c83",
          900: "#0a406d",
          950: "#072849",
        },
        crude: {
          50: "#f6f5f0",
          100: "#e9e6db",
          200: "#d5cfba",
          300: "#bdb393",
          400: "#a89973",
          500: "#998762",
          600: "#836f54",
          700: "#6a5846",
          800: "#5a4a3e",
          900: "#4e4138",
          950: "#2c231d",
        },
        flame: {
          50: "#fff8ed",
          100: "#ffefd4",
          200: "#ffdba8",
          300: "#ffc170",
          400: "#ff9c37",
          500: "#ff7f11",
          600: "#f06307",
          700: "#c74908",
          800: "#9e3a0f",
          900: "#7f3210",
          950: "#451606",
        },
        rig: {
          50: "#f4f6f7",
          100: "#e3e7ea",
          200: "#c9d2d7",
          300: "#a4b2bb",
          400: "#788b98",
          500: "#5d707d",
          600: "#505e6a",
          700: "#454f59",
          800: "#3d454c",
          900: "#363c42",
          950: "#21262b",
        },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
        display: ["var(--font-syne)", "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "gradient": "gradient 8s linear infinite",
        "float": "float 6s ease-in-out infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        gradient: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        glow: {
          "0%": { boxShadow: "0 0 20px rgba(255, 127, 17, 0.3)" },
          "100%": { boxShadow: "0 0 40px rgba(255, 127, 17, 0.6)" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "hero-pattern": "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\")",
      },
    },
  },
  plugins: [],
};

export default config;

