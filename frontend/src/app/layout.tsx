import type { Metadata } from "next";
import { Syne, Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
});

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-syne",
});

export const metadata: Metadata = {
  title: "Oil & Gas Document Translator | Fast & Accurate Technical Translation",
  description: "Production-grade document translation specialized for the oil and gas industry. Translate technical documents with 97%+ accuracy in 20+ languages.",
  keywords: ["oil and gas", "document translation", "technical translation", "PDF translation", "multilingual"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${outfit.variable} ${syne.variable}`}>
      <body className="font-sans min-h-screen bg-rig-950 overflow-x-hidden">
        <div className="fixed inset-0 bg-hero-pattern opacity-50 pointer-events-none" />
        <div className="fixed inset-0 bg-gradient-radial from-petroleum-950/20 via-transparent to-transparent pointer-events-none" />
        <main className="relative z-10">
          {children}
        </main>
      </body>
    </html>
  );
}

