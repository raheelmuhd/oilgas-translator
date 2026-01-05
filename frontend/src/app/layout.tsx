import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Document Translator | Oil & Gas Industry",
  description: "Professional document translation system specialized for the oil and gas industry. High-accuracy translation with technical terminology support.",
  keywords: ["oil and gas", "document translation", "technical translation", "PDF translation"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans min-h-screen bg-gray-50">
        {children}
      </body>
    </html>
  );
}
