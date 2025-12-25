"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  FileText,
  Languages,
  Zap,
  Shield,
  Clock,
  CheckCircle,
  XCircle,
  Download,
  Loader2,
  ChevronDown,
  Fuel,
  Settings,
  Sparkles,
  DollarSign,
  Crown,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Job {
  job_id: string;
  status: "pending" | "extracting" | "translating" | "completed" | "failed";
  progress: number;
  message: string;
  filename?: string;
}

interface Language {
  code: string;
  name: string;
}

const LANGUAGES: Language[] = [
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "ar", name: "Arabic" },
  { code: "pt", name: "Portuguese" },
  { code: "ru", name: "Russian" },
  { code: "zh", name: "Chinese" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "hi", name: "Hindi" },
  { code: "tr", name: "Turkish" },
  { code: "nl", name: "Dutch" },
  { code: "pl", name: "Polish" },
  { code: "uk", name: "Ukrainian" },
  { code: "vi", name: "Vietnamese" },
  { code: "th", name: "Thai" },
  { code: "id", name: "Indonesian" },
  { code: "ms", name: "Malay" },
];

const MODES = [
  {
    id: "self_hosted",
    name: "Free",
    description: "CPU-based, slower but completely free",
    icon: Sparkles,
    color: "from-green-500 to-emerald-600",
    cost: "$0",
  },
  {
    id: "budget",
    name: "Budget",
    description: "DeepSeek API, good balance of speed and cost",
    icon: DollarSign,
    color: "from-blue-500 to-cyan-600",
    cost: "~$3/doc",
  },
  {
    id: "premium",
    name: "Premium",
    description: "Claude + Azure, highest accuracy",
    icon: Crown,
    color: "from-amber-500 to-orange-600",
    cost: "~$220/doc",
  },
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState<string>("");
  const [targetLang, setTargetLang] = useState<string>("en");
  const [mode, setMode] = useState<string>("self_hosted");
  const [isUploading, setIsUploading] = useState(false);
  const [currentJob, setCurrentJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [quickText, setQuickText] = useState("");
  const [quickResult, setQuickResult] = useState<string | null>(null);
  const [isQuickTranslating, setIsQuickTranslating] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
      "image/*": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
    },
    maxFiles: 1,
  });

  const uploadFile = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_language", targetLang);
    if (sourceLang) formData.append("source_language", sourceLang);

    try {
      const response = await fetch(`${API_URL}/api/v1/translate`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      setCurrentJob({
        job_id: data.job_id,
        status: "pending",
        progress: 0,
        message: data.message,
        filename: file.name,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  // Poll for job status
  useEffect(() => {
    if (!currentJob || currentJob.status === "completed" || currentJob.status === "failed") {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/api/v1/status/${currentJob.job_id}`);
        if (response.ok) {
          const data = await response.json();
          setCurrentJob({
            ...currentJob,
            status: data.status,
            progress: data.progress,
            message: data.message,
          });
        }
      } catch (err) {
        console.error("Failed to poll status:", err);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [currentJob]);

  const downloadResult = async () => {
    if (!currentJob) return;

    try {
      const response = await fetch(`${API_URL}/api/v1/download/${currentJob.job_id}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${currentJob.filename?.replace(/\.[^/.]+$/, "")}_translated.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (err) {
      setError("Failed to download file");
    }
  };

  const quickTranslate = async () => {
    if (!quickText.trim()) return;

    setIsQuickTranslating(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/v1/translate/quick`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: quickText,
          target_language: targetLang,
          source_language: sourceLang || undefined,
        }),
      });

      if (!response.ok) throw new Error("Translation failed");

      const data = await response.json();
      setQuickResult(data.translated_text);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Translation failed");
    } finally {
      setIsQuickTranslating(false);
    }
  };

  const resetJob = () => {
    setFile(null);
    setCurrentJob(null);
    setError(null);
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative pt-12 pb-20 px-4">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <div className="flex items-center justify-center gap-3 mb-6">
              <motion.div
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 4, repeat: Infinity }}
                className="w-14 h-14 rounded-2xl bg-gradient-to-br from-flame-500 to-flame-600 flex items-center justify-center shadow-lg shadow-flame-500/30"
              >
                <Fuel className="w-8 h-8 text-white" />
              </motion.div>
            </div>
            <h1 className="text-4xl md:text-6xl font-display font-bold mb-4 tracking-tight">
              <span className="text-rig-100">Oil & Gas</span>{" "}
              <span className="gradient-text">Document Translator</span>
            </h1>
            <p className="text-lg md:text-xl text-rig-400 max-w-2xl mx-auto">
              Production-grade translation for technical documents.
              <br className="hidden md:block" />
              <span className="text-flame-400">97%+ accuracy</span> across{" "}
              <span className="text-petroleum-400">20+ languages</span>.
            </p>
          </motion.div>

          {/* Mode Selector */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12 max-w-3xl mx-auto"
          >
            {MODES.map((m) => (
              <button
                key={m.id}
                onClick={() => setMode(m.id)}
                className={`relative p-4 rounded-xl border transition-all duration-300 text-left ${
                  mode === m.id
                    ? "border-flame-500/50 bg-flame-500/10"
                    : "border-rig-700/50 bg-rig-900/30 hover:border-rig-600/50"
                }`}
              >
                {mode === m.id && (
                  <motion.div
                    layoutId="mode-indicator"
                    className="absolute inset-0 rounded-xl border-2 border-flame-500"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <div className="relative z-10">
                  <div className="flex items-center gap-2 mb-2">
                    <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${m.color} flex items-center justify-center`}>
                      <m.icon className="w-4 h-4 text-white" />
                    </div>
                    <span className="font-semibold text-rig-100">{m.name}</span>
                    <span className="ml-auto text-sm font-mono text-flame-400">{m.cost}</span>
                  </div>
                  <p className="text-xs text-rig-400">{m.description}</p>
                </div>
              </button>
            ))}
          </motion.div>

          {/* Main Upload Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="glass rounded-3xl p-8 md:p-10 max-w-3xl mx-auto"
          >
            <AnimatePresence mode="wait">
              {!currentJob ? (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  {/* Dropzone */}
                  <div
                    {...getRootProps()}
                    className={`dropzone ${isDragActive ? "active" : ""}`}
                  >
                    <input {...getInputProps()} />
                    <div className="text-center">
                      {file ? (
                        <motion.div
                          initial={{ scale: 0.9 }}
                          animate={{ scale: 1 }}
                          className="flex flex-col items-center"
                        >
                          <div className="w-16 h-16 rounded-2xl bg-flame-500/20 flex items-center justify-center mb-4">
                            <FileText className="w-8 h-8 text-flame-400" />
                          </div>
                          <p className="text-lg font-medium text-rig-100 mb-1">{file.name}</p>
                          <p className="text-sm text-rig-400">
                            {(file.size / (1024 * 1024)).toFixed(2)} MB
                          </p>
                        </motion.div>
                      ) : (
                        <>
                          <motion.div
                            animate={{ y: isDragActive ? -10 : 0 }}
                            className="w-16 h-16 rounded-2xl bg-rig-800/50 flex items-center justify-center mx-auto mb-4"
                          >
                            <Upload className="w-8 h-8 text-rig-400" />
                          </motion.div>
                          <p className="text-lg text-rig-200 mb-2">
                            {isDragActive ? "Drop your file here" : "Drop your document here"}
                          </p>
                          <p className="text-sm text-rig-500">
                            PDF, DOCX, XLSX, PPTX, or images up to 600MB
                          </p>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Language Selection */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                    <div>
                      <label className="block text-sm font-medium text-rig-300 mb-2">
                        Source Language
                      </label>
                      <div className="relative">
                        <select
                          value={sourceLang}
                          onChange={(e) => setSourceLang(e.target.value)}
                          className="input-field appearance-none pr-10"
                        >
                          <option value="">Auto-detect</option>
                          {LANGUAGES.map((lang) => (
                            <option key={lang.code} value={lang.code}>
                              {lang.name}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-rig-500 pointer-events-none" />
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-rig-300 mb-2">
                        Target Language
                      </label>
                      <div className="relative">
                        <select
                          value={targetLang}
                          onChange={(e) => setTargetLang(e.target.value)}
                          className="input-field appearance-none pr-10"
                        >
                          {LANGUAGES.map((lang) => (
                            <option key={lang.code} value={lang.code}>
                              {lang.name}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-rig-500 pointer-events-none" />
                      </div>
                    </div>
                  </div>

                  {/* Error Display */}
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30"
                      >
                        <div className="flex items-center gap-2 text-red-400">
                          <XCircle className="w-5 h-5" />
                          <span>{error}</span>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Upload Button */}
                  <button
                    onClick={uploadFile}
                    disabled={!file || isUploading}
                    className="btn-primary w-full mt-6 flex items-center justify-center gap-2"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Languages className="w-5 h-5" />
                        Translate Document
                      </>
                    )}
                  </button>
                </motion.div>
              ) : (
                <motion.div
                  key="progress"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-center"
                >
                  {/* Progress Display */}
                  <div className="mb-6">
                    {currentJob.status === "completed" ? (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="w-20 h-20 rounded-full bg-green-500/20 flex items-center justify-center mx-auto"
                      >
                        <CheckCircle className="w-10 h-10 text-green-400" />
                      </motion.div>
                    ) : currentJob.status === "failed" ? (
                      <div className="w-20 h-20 rounded-full bg-red-500/20 flex items-center justify-center mx-auto">
                        <XCircle className="w-10 h-10 text-red-400" />
                      </div>
                    ) : (
                      <div className="w-20 h-20 rounded-full bg-flame-500/20 flex items-center justify-center mx-auto relative">
                        <Loader2 className="w-10 h-10 text-flame-400 animate-spin" />
                        <div className="absolute inset-0 rounded-full border-4 border-rig-700">
                          <svg className="w-full h-full -rotate-90">
                            <circle
                              cx="40"
                              cy="40"
                              r="36"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="4"
                              className="text-flame-500"
                              strokeDasharray={`${2 * Math.PI * 36}`}
                              strokeDashoffset={`${2 * Math.PI * 36 * (1 - currentJob.progress / 100)}`}
                              strokeLinecap="round"
                            />
                          </svg>
                        </div>
                      </div>
                    )}
                  </div>

                  <h3 className="text-xl font-semibold text-rig-100 mb-2">
                    {currentJob.status === "completed"
                      ? "Translation Complete!"
                      : currentJob.status === "failed"
                      ? "Translation Failed"
                      : "Translating..."}
                  </h3>

                  <p className="text-rig-400 mb-4">{currentJob.message}</p>

                  {currentJob.status !== "completed" && currentJob.status !== "failed" && (
                    <div className="max-w-xs mx-auto mb-6">
                      <div className="progress-bar">
                        <motion.div
                          className="progress-bar-fill"
                          initial={{ width: 0 }}
                          animate={{ width: `${currentJob.progress}%` }}
                        />
                      </div>
                      <p className="text-sm text-rig-500 mt-2">
                        {Math.round(currentJob.progress)}% complete
                      </p>
                    </div>
                  )}

                  <div className="flex gap-3 justify-center">
                    {currentJob.status === "completed" && (
                      <button onClick={downloadResult} className="btn-primary flex items-center gap-2">
                        <Download className="w-5 h-5" />
                        Download Translation
                      </button>
                    )}
                    <button onClick={resetJob} className="btn-secondary">
                      {currentJob.status === "completed" ? "Translate Another" : "Cancel"}
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Quick Translation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-8 max-w-3xl mx-auto"
          >
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-rig-400 hover:text-rig-200 transition-colors mx-auto mb-4"
            >
              <Settings className="w-4 h-4" />
              <span className="text-sm">Quick Text Translation</span>
              <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
            </button>

            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="glass rounded-2xl p-6 overflow-hidden"
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-rig-300 mb-2">
                        Text to Translate
                      </label>
                      <textarea
                        value={quickText}
                        onChange={(e) => setQuickText(e.target.value)}
                        placeholder="Enter text to translate..."
                        className="input-field h-32 resize-none"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-rig-300 mb-2">
                        Translation
                      </label>
                      <div className="input-field h-32 overflow-auto">
                        {isQuickTranslating ? (
                          <div className="flex items-center gap-2 text-rig-400">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Translating...
                          </div>
                        ) : quickResult ? (
                          <p className="text-rig-100">{quickResult}</p>
                        ) : (
                          <p className="text-rig-500">Translation will appear here...</p>
                        )}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={quickTranslate}
                    disabled={!quickText.trim() || isQuickTranslating}
                    className="btn-primary mt-4"
                  >
                    {isQuickTranslating ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <>
                        <Zap className="w-4 h-4 inline mr-2" />
                        Quick Translate
                      </>
                    )}
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="section-title mb-4">Why Choose Us?</h2>
            <p className="text-rig-400 max-w-xl mx-auto">
              Built specifically for oil & gas industry documents with specialized terminology
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                icon: Shield,
                title: "97%+ Accuracy",
                description: "Azure Document Intelligence OCR with Claude AI translation ensures industry-leading accuracy for technical documents.",
                color: "from-green-500 to-emerald-600",
              },
              {
                icon: Languages,
                title: "20+ Languages",
                description: "Full support for Arabic, Russian, Chinese, Spanish, Portuguese, and more with specialized O&G terminology.",
                color: "from-blue-500 to-cyan-600",
              },
              {
                icon: Clock,
                title: "Fast Processing",
                description: "Background processing with real-time progress updates. Handle documents up to 600MB efficiently.",
                color: "from-amber-500 to-orange-600",
              },
            ].map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="card group"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-rig-100 mb-2">{feature.title}</h3>
                <p className="text-rig-400">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Terminology Section */}
      <section className="py-20 px-4 bg-rig-900/30">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="section-title mb-4">O&G Terminology Support</h2>
            <p className="text-rig-400 max-w-xl mx-auto">
              200+ curated technical terms across 8 languages for accurate translations
            </p>
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              "Drilling",
              "Production",
              "Reservoir",
              "Safety (HSE)",
              "Equipment",
              "Geology",
              "Economics",
              "Operations",
            ].map((category, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className="p-4 rounded-xl bg-rig-800/30 border border-rig-700/30 text-center hover:border-flame-500/30 transition-colors"
              >
                <span className="text-rig-200 font-medium">{category}</span>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-8 p-6 rounded-2xl bg-rig-800/20 border border-rig-700/30"
          >
            <p className="text-sm text-rig-400 text-center">
              <span className="text-flame-400 font-semibold">Example terms:</span> BHA, WOB, ROP, MWD, LWD, ESP, BOP, H2S, SIMOPS, PTW, JSA, PDC, CAPEX, OPEX, NPV, IRR, EOR, PSC
            </p>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-rig-800/50">
        <div className="max-w-6xl mx-auto text-center text-rig-500 text-sm">
          <p>Oil & Gas Document Translator v1.0.0</p>
          <p className="mt-1">
            Built with FastAPI, Next.js, NLLB, PaddleOCR, and Claude AI
          </p>
        </div>
      </footer>
    </div>
  );
}

