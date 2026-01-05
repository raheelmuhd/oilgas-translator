"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  FileText,
  Languages,
  CheckCircle,
  XCircle,
  Download,
  Loader2,
  ChevronDown,
  AlertCircle,
  File,
  Clock,
  Settings,
} from "lucide-react";
import { useSystemInfo } from "../hooks/useSystemInfo";
import { SpeedWarning } from "../components/SpeedWarning";
import { ProviderSelector } from "../components/ProviderSelector";

// Get API URL from environment or default to backend
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Helper to check if backend is accessible
async function checkBackendHealth(): Promise<boolean> {
  try {
    // Try multiple endpoints in case one fails
    const endpoints = [`${API_URL}/health`, `${API_URL}/api/v1/health`];
    
    for (const endpoint of endpoints) {
      try {
        // Create timeout manually (AbortSignal.timeout not available in all browsers)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // Increased to 5 seconds
        
        const response = await fetch(endpoint, { 
          method: "GET",
          signal: controller.signal,
          mode: 'cors', // Explicitly enable CORS
          credentials: 'omit', // Don't send credentials for health check
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
          console.log(`✅ Backend connected via ${endpoint}`);
          return true;
        }
      } catch (err) {
        // Try next endpoint
        continue;
      }
    }
    
    console.warn("⚠️ Backend health check failed - all endpoints unreachable");
    return false;
  } catch (error) {
    console.error("Backend health check error:", error);
    return false;
  }
}

// Error boundary component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Error caught by boundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
          <div className="bg-white rounded-lg border border-red-200 p-8 max-w-md">
            <h2 className="text-xl font-semibold text-red-600 mb-4">Something went wrong</h2>
            <p className="text-gray-600 mb-4">{this.state.error?.message}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

interface Job {
  job_id: string;
  status: "pending" | "extracting" | "translating" | "completed" | "failed";
  progress: number;
  message: string;
  filename?: string;
  current_page?: number;
  total_pages?: number;
  processed_chunks?: number;
  total_chunks?: number;
  translation_provider?: string;
  gpu_used?: boolean;
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

function HomeComponent() {
  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState<string>("");
  const [targetLang, setTargetLang] = useState<string>("en");
  const [translationProvider, setTranslationProvider] = useState<string>("ollama");  // Default to Ollama/qwen3:8b
  const [isUploading, setIsUploading] = useState(false);
  const [currentJob, setCurrentJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [elapsedTime, setElapsedTime] = useState<string>("0:00");
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
  const progressRef = useRef<number>(0);
  const { systemInfo, loading: systemLoading } = useSystemInfo();

  // Check backend connection on mount and retry if failed
  useEffect(() => {
    let mounted = true;
    
    const checkConnection = async () => {
      try {
        const connected = await checkBackendHealth();
        if (mounted) {
          setBackendConnected(connected);
        }
        
        // Retry after 2 seconds if not connected
        if (!connected && mounted) {
          setTimeout(async () => {
            if (mounted) {
              const retry = await checkBackendHealth();
              setBackendConnected(retry);
            }
          }, 2000);
        }
      } catch (error) {
        console.error("Connection check error:", error);
        if (mounted) {
          setBackendConnected(false);
        }
      }
    };
    
    checkConnection();
    
    // Check every 5 seconds if not connected
    const interval = setInterval(() => {
      if (mounted && backendConnected === false) {
        checkBackendHealth().then((connected) => {
          if (mounted) {
            setBackendConnected(connected);
          }
        });
      }
    }, 5000);
    
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

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
    setStartTime(new Date());
    progressRef.current = 0;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_language", targetLang);
    if (sourceLang) formData.append("source_language", sourceLang);
    formData.append("translation_provider", translationProvider);

    try {
      const response = await fetch(`${API_URL}/api/v1/translate`, {
        method: "POST",
        body: formData,
        headers: {
          // Don't set Content-Type, let browser set it with boundary for multipart/form-data
        },
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
        translation_provider: translationProvider,
        gpu_used: systemInfo?.gpu.gpu_available && translationProvider === 'nllb',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setIsUploading(false);
    } finally {
      setIsUploading(false);
    }
  };

  // Poll for job status with better progress tracking
  useEffect(() => {
    if (!currentJob || currentJob.status === "completed" || currentJob.status === "failed") {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/api/v1/status/${currentJob.job_id}`);
        if (response.ok) {
          const data = await response.json();
          
          // Update progress smoothly
          const newProgress = Math.max(progressRef.current, data.progress || 0);
          progressRef.current = newProgress;
          
          setCurrentJob((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              status: data.status,
              progress: newProgress,
              message: data.message,
              current_page: data.current_page || prev.current_page,
              total_pages: data.total_pages || prev.total_pages,
              processed_chunks: data.processed_chunks || prev.processed_chunks,
              total_chunks: data.total_chunks || prev.total_chunks,
              translation_provider: data.translation_provider || prev.translation_provider,
              gpu_used: data.gpu_used !== undefined ? data.gpu_used : prev.gpu_used,
            };
          });
        }
      } catch (err) {
        console.error("Failed to poll status:", err);
      }
    }, 1000); // Poll every second for smoother updates

    return () => clearInterval(pollInterval);
  }, [currentJob]);

  // Elapsed time counter
  useEffect(() => {
    if (!startTime || !currentJob || currentJob.status === "completed" || currentJob.status === "failed") {
      return;
    }

    const timer = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime.getTime()) / 1000);
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;
      setElapsedTime(`${minutes}:${seconds.toString().padStart(2, "0")}`);
    }, 1000);

    return () => clearInterval(timer);
  }, [startTime, currentJob]);

  const downloadResult = async () => {
    if (!currentJob) return;

    try {
      const response = await fetch(`${API_URL}/api/v1/download/${currentJob.job_id}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        // Download as .docx or .txt based on file extension from response headers
        const contentDisposition = response.headers.get('content-disposition');
        const filenameMatch = contentDisposition?.match(/filename="?([^"]+)"?/);
        const filename = filenameMatch ? filenameMatch[1] : `${currentJob.filename?.replace(/\.[^/.]+$/, "")}_translated.docx`;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (err) {
      setError("Failed to download file");
    }
  };

  const resetJob = () => {
    setFile(null);
    setCurrentJob(null);
    setError(null);
    setStartTime(null);
    setElapsedTime("0:00");
    progressRef.current = 0;
  };

  const getStatusMessage = () => {
    if (!currentJob) return "";
    
    if (currentJob.status === "extracting" && currentJob.current_page && currentJob.total_pages) {
      return `Processing page ${currentJob.current_page} of ${currentJob.total_pages}`;
    }
    if (currentJob.status === "translating" && currentJob.processed_chunks && currentJob.total_chunks) {
      return `Translating section ${currentJob.processed_chunks} of ${currentJob.total_chunks}`;
    }
    return currentJob.message || "Processing...";
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Professional Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Document Translator</h1>
                <p className="text-sm text-gray-500">Oil & Gas Industry Specialized</p>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-600">
              <span className="flex items-center space-x-1">
                <Clock className="w-4 h-4" />
                <span>{elapsedTime}</span>
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Main Upload Card */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          {!currentJob ? (
            <>
              {/* File Upload Area */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                  isDragActive
                    ? "border-blue-500 bg-blue-50"
                    : "border-gray-300 hover:border-gray-400"
                }`}
              >
                <input {...getInputProps()} />
                {file ? (
                  <div className="flex flex-col items-center">
                    <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                      <File className="w-8 h-8 text-blue-600" />
                    </div>
                    <p className="text-lg font-medium text-gray-900 mb-1">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Upload className="w-8 h-8 text-gray-400" />
                    </div>
                    <p className="text-lg font-medium text-gray-900 mb-2">
                      {isDragActive ? "Drop your file here" : "Upload Document"}
                    </p>
                    <p className="text-sm text-gray-500">
                      PDF, DOCX, XLSX, PPTX, or images up to 600MB
                    </p>
                  </>
                )}
              </div>

              {/* Speed Warning Banner */}
              {systemInfo?.speed_warning && (
                <SpeedWarning 
                  warning={systemInfo.speed_warning}
                  gpuAvailable={systemInfo.gpu.gpu_available}
                  onSelectFastProvider={() => setTranslationProvider('deepseek')}
                />
              )}

              {/* Provider Selection */}
              {systemInfo && (
                <ProviderSelector
                  providers={systemInfo.providers}
                  selectedProvider={translationProvider}
                  onSelect={setTranslationProvider}
                  gpuAvailable={systemInfo.gpu.gpu_available}
                />
              )}

              {/* Language Selection */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Source Language
                  </label>
                  <div className="relative">
                    <select
                      value={sourceLang}
                      onChange={(e) => setSourceLang(e.target.value)}
                      className="w-full px-4 py-2.5 border border-gray-300 rounded-lg text-gray-900 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 appearance-none pr-10"
                    >
                      <option value="">Auto-detect</option>
                      {LANGUAGES.map((lang) => (
                        <option key={lang.code} value={lang.code}>
                          {lang.name}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Target Language
                  </label>
                  <div className="relative">
                    <select
                      value={targetLang}
                      onChange={(e) => setTargetLang(e.target.value)}
                      className="w-full px-4 py-2.5 border border-gray-300 rounded-lg text-gray-900 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 appearance-none pr-10"
                    >
                      {LANGUAGES.map((lang) => (
                        <option key={lang.code} value={lang.code}>
                          {lang.name}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                  </div>
                </div>
              </div>

              {/* Backend Connection Status */}
              {backendConnected === false && (
                <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start space-x-3">
                  <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-yellow-800">Backend not connected</p>
                    <p className="text-xs text-yellow-700 mt-1">
                      Make sure the backend is running on {API_URL}
                    </p>
                    <p className="text-xs text-yellow-600 mt-2">
                      Check browser console (F12) for detailed error messages
                    </p>
                    <button
                      onClick={async () => {
                        const connected = await checkBackendHealth();
                        setBackendConnected(connected);
                      }}
                      className="mt-2 px-3 py-1 text-xs bg-yellow-100 text-yellow-800 rounded hover:bg-yellow-200 transition-colors"
                    >
                      Retry Connection
                    </button>
                  </div>
                </div>
              )}
              
              {backendConnected === true && (
                <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  <p className="text-xs text-green-800">Backend connected successfully</p>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
                  <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
                  <p className="text-sm text-red-800">{error}</p>
                </div>
              )}

              {/* Upload Button */}
              <button
                onClick={uploadFile}
                disabled={!file || isUploading}
                className="w-full mt-6 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <Languages className="w-5 h-5" />
                    <span>Translate Document</span>
                  </>
                )}
              </button>
            </>
          ) : (
            <>
              {/* Progress Display */}
              <div className="text-center mb-8">
                {currentJob.status === "completed" ? (
                  <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <CheckCircle className="w-10 h-10 text-green-600" />
                  </div>
                ) : currentJob.status === "failed" ? (
                  <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <XCircle className="w-10 h-10 text-red-600" />
                  </div>
                ) : (
                  <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 relative">
                    <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
                  </div>
                )}

                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {currentJob.status === "completed"
                    ? "Translation Complete"
                    : currentJob.status === "failed"
                    ? "Translation Failed"
                    : "Processing Document"}
                </h3>

                <p className="text-gray-600 mb-6">{getStatusMessage()}</p>

                {/* Provider Info Display */}
                {currentJob.translation_provider && (
                  <div className="text-sm text-gray-600 mb-4">
                    {currentJob.translation_provider === 'deepseek' ? (
                      <span className="text-blue-600">⚡ Using DeepSeek API (fast)</span>
                    ) : currentJob.translation_provider === 'claude' ? (
                      <span className="text-purple-600">⚡ Using Claude API (premium)</span>
                    ) : currentJob.translation_provider === 'ollama' ? (
                      systemInfo?.gpu.gpu_available ? (
                        <span className="text-green-600">🚀 Using FREE Ollama/qwen3:8b (GPU accelerated)</span>
                      ) : (
                        <span className="text-yellow-600">⏳ Using FREE Ollama/qwen3:8b (CPU - slower)</span>
                      )
                    ) : currentJob.translation_provider === 'nllb' ? (
                      currentJob.gpu_used || systemInfo?.gpu.gpu_available ? (
                        <span className="text-green-600">🚀 Using GPU-accelerated NLLB</span>
                      ) : (
                        <span className="text-yellow-600">⏳ Using CPU-based NLLB (slower)</span>
                      )
                    ) : (
                      <span className="text-gray-600">Using {currentJob.translation_provider}</span>
                    )}
                  </div>
                )}

                {/* Progress Bar */}
                {currentJob.status !== "completed" && currentJob.status !== "failed" && (
                  <div className="max-w-md mx-auto">
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-600 transition-all duration-500 ease-out"
                        style={{ width: `${currentJob.progress}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      {Math.round(currentJob.progress)}% complete
                    </p>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-3 justify-center mt-6">
                  {currentJob.status === "completed" && (
                    <button
                      onClick={downloadResult}
                      className="px-6 py-2.5 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
                    >
                      <Download className="w-5 h-5" />
                      <span>Download Translation</span>
                    </button>
                  )}
                  <button
                    onClick={resetJob}
                    className="px-6 py-2.5 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    {currentJob.status === "completed" ? "Translate Another" : "Cancel"}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Info Section */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-blue-600" />
              </div>
              <h4 className="font-semibold text-gray-900">Multi-Format</h4>
            </div>
            <p className="text-sm text-gray-600">
              Supports PDF, DOCX, XLSX, PPTX, and image files
            </p>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                <Languages className="w-5 h-5 text-green-600" />
              </div>
              <h4 className="font-semibold text-gray-900">20+ Languages</h4>
            </div>
            <p className="text-sm text-gray-600">
              Full support for major languages with O&G terminology
            </p>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                <Settings className="w-5 h-5 text-purple-600" />
              </div>
              <h4 className="font-semibold text-gray-900">High Accuracy</h4>
            </div>
            <p className="text-sm text-gray-600">
              Specialized terminology for oil & gas industry documents
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default function Home() {
  return (
    <ErrorBoundary>
      <HomeComponent />
    </ErrorBoundary>
  );
}
