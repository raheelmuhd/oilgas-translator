import { useState, useEffect } from 'react';

interface SystemInfo {
  gpu: {
    gpu_available: boolean;
    gpu_name: string | null;
    using_device: string;
  };
  device_recommendation: string;
  providers: {
    nllb: {
      available: boolean;
      speed: string;
      estimate: string;
      cost: string;
    };
    deepseek: {
      available: boolean;
      speed: string;
      estimate: string;
      cost: string;
    };
  };
  speed_warning: string | null;
  recommendation: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useSystemInfo() {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/v1/system-info`)
      .then(res => {
        if (!res.ok) {
          throw new Error('Failed to fetch system info');
        }
        return res.json();
      })
      .then(data => {
        setSystemInfo(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch system info:', err);
        setLoading(false);
      });
  }, []);

  return { systemInfo, loading };
}

