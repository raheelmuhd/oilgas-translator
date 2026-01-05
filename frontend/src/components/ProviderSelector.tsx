import React from 'react';

interface ProviderInfo {
  available: boolean;
  speed: string;
  estimate: string;
  cost: string;
}

interface ProviderSelectorProps {
  providers: {
    ollama?: ProviderInfo;
    nllb: ProviderInfo;
    deepseek: ProviderInfo;
  };
  selectedProvider: string;
  onSelect: (provider: string) => void;
  gpuAvailable: boolean;
}

export function ProviderSelector({
  providers,
  selectedProvider,
  onSelect,
  gpuAvailable
}: ProviderSelectorProps) {
  // Check if Ollama is available
  const ollamaAvailable = providers.ollama?.available ?? false;

  return (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Translation Method
      </label>
      <div className="space-y-2">
        {/* Ollama Option (Primary FREE - qwen3:8b) */}
        {ollamaAvailable && (
          <label className={`flex items-center p-3 border rounded-lg cursor-pointer ${
            selectedProvider === 'ollama' ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
          }`}>
            <input
              type="radio"
              name="provider"
              value="ollama"
              checked={selectedProvider === 'ollama'}
              onChange={() => onSelect('ollama')}
              className="mr-3"
            />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="font-medium">Free - High Accuracy (Ollama)</span>
                <span className="text-green-600 text-sm">$0</span>
              </div>
              <p className="text-sm text-gray-500">
                {gpuAvailable ? (
                  <>🚀 qwen3:8b with GPU - Best free option for technical documents</>
                ) : (
                  <>🧠 qwen3:8b (CPU) - Best free option, but slower without GPU</>
                )}
              </p>
            </div>
          </label>
        )}

        {/* NLLB Option (Fallback FREE) */}
        <label className={`flex items-center p-3 border rounded-lg cursor-pointer ${
          selectedProvider === 'nllb' ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
        }`}>
          <input
            type="radio"
            name="provider"
            value="nllb"
            checked={selectedProvider === 'nllb'}
            onChange={() => onSelect('nllb')}
            className="mr-3"
          />
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <span className="font-medium">Free - Fast (NLLB)</span>
              <span className="text-green-600 text-sm">{providers.nllb.cost}</span>
            </div>
            <p className="text-sm text-gray-500">
              {gpuAvailable ? (
                <>🚀 GPU accelerated - {providers.nllb.estimate}</>
              ) : (
                <>⚠️ CPU only - {providers.nllb.estimate} (slow)</>
              )}
            </p>
          </div>
        </label>

        {/* Paid Option (DeepSeek) */}
        {providers.deepseek.available && (
          <label className={`flex items-center p-3 border rounded-lg cursor-pointer ${
            selectedProvider === 'deepseek' ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
          }`}>
            <input
              type="radio"
              name="provider"
              value="deepseek"
              checked={selectedProvider === 'deepseek'}
              onChange={() => onSelect('deepseek')}
              className="mr-3"
            />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="font-medium">Fast (DeepSeek API)</span>
                <span className="text-orange-600 text-sm">{providers.deepseek.cost}</span>
              </div>
              <p className="text-sm text-gray-500">
                ⚡ {providers.deepseek.estimate}
              </p>
            </div>
          </label>
        )}

        {/* Info message */}
        {!ollamaAvailable && (
          <div className="p-3 border border-dashed border-gray-300 rounded-lg text-gray-500 text-sm">
            💡 <strong>Want better accuracy?</strong> Install Ollama with qwen3:8b for high-quality free translations.
          </div>
        )}
      </div>
    </div>
  );
}
