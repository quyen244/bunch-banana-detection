'use client';

import React, { useState, useCallback, useRef } from 'react';
import { Upload, Loader2, RotateCcw, Banana, AlertCircle, CheckCircle2 } from 'lucide-react';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
const PREDICT_URL = `${BACKEND_URL}/predict`;

// Resize image on client before upload — YOLO only uses 640×480 internally,
// so sending a 12MP original wastes bandwidth with zero accuracy gain.
function resizeImageFile(file: File, maxDim = 1280): Promise<File> {
  return new Promise((resolve) => {
    const img = new window.Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const scale = Math.min(maxDim / img.width, maxDim / img.height, 1);
      if (scale === 1) { resolve(file); return; }
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => resolve(blob ? new File([blob], file.name, { type: 'image/jpeg' }) : file),
        'image/jpeg',
        0.85,
      );
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

interface PredictResponse {
  bunch_count: number;
  confidences: number[];
  annotated_image: string;
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (selected: File) => {
    if (!selected.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      return;
    }
    setResult(null);
    setError(null);
    const resized = await resizeImageFile(selected);
    setFile(resized);
    setPreview(URL.createObjectURL(resized));
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const handlePredict = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(PREDICT_URL, { method: 'POST', body: form });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as { detail?: string }).detail ?? `Server error ${res.status}`);
      }

      const data: PredictResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <main className="min-h-screen bg-slate-50 py-12 px-4">
      {/* Header */}
      <div className="max-w-3xl mx-auto text-center mb-12">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-yellow-50 text-yellow-700 text-sm font-semibold mb-5">
          <Banana className="w-4 h-4" />
          Powered by CS431 AI Team
        </div>
        <h1 className="text-4xl font-black text-slate-900 tracking-tight mb-3">
          Banana Bunch <span className="text-yellow-500">Detection</span>
        </h1>
        <p className="text-slate-500 text-lg">
          Upload a banana bunch image — YOLO will locate and count every bunch.
        </p>
      </div>

      <div className="max-w-3xl mx-auto space-y-6">
        {/* Upload Zone */}
        {!result && (
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-colors
              ${isDragging ? 'border-yellow-400 bg-yellow-50' : 'border-slate-200 bg-white hover:border-yellow-300 hover:bg-yellow-50/40'}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={onInputChange}
            />

            {preview ? (
              <div className="space-y-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={preview}
                  alt="Preview"
                  className="mx-auto max-h-72 rounded-xl object-contain shadow"
                />
                <p className="text-sm text-slate-500">{file?.name}</p>
                <p className="text-xs text-slate-400">Click to change image</p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-yellow-100">
                  <Upload className="h-7 w-7 text-yellow-600" />
                </div>
                <p className="font-semibold text-slate-700">Drag & drop or click to upload</p>
                <p className="text-sm text-slate-400">PNG, JPG, JPEG supported</p>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="flex items-center gap-3 rounded-xl bg-red-50 px-5 py-4 text-red-600">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <span className="text-sm font-medium">{error}</span>
          </div>
        )}

        {/* Action buttons */}
        {file && !result && (
          <div className="flex gap-3">
            <button
              onClick={handleReset}
              disabled={isLoading}
              className="flex items-center gap-2 rounded-xl border-2 border-slate-200 px-5 py-3 font-semibold text-slate-500 hover:bg-slate-100 transition-colors disabled:opacity-50"
            >
              <RotateCcw className="h-4 w-4" />
              Reset
            </button>
            <button
              onClick={handlePredict}
              disabled={isLoading}
              className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-yellow-400 px-6 py-3 font-bold text-slate-900 shadow-md hover:bg-yellow-500 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Detecting...
                </>
              ) : (
                <>
                  <Banana className="h-5 w-5" />
                  Detect Bunches
                </>
              )}
            </button>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-5">
            {/* Summary badge */}
            <div className="flex items-center justify-center gap-3">
              <div className="flex items-center gap-2 rounded-full bg-green-100 px-5 py-2 text-green-700 font-semibold">
                <CheckCircle2 className="h-5 w-5" />
                {result.bunch_count} bunch{result.bunch_count !== 1 ? 'es' : ''} detected
              </div>
            </div>

            {/* Annotated image */}
            <div className="overflow-hidden rounded-2xl border border-slate-100 bg-white shadow">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/jpeg;base64,${result.annotated_image}`}
                alt="Detection result"
                className="w-full object-contain"
              />
            </div>

            {/* Confidence scores */}
            {result.confidences.length > 0 && (
              <div className="rounded-2xl border border-slate-100 bg-white p-5 shadow">
                <p className="mb-4 text-sm font-semibold text-slate-500 uppercase tracking-wide">
                  Confidence per detection
                </p>
                <div className="space-y-3">
                  {result.confidences.map((conf, i) => (
                    <div key={i} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium text-slate-700">Bunch #{i + 1}</span>
                        <span className="font-bold text-slate-900">{(conf * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-yellow-400 transition-all"
                          style={{ width: `${conf * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button
              onClick={handleReset}
              className="flex w-full items-center justify-center gap-2 rounded-xl border-2 border-slate-200 py-3 font-semibold text-slate-600 hover:bg-slate-50 transition-colors"
            >
              <RotateCcw className="h-4 w-4" />
              Try another image
            </button>
          </div>
        )}
      </div>

      <footer className="mt-20 text-center text-slate-400 text-sm">
        &copy; 2026 Banana Bunch Detection &mdash; CS431 AI Team
      </footer>
    </main>
  );
}
