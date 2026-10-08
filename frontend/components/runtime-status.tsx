"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import type { HealthStatus } from "@/lib/types";

const POLL_INTERVAL_MS = 8_000;

export default function RuntimeStatus() {
  const [result, setResult] = useState<{ checked: boolean; health: HealthStatus | null }>({
    checked: false,
    health: null,
  });

  useEffect(() => {
    let active = true;

    const refresh = async () => {
      try {
        const currentHealth = await fetchHealth();
        if (active) setResult({ checked: true, health: currentHealth });
      } catch {
        if (active) setResult({ checked: true, health: null });
      }
    };

    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const { checked, health } = result;
  const apiConnected = health !== null;
  const modelStatus = health?.model_status ?? "checking";
  const gpuLabel = !checked
    ? "GPU Checking"
    : !apiConnected
      ? "GPU Unknown"
      : health.gpu_available
        ? "GPU Detected"
        : "GPU Not Detected";
  const modelLabel = !checked
    ? "AI Checking"
    : !apiConnected
      ? "AI Offline"
      : modelStatus === "ready"
      ? "AI Ready"
      : modelStatus === "loading"
        ? "AI Loading"
        : "AI Unavailable";
  const detail = !checked
    ? "Checking the backend and local model status."
    : !apiConnected
      ? "The MeetingMind API is not reachable. Check that the backend is running."
      : health.model_status_detail;

  return (
    <div className="runtime-status" aria-live="polite">
      <span className={`runtime-api ${!checked ? "is-checking" : apiConnected ? "is-connected" : "is-disconnected"}`}>
        <span className="runtime-dot" />
        {!checked ? "API Checking" : apiConnected ? "API Connected" : "API Offline"}
      </span>
      <span
        className={`runtime-gpu ${!checked ? "is-checking" : !apiConnected ? "is-unknown" : health.gpu_available ? "is-detected" : "is-unavailable"}`}
        title={apiConnected ? (health.gpu_available ? "GPU hardware detected by the backend." : "No GPU hardware detected by the backend.") : "GPU status is unknown while the backend is unreachable."}
      >
        <span className="runtime-dot" />
        {gpuLabel}
      </span>
      <details className={`runtime-ai is-${modelStatus}`} title={detail}>
        <summary>
          <span className="runtime-dot" />
          {modelLabel}
        </summary>
        <div className="runtime-details">
          <strong>{apiConnected ? "MeetingMind status" : "Backend status"}</strong>
          <p>{detail}</p>
          {health && <span>Model: {health.model_name}</span>}
        </div>
      </details>
    </div>
  );
}
