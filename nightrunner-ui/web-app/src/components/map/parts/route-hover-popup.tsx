import { Popup } from 'react-map-gl/maplibre';
import { MoveHorizontal, Clock, ShieldCheck, Lightbulb } from 'lucide-react';
import { formatDuration } from '@/utils/date-time';
import type { Summary } from '@/components/types';

interface RouteHoverPopupProps {
  lng: number;
  lat: number;
  summary: Summary;
}

export function RouteHoverPopup({ lng, lat, summary }: RouteHoverPopupProps) {
  // safety_score is injected by our reranker proxy (0-100, 100 = route
  // never passes near an edge our ML model flagged as risky). Optional in
  // the Summary type, so this is undefined if the reranker was bypassed.
  const safetyScore = summary.safety_score;

  // lighting_score is injected the same way (0-100, 100 = route passes
  // near a lit street for its entire length). Purely descriptive.
  const lightingScore = summary.lighting_score;

  return (
    <Popup
      longitude={lng}
      latitude={lat}
      anchor="bottom"
      closeButton={false}
      closeOnClick={false}
      maxWidth="none"
    >
      <div className="min-w-[120px] px-2">
        <div className="font-bold text-muted-foreground">Route Summary</div>
        <div className="flex items-center gap-1">
          <MoveHorizontal className="size-3.5" />
          <span>
            {`${summary.length.toFixed(summary.length > 1000 ? 0 : 1)} km`}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Clock className="size-3.5" />
          <span>{formatDuration(summary.time)}</span>
        </div>
        {typeof safetyScore === 'number' && (
          <div className="flex items-center gap-1">
            <ShieldCheck className="size-3.5" />
            <span>{`Safety: ${safetyScore.toFixed(0)}/100`}</span>
          </div>
        )}
        {typeof lightingScore === 'number' && (
          <div className="flex items-center gap-1">
            <Lightbulb className="size-3.5" />
            <span>{`Lighting: ${lightingScore.toFixed(0)}/100`}</span>
          </div>
        )}
      </div>
    </Popup>
  );
}
