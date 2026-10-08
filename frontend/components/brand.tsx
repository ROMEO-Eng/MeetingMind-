import Link from "next/link";
import { AudioLines } from "lucide-react";

export default function Brand() {
  return (
    <Link className="brand" href="/" aria-label="MeetingMind AI home">
      <span className="brand-mark" aria-hidden="true">
        <AudioLines size={17} strokeWidth={2.2} />
      </span>
      <span className="brand-wordmark">
        Meeting<span className="brand-accent">Mind</span>
      </span>
      <span className="brand-description">AI Meeting Intelligence</span>
    </Link>
  );
}
