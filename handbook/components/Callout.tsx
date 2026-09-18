import type { ReactNode } from "react";

type Kind = "note" | "warn" | "danger" | "ok";

const LABELS: Record<Kind, string> = {
  note: "Note",
  warn: "Watch this",
  danger: "Do not do this",
  ok: "In our repo",
};

export function Callout({
  kind = "note",
  title,
  children,
}: {
  kind?: Kind;
  title?: string;
  children: ReactNode;
}) {
  return (
    <div className={`callout ${kind}`}>
      <div className="label">{title ?? LABELS[kind]}</div>
      {children}
    </div>
  );
}
