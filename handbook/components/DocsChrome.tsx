"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo, useState } from "react";
import { NAV } from "@/lib/nav";

export function DocsChrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return NAV;
    return NAV.map((section) => ({
      ...section,
      items: section.items.filter(
        (item) =>
          item.title.toLowerCase().includes(q) ||
          item.blurb.toLowerCase().includes(q),
      ),
    })).filter((section) => section.items.length > 0);
  }, [query]);

  return (
    <div className="shell">
      <div className="topbar">
        <button type="button" className="menu-btn" onClick={() => setOpen((v) => !v)}>
          {open ? "Close" : "Menu"}
        </button>
        <strong>Bangla ASR handbook</strong>
      </div>
      <aside className={`sidebar${open ? " open" : ""}`}>
        <Link href="/" className="brand" onClick={() => setOpen(false)}>
          <small>CSE499 · NSU</small>
          <strong>Bangla ASR handbook</strong>
        </Link>
        <input
          className="search"
          placeholder="Filter pages…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        {filtered.map((section) => (
          <div key={section.id} className="nav-section">
            <div className="nav-label">{section.label}</div>
            {section.items.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`nav-link${pathname === item.href ? " active" : ""}`}
                onClick={() => setOpen(false)}
              >
                {item.title}
              </Link>
            ))}
          </div>
        ))}
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
