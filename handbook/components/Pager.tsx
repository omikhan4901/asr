import Link from "next/link";
import type { NavItem } from "@/lib/nav";

export function Pager({ prev, next }: { prev?: NavItem; next?: NavItem }) {
  return (
    <nav className="pager">
      {prev ? (
        <Link href={prev.href}>
          <small>Previous</small>
          {prev.title}
        </Link>
      ) : (
        <span />
      )}
      {next ? (
        <Link href={next.href} className="right">
          <small>Next</small>
          {next.title}
        </Link>
      ) : null}
    </nav>
  );
}
