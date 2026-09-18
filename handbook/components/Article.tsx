import type { ReactNode } from "react";
import { neighbors } from "@/lib/nav";
import { Pager } from "@/components/Pager";

export function Article({
  kicker,
  title,
  lede,
  path,
  children,
}: {
  kicker: string;
  title: string;
  lede: string;
  path: string;
  children: ReactNode;
}) {
  const { prev, next } = neighbors(path);
  return (
    <article className="article">
      <p className="kicker">{kicker}</p>
      <h1>{title}</h1>
      <p className="lede">{lede}</p>
      <div className="prose">{children}</div>
      <Pager prev={prev} next={next} />
    </article>
  );
}
