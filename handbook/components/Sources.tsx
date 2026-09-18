export type Source = {
  title: string;
  href?: string;
  why: string;
};

export function Sources({ items }: { items: Source[] }) {
  return (
    <section className="sources">
      <h2>If you want to go deeper</h2>
      <p>
        You do not need these to understand the page. They are here when a
        sentence made you curious, or when you are writing related-work notes.
      </p>
      <ul>
        {items.map((item) => (
          <li key={item.title}>
            {item.href ? (
              <a href={item.href} target="_blank" rel="noreferrer">
                {item.title}
              </a>
            ) : (
              <strong>{item.title}</strong>
            )}
            {" — "}
            {item.why}
          </li>
        ))}
      </ul>
    </section>
  );
}
