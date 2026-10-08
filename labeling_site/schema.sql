-- Labeling site database (Supabase / Postgres). Paste into the Supabase SQL editor once.
-- Safe to run again: it only creates what is missing.

create table if not exists items (
  id        text primary key,                 -- R0001 (reference), A0001 (error audit), P0001 (word pair)
  task      text not null check (task in ('ref', 'audit', 'pair')),
  priority  int  not null,                    -- lower is served first
  target    int  not null default 1,          -- how many annotators should label it (2-3 = agreement item)
  clip      text,                             -- path in the private storage bucket "clips"
  payload   jsonb not null,                   -- what the page shows
  meta      jsonb not null default '{}'::jsonb  -- hidden from annotators (rule type, source, model ...)
);
create index if not exists items_task_priority on items (task, priority, id);

create table if not exists labels (
  item_id    text not null references items (id) on delete cascade,
  annotator  text not null,
  status     text not null check (status in ('claimed', 'done')),
  data       jsonb,
  claimed_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (item_id, annotator)
);
create index if not exists labels_annotator on labels (annotator, status);

-- Agreed answers for items several people labeled differently (settled together).
-- The independent labels stay in "labels" untouched (needed for agreement / kappa).
create table if not exists resolutions (
  item_id     text primary key references items (id) on delete cascade,
  data        jsonb not null,
  resolved_by text not null,
  resolved_at timestamptz not null default now()
);
alter table resolutions enable row level security;

-- Only the server (secret key) may read or write: RLS on, no policies.
alter table items  enable row level security;
alter table labels enable row level security;

-- Hand the next item of a task to an annotator.
-- 1. An item this annotator claimed but has not finished comes back first (resume).
-- 2. Otherwise the highest-priority item that this annotator has not labeled and that
--    still needs labels (done + fresh claims by others < target). Claims older than
--    30 minutes are treated as abandoned. SKIP LOCKED keeps two people from getting
--    the same single-label item at the same moment.
create or replace function claim_next(p_annotator text, p_task text)
returns setof items
language plpgsql
as $$
declare
  v_id text;
begin
  select l.item_id into v_id
  from labels l join items i on i.id = l.item_id
  where l.annotator = p_annotator and l.status = 'claimed' and i.task = p_task
  order by i.priority, i.id
  limit 1;

  if v_id is null then
    select i.id into v_id
    from items i
    where i.task = p_task
      and not exists (select 1 from labels m where m.item_id = i.id and m.annotator = p_annotator)
      and (select count(*) from labels o
           where o.item_id = i.id
             and (o.status = 'done' or o.claimed_at > now() - interval '30 minutes')) < i.target
    order by i.priority, i.id
    limit 1
    for update skip locked;

    if v_id is null then
      return;
    end if;
    insert into labels (item_id, annotator, status) values (v_id, p_annotator, 'claimed')
    on conflict (item_id, annotator) do nothing;
  end if;

  return query select * from items where id = v_id;
end;
$$;

-- Progress per task: items, labels needed, labels done (team), and per annotator.
create or replace function progress()
returns table (task text, annotator text, done bigint, needed bigint, items bigint)
language sql
as $$
  select i.task, null::text,
         (select count(*) from labels l join items j on j.id = l.item_id where j.task = i.task and l.status = 'done'),
         sum(i.target), count(*)
  from items i group by i.task
  union all
  select i.task, l.annotator, count(*), null, null
  from labels l join items i on i.id = l.item_id
  where l.status = 'done'
  group by i.task, l.annotator;
$$;

-- Private bucket for the audio clips (served through short-lived signed URLs).
insert into storage.buckets (id, name, public) values ('clips', 'clips', false)
on conflict (id) do nothing;

-- Functions are callable only with the secret key (the site's server code).
revoke execute on function claim_next(text, text) from public, anon, authenticated;
revoke execute on function progress() from public, anon, authenticated;
