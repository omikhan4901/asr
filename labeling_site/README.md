# Labeling site

Shared web page where the team labels data for the Phase 2 analyses:

1. **References**: listen to a chunk and correct its transcript (reference noise, E6).
2. **Error audit**: judge each difference between transcript and recognizer output
   (model wrong / transcript wrong / both wrong / same word spelled differently / dialect form / can't tell).
3. **Word pairs**: is a substituted word the same word spelled differently, a dialect form, or a different word?

Some items go to two or three people to measure agreement. Afterwards the team settles
disagreements together on the "Settle disagreements" screen; the agreed answer is stored
separately and the independent answers are kept.

Stack: static page (`public/index.html`) + one Vercel function (`api/[action].js`) +
Supabase (Postgres for items and labels, private storage bucket for audio clips).
The browser never sees a database key; every request needs the team passcode.

## Setup (once)

1. **Supabase project**: supabase.com → New project (region: Singapore). When it is ready:
   SQL Editor → New query → paste `schema.sql` → Run.
2. **Keys** (Project Settings → API Keys): copy the project URL and a *secret* key into `.env`
   in the repo root (never committed):
   ```
   SUPABASE_URL=https://<project>.supabase.co
   SUPABASE_SECRET_KEY=sb_secret_...
   ```
3. **Upload the tasks** (cuts ~1,300 clips, ~95 MB; safe to re-run):
   ```bash
   PYTHONPATH=src ./Scripts/python.exe analysis/labeling/build_tasks.py
   ```
4. **Deploy** from this folder:
   ```bash
   cd labeling_site
   vercel login
   vercel link
   vercel env add SUPABASE_URL production
   vercel env add SUPABASE_SECRET_KEY production
   vercel env add TEAM_PASSCODE production
   vercel env add ANNOTATORS production
   vercel --prod
   ```
   `ANNOTATORS` is a comma-separated list of names, e.g. `Omi,Nabigah,Name3`.

## Results

```bash
PYTHONPATH=src ./Scripts/python.exe analysis/labeling/import_labels.py
```
downloads a dated snapshot of all items, labels and agreed answers to `results/labels/` and
writes `summary.json` (agreement, reference noise per test set, audit shares by
selection-proof vs fixable errors, rule precision).
