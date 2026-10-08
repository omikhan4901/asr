// Labeling API (Vercel Node function). All data access goes through here with the
// Supabase secret key; the browser only ever sees the team passcode it typed.
//
// Env: SUPABASE_URL, SUPABASE_SECRET_KEY, TEAM_PASSCODE, ANNOTATORS ("Omi,Name2,Name3")
import { createClient } from "@supabase/supabase-js";
import { timingSafeEqual } from "node:crypto";

const db = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SECRET_KEY, {
  auth: { persistSession: false },
});
const NAMES = (process.env.ANNOTATORS || "").split(",").map((s) => s.trim()).filter(Boolean);
const TASKS = ["ref", "audit", "pair"];
const CLIP_TTL = 60 * 60; // signed audio URLs live one hour

function passOk(given) {
  const a = Buffer.from(String(given || ""));
  const b = Buffer.from(String(process.env.TEAM_PASSCODE || ""));
  return b.length > 0 && a.length === b.length && timingSafeEqual(a, b);
}

async function withClip(item) {
  const { meta, ...shown } = item; // meta stays on the server
  if (item.clip) {
    const { data, error } = await db.storage.from("clips").createSignedUrl(item.clip, CLIP_TTL);
    if (error) throw error;
    shown.clip_url = data.signedUrl;
  }
  return shown;
}

const actions = {
  // POST {passcode} -> the annotator names to pick from
  async login(req) {
    if (!passOk(req.body?.passcode)) return [401, { error: "Wrong passcode" }];
    return [200, { names: NAMES }];
  },

  // GET ?task=ref|audit|pair -> next item for this annotator, or {done: true}
  async next(req, who) {
    const task = req.query.task;
    if (!TASKS.includes(task)) return [400, { error: "Unknown task" }];
    const { data, error } = await db.rpc("claim_next", { p_annotator: who, p_task: task });
    if (error) throw error;
    if (!data?.length) return [200, { done: true }];
    return [200, { item: await withClip(data[0]) }];
  },

  // GET ?id= -> one of this annotator's items with their label (to go back and edit)
  async item(req, who) {
    const id = String(req.query.id || "");
    const { data: lab, error: e1 } = await db.from("labels").select("data,status")
      .eq("item_id", id).eq("annotator", who).maybeSingle();
    if (e1) throw e1;
    if (!lab) return [404, { error: "Not one of your items" }];
    const { data: item, error: e2 } = await db.from("items").select("*").eq("id", id).single();
    if (e2) throw e2;
    return [200, { item: await withClip(item), label: lab.data, status: lab.status }];
  },

  // POST {item_id, data} -> save (or update) this annotator's label
  async label(req, who) {
    const { item_id, data } = req.body || {};
    if (typeof item_id !== "string" || typeof data !== "object" || data === null) {
      return [400, { error: "item_id and data are required" }];
    }
    if (JSON.stringify(data).length > 20000) return [400, { error: "Label too large" }];
    // Only rows handed out by claim_next exist, so an update that touches nothing
    // means the item was never given to this annotator.
    const { data: rows, error } = await db.from("labels")
      .update({ status: "done", data, updated_at: new Date().toISOString() })
      .eq("item_id", item_id).eq("annotator", who).select("item_id");
    if (error) throw error;
    if (!rows.length) return [403, { error: "This item was not given to you" }];
    return [200, { ok: true }];
  },

  // GET ?task= -> this annotator's finished items, newest first
  async mine(req, who) {
    const task = req.query.task;
    if (!TASKS.includes(task)) return [400, { error: "Unknown task" }];
    const { data, error } = await db.from("labels").select("item_id,updated_at,items!inner(task)")
      .eq("annotator", who).eq("status", "done").eq("items.task", task)
      .order("updated_at", { ascending: false }).limit(300);
    if (error) throw error;
    return [200, { items: data.map((r) => ({ id: r.item_id, at: r.updated_at })) }];
  },

  // GET -> team progress per task and per annotator, and open disagreements per task
  async progress() {
    const { data, error } = await db.rpc("progress");
    if (error) throw error;
    const open = {};
    for (const task of TASKS) open[task] = (await disagreements(task)).filter((d) => !d.resolution).length;
    return [200, { rows: data, open }];
  },

  // GET ?task= -> shared items whose labels differ, with everyone's answer and the agreed one if set
  async mismatches(req) {
    const task = req.query.task;
    if (!TASKS.includes(task)) return [400, { error: "Unknown task" }];
    const list = await disagreements(task);
    const clips = list.map((d) => d.item.clip).filter(Boolean);
    const urls = {};
    if (clips.length) {
      const { data, error } = await db.storage.from("clips").createSignedUrls(clips, CLIP_TTL);
      if (error) throw error;
      for (const u of data) urls[u.path] = u.signedUrl;
    }
    const items = list.map(({ item: { meta, ...shown }, ...rest }) =>
      ({ ...rest, item: { ...shown, clip_url: urls[shown.clip] } }));
    return [200, { items }];
  },

  // POST {item_id, data} -> store the answer the team agreed on
  async resolve(req, who) {
    const { item_id, data } = req.body || {};
    if (typeof item_id !== "string" || typeof data !== "object" || data === null) {
      return [400, { error: "item_id and data are required" }];
    }
    const { error } = await db.from("resolutions")
      .upsert({ item_id, data, resolved_by: who, resolved_at: new Date().toISOString() });
    if (error) throw error;
    return [200, { ok: true }];
  },
};

// Answers that count as different for each task.
const norm = (s) => String(s || "").trim().replace(/\s+/g, " ");
const SAME = {
  ref: (a, b) => a.status === b.status && (a.status === "unusable" || norm(a.text) === norm(b.text)),
  audit: (a, b) => JSON.stringify(a.units) === JSON.stringify(b.units),
  pair: (a, b) => a.label === b.label,
};

// Items labeled by 2+ people (finished) whose answers are not all the same.
async function disagreements(task) {
  const { data: items, error: e1 } = await db.from("items").select("*").eq("task", task).gt("target", 1)
    .order("priority");
  if (e1) throw e1;
  if (!items.length) return [];
  const ids = items.map((i) => i.id);
  const { data: labs, error: e2 } = await db.from("labels").select("item_id,annotator,data")
    .in("item_id", ids).eq("status", "done");
  if (e2) throw e2;
  const { data: res, error: e3 } = await db.from("resolutions").select("item_id,data,resolved_by").in("item_id", ids);
  if (e3) throw e3;
  const byItem = {};
  for (const l of labs) (byItem[l.item_id] = byItem[l.item_id] || []).push(l);
  const resOf = Object.fromEntries(res.map((r) => [r.item_id, r]));
  return items
    .filter((i) => (byItem[i.id] || []).length >= 2)
    .filter((i) => byItem[i.id].some((l) => !SAME[task](l.data, byItem[i.id][0].data)))
    .map((i) => ({ item: i, answers: byItem[i.id].map((l) => ({ annotator: l.annotator, data: l.data })),
                   resolution: resOf[i.id] || null }));
}

export default async function handler(req, res) {
  const action = actions[req.query.action];
  if (!action) return res.status(404).json({ error: "Unknown action" });
  const isPost = ["login", "label", "resolve"].includes(req.query.action);
  if ((req.method === "POST") !== isPost) return res.status(405).json({ error: "Wrong method" });
  res.setHeader("Cache-Control", "no-store");
  let who = null;
  if (req.query.action !== "login") {
    who = String(req.headers["x-annotator"] || "");
    if (!passOk(req.headers["x-passcode"]) || !NAMES.includes(who)) {
      return res.status(401).json({ error: "Please sign in again" });
    }
  }
  try {
    const [status, body] = await action(req, who);
    res.status(status).json(body);
  } catch (err) {
    console.error(req.query.action, err);
    res.status(500).json({ error: "Server error, try again" });
  }
}
