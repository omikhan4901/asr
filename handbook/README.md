# Bangla ASR handbook (local textbook)

A Mintlify-style documentation site for the CSE499 group. No Mintlify
account. No hosting bill. Run it on your laptop.

## Start

You need [Node.js](https://nodejs.org/) 18 or newer (LTS is fine).

```bash
cd handbook
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Leave that terminal running. Edit a page under `app/` and the browser
reloads.

## What this is

A crash-course textbook plus a description of our research. It is not a
task board. Every crash-course page ends with sources if you want to
study further.

## Build (optional)

```bash
npm run build
npm start
```

That is only if you want a production server. For reading, `npm run dev`
is enough.
