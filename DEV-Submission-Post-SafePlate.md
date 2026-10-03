<!--
═══════════════════════════════════════════════════════════════════
  SUBMISSION-READY POST — all correct info filled in and verified.
  HOW TO USE:

  1. In the DEV editor (opened via "Submission Template"), select all
     and paste everything BELOW this comment block — it keeps the
     required first line and the auto-added tags.
  2. Already done for you (verified):
     [x] Demo link → https://karleeov-dev-hack.kimi.page (opened, works)
     [x] Deep links → URL-encoded, param names match app.js
     [x] GitHub repo → karleeov/Maya (code pushed & confirmed live)
     [x] Prize Categories removed (demo is on kimi.page, not Render —
         claiming Render without deploying there breaks the rules).
         Re-add it only if you deploy on Render — see the note at the
         bottom of this file.
  3. STILL TO DO BEFORE PUBLISHING (only you can do these):
     [ ] "Maya" → your real friend's name (Ctrl/Cmd+H, ~10 spots)
         — skip if her name really is Maya
     [ ] Check the allergies (peanut / tree nut / sesame) match hers
     [ ] Upload the 3 screenshots in the DEV editor (drag & drop) and
         replace the placeholder image paths
     [ ] Paste her real reaction in "What Maya Said"
     [ ] Delete "My Agent Session" if you're not using DevRelay
  4. Deadline: Oct 5, 2026, 6:59 AM UTC. Verify under "View Entries".
═══════════════════════════════════════════════════════════════════
-->

*This is a submission for the [Hacktoberfest Weekend Challenge: Build for a Friend](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01)*

## What I Built

My friend **Maya** has peanut, tree-nut, and sesame allergies. Every group dinner starts the same way: her squinting at ingredient lists, googling "does pesto have nuts" (it does — pine nuts), and quietly eating plain rice while everyone else shares satay.

So I built her **SafePlate** — an allergy-aware meal planner that reads every ingredient of every recipe *for her*:

- 🥜 **Her profile, not a generic one** — name, allergies, and how strict to be ("Sensitive" vs "Severe: flag every trace").
- 🍽️ **A verdict on every recipe** — *All clear*, *Worth a closer look*, or *Not safe for Maya* — with the exact ingredient that triggered it and why.
- 🕵️ **It catches the hidden ones** — pesto → pine nuts, Worcestershire sauce → anchovies, marzipan → almonds, tahini → sesame, green curry paste → shrimp paste.
- 🔁 **Every flag comes with a swap** — peanut sauce → sunflower seed butter, tahini → sunflower seed butter + lemon.
- 🏷️ **"Check anything" scanner** — paste any ingredient list (a sauce label, a snack bar) and it flags what's risky for *her*.
- 🗓️ **One-click week planner** — a full week of safe plates, saved locally on her device.

![SafePlate scanning every recipe for Maya's allergies](safeplate-screenshot-home.png)

## Demo

👉 **Live demo: https://karleeov-dev-hack.kimi.page**

It's fully shareable by URL, which makes for a fun try-it-yourself:

- `https://karleeov-dev-hack.kimi.page?friend=Maya&allergens=peanut,tree-nut,sesame` — pre-loaded profile
- `https://karleeov-dev-hack.kimi.page?scan=nutella%20spread,%20tahini%20drizzle,%20oat%20milk,%20eggplant,%20honey%20roasted%20peanuts,%20rice%20crackers` — auto-runs the scanner on a tricky list. Three of those are dangerous for Maya; three only *sound* dangerous.

![The scanner flags nutella, tahini and peanuts — and correctly clears oat milk, eggplant and rice crackers](safeplate-screenshot-scanner.png)

First visit downloads the model (~23 MB); after that it's cached and the whole app works offline.

![A week of safe plates, planned in one click](safeplate-screenshot-week.png)

## Code

{% github karleeov/Maya %}

No build step, no framework — `index.html`, `app.js`, `recipes.js`, `styles.css`. The phrasebook, veto table, and swap list in `recipes.js` are plain readable data, so adapting SafePlate for a different allergy set means editing a list, not a model.

## How I Built It

The core is **`all-MiniLM-L6-v2`**, an open-weight sentence-embedding model (Apache-2.0), running **entirely in the browser** through transformers.js + ONNX Runtime Web. Ingredients and allergen profiles become 384-dimensional vectors; cosine similarity is what knows *marzipan* lives suspiciously close to *almonds*.

But here's the honest part: **embeddings alone were too noisy to trust with someone's health.** I validated the scorer offline against the exact quantized ONNX build the browser ships, and the false-positive ceiling was brutal — `rice noodles → egg` scored 0.80 (because "egg noodles" exist), `tomato sauce → fish` hit 0.76, `butter → tree nuts` 0.74. Pure semantic search would have cried wolf about pancakes and stayed quiet about satay.

So SafePlate is a **hybrid**:

1. **A curated, word-boundary-aware phrasebook** per allergen carries precision — it knows `eggplant` ≠ `egg` and `buttermilk` ≠ `butter`.
2. **The embedding layer rides shotgun** — powering the "semantically close to X (matched *term*)" explanations, catching long-tail phrasings, and driving the caution tier.
3. **A small veto table** encodes domain truth the model can't know: coconut milk isn't dairy, oat milk is fine, peanut butter contains no butter.
4. **Severity modes shift the thresholds**, because "sensitive" and "anaphylactic" are not the same product. In Severe mode it still raises coconut as a tree-nut caution (the FDA lists it as one) — I'd rather explain a false alarm to Maya than apologize for a missed one.

The model downloads from the Hugging Face hub (with a mirror fallback), caches in the browser, and after that the app — model included — runs with the Wi-Fi off.

## Why Does Open Innovation Matter?

This project only makes sense open:

- **Privacy.** Allergies are health data. Maya's profile, scans, and meal plan never touch a server, because there *is* no server. Open weights are what let the AI come to the data instead of the data going to the AI.
- **Cost.** Actually zero — not freemium-zero. A closed API puts a meter between my friend and dinner.
- **Offline.** The restaurant with one bar of signal, the flight, the camping trip — allergies don't take a break, so the model lives on her device.
- **Auditable & swappable.** I could read the model card, pin the version, and test the exact artifact I shipped. Next month I can swap in a better open model without asking anyone's permission.
- **Hackable.** If her little cousin has a different allergy set, we edit a data list — not a billing plan.

The honest trade-off: a small open model is dumber than a frontier closed one. That's *why* the architecture is hybrid — the open model does what it's good at (semantics, explanations, long-tail matching) and boring curated data does what it's good at (precision). Open source didn't lower the ceiling; it changed where I put the walls.

## What Maya Said

> [PASTE HER REAL REACTION HERE — actually hand it over first. Bonus points, and honestly the best part of the build.]

## My Agent Session

[DELETE THIS SECTION if you're not using DevRelay. If you are: save your build session and embed it here with the `agent_session` tag shown on the challenge page, or link to it.]

<!-- Prize Categories: REMOVED — the demo is hosted on kimi.page, not on a
     partner platform, so no partner prize category applies. Only claim
     "Best Use of Render" if you genuinely deploy there. If you later host
     SafePlate on Render as a static site, re-add:
     ## Prize Categories
     - **Best Use of Render** — SafePlate's front end is hosted on Render. -->

<!-- Team Submissions: Please pick one member to publish the submission and credit teammates by listing their DEV usernames directly in the body of the post. -->

<!-- Thanks for participating! -->
