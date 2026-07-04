# Admin guide: Screening questions

This guide is for property managers and admins who configure what the phone assistant asks during screening calls.

The **Screening questions** page (`Settings` → `Questions`) controls:

- The conversation order and wording
- Which answers are saved to applicant profiles
- When follow-up questions are skipped (conditionals)
- How answers affect the qualification score

Open **How scoring works** from the Questions page (or **Help** in the top bar) anytime.

---

## What you see on the page

At the top:

- A summary: **active questions** vs **total** in your flow
- **Total scoring points enabled** (recommended max: 100)
- Warning banners if something needs attention before you save

Toolbar buttons:

| Button | What it does |
|--------|----------------|
| **Add question** (dropdown + button) | Adds a new question of the chosen type |
| **Reset to defaults** | Replaces the entire flow with the system default |
| **Preview conversation** | Shows a sample path through the flow |
| **Save all changes** | Writes your edits for **future calls only** |

Each question appears as a **conversation card** with:

- Question number and the wording callers hear
- Quick toggles (edit mode): **On**, **Required**, **Affects score**
- Badges: answer type, scoring, conditionals, read-back, etc.
- **↑ / ↓** reorder, **Edit**, **Delete**

---

## Important rules

1. **Changes affect new calls only.** A call already in progress keeps the question setup it started with.
2. **Save all changes** is required after edits. Quick toggles and reordering update the cards locally until you save.
3. Built-in questions have a **locked primary field key** (e.g. `full_name`, `monthly_income`). You can change wording; do not expect to rename those keys.
4. Disabling or deleting a scored question removes its points from **future** scoring.

---

## Add a question

1. Choose a type in the dropdown:
   - **Short answer** — names, short text
   - **Yes / no** — pets, smoking, eviction, etc.
   - **Open-ended** — longer explanations
   - **Number** — counts (occupants, bedrooms)
   - **Money amount** — income, deposits
   - **Date / timeframe** — move-in timing
2. Click **Add question**.
3. Click **Edit** on the new card.
4. Fill in **What should the assistant ask?**
5. Set **Answer type** if needed.
6. Optionally add **If the caller is unclear, say…** (retry prompt).
7. Click **Done**, then **Save all changes**.
8. Use **Preview conversation** to confirm placement.

---

## Edit a question

1. Click **Edit** on the card (or use quick toggles for On / Required / Affects score).
2. In the simple panel:
   - **What should the assistant ask?** — caller-facing wording
   - **Answer type**
   - **If the caller is unclear, say…** — first retry prompt
   - **When should this be asked?** — always, only after yes/no on a prior question, etc.
   - **On — ask during calls** — turn the question on or off
   - **Required** — caller must answer before moving on
   - **Affects applicant score** — include in qualification scoring
   - **Points for this question (0–100)** — when scoring is on
   - **Read answer back to confirm** — for name, phone, email
3. Expand **Advanced options** for field keys, extra fields, more retries, custom conditionals, and custom scoring rules.
4. Click **Done**, then **Save all changes**.

---

## Quick toggles (without opening Edit)

On each card you can flip:

- **On** — include in live calls
- **Required** — must answer
- **Affects score** — counts toward qualification

Still click **Save all changes** when finished.

---

## Reorder questions

Use **↑** and **↓** on each card. The assistant asks **active** questions in this order (skipping conditionals when rules say so).

Click **Save all changes** after reordering.

---

## Disable a question (without deleting)

Turn off **On** (quick toggle or in Edit), then **Save all changes**.

Inactive questions are not asked. Their scoring points are not counted for new calls.

---

## Delete a question

1. Click **Delete** on the card.
2. Confirm in the dialog.
3. Click **Save all changes**.

Be careful removing the only active question that collects **full name**, **contact phone**, or **email** — result emails and exports may lack contact details.

---

## Preview conversation

Click **Preview conversation** to see a **sample conversation path** using your current cards (including conditionals). Use this before saving large changes.

---

## When should this be asked? (conditionals)

In **Edit**, choose:

| Option | Meaning |
|--------|---------|
| **Always** | Every caller hears it |
| **Only if a previous answer was Yes** | After a yes/no question |
| **Only if a previous answer was No** | After a yes/no question |
| **Only if a previous question was answered** | Prior question has any answer |
| **Custom rule (in Advanced)** | Field, operator, value |

For yes/no / answered options, pick **Which previous question?** from the dropdown.

---

## Scoring (simple)

1. Turn on **Affects applicant score**.
2. Set **Points for this question (0–100)**.
3. In **Advanced options** → **Scoring rule**, pick a preset when needed:
   - Any answer earns full points
   - Required field must be captured
   - Yes/no variants (yes better, no better, disqualify on yes/no)
   - Minimum number to pass
   - Move-in date within N days

The page shows **Total scoring points enabled** at the top. Keeping near **100** total is recommended.

---

## Advanced options

Expand **Advanced options — field keys, extra data, special scoring rules** for:

- **Primary field key** — where the answer is stored (custom questions only; built-in keys are locked)
- **Field label (spoken)** — how the assistant refers to the field when confirming
- **Understanding guide** — hints for the AI parser (optional)
- **Additional fields** — capture multiple values from one question (e.g. pet type + weight)
- **Retry prompt (2nd)** and **(3rd)** — escalation if the caller is still unclear
- **Custom conditional rule** — field, operator (`eq`, `ne`, `truthy`, `falsy`), value
- **Custom scoring rule** — rule type, max points, pass config (yes/no points, min/max numeric, date window)

---

## Reset to defaults

**Reset to defaults** replaces the **entire** question list with the system default flow. Use only if the setup is badly broken. Custom questions and wording are lost until you recreate them.

---

## Built-in field keys (do not rename)

These connect to applicant profiles, emails, scoring, and call detail:

- `full_name`, `contact_phone`, `email`
- `monthly_income`, `employer`, `employment_duration`
- `move_in_date`, `move_timing`, `occupants_count`, `adults_count`, `children_count`
- `has_pets`, `pet_type`, `has_eviction`, and other standard screening fields

Custom questions should use new lowercase underscore keys, e.g. `smokes_inside`, `needs_parking`.

---

## After you save

- New calls use the updated flow and scoring.
- Existing applicant records are not re-scored automatically unless you edit their profile.
- Run a **test call** from **Try a screening call** on the Questions page to hear changes live.

---

## Tips

- Prefer plain, spoken wording — callers hear questions, they do not read them.
- Keep the flow short; use conditionals instead of asking everyone everything.
- Use **Preview conversation** after reordering or adding conditionals.
- Watch warning banners before **Save all changes** — they flag identity fields, scoring totals, and risky deletes.

---

## Related settings

- **General settings** — qualification thresholds (qualified / needs review / not qualified)
- **Email settings** — result email templates
- **Caller FAQs** — answers the assistant can give without leaving the screening flow

For day-to-day review after calls, use **Applicants** and **Calls** in the sidebar. Mark applicants **reviewed** when you are done — you can do this from the list or on the applicant profile.
