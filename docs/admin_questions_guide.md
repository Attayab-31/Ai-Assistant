# Admin Guide: Screening Questions

This guide is for non-technical admins who need to add, edit, reorder, disable, or score screening questions.

The Questions page controls what the voice agent asks during future calls. It also controls what information is saved, how answers are interpreted, which follow-up questions are skipped, and how applicants are scored.

## Where To Go

Open the admin dashboard, then go to:

`Settings` -> `Questions`

You will see every screening question in order. Each card shows:

- The question number and ID
- The answer type, such as `text`, `yes_no`, `date`, or `currency`
- The field or fields where the answer is saved
- Whether the question is active
- Whether read-back confirmation is enabled
- Whether the question is conditional
- Whether the question affects scoring

Use `Preview Flow` before saving major changes. It shows a sample path through the conversation.

## Important Rule

Changes affect new calls only. A call that is already in progress keeps the question setup it started with.

## Common Tasks

### Add A New Question

1. Click `Add question`.
2. Click `Edit` on the new card.
3. Replace `New screening question` with the wording you want the caller to hear.
4. Choose the best `Answer type`.
5. Set the `Primary field key`.
6. Add a clear `Field label`.
7. Add retry prompts if needed.
8. Decide whether it should affect scoring.
9. Click `Done`.
10. Click `Save All Changes`.
11. Use `Preview Flow` to confirm the question appears where expected.

Example:

- Question text: `Do you smoke or vape inside the home?`
- Answer type: `yes_no`
- Primary field key: `smokes_inside`
- Field label: `smokes or vapes inside`
- Understanding guide: `Extract a clear yes or no. If they only smoke outside, count that as no for inside the home.`

### Edit An Existing Question

1. Click `Edit` on the question card.
2. Update the wording, retry prompts, scoring, or condition.
3. Click `Done`.
4. Click `Save All Changes`.

For built-in questions, the primary field key is locked. This is intentional. Built-in fields such as `full_name`, `contact_phone`, `email`, `monthly_income`, and `has_eviction` connect to applicant profiles, emails, scoring, and call detail pages.

You can safely change the wording of a built-in question, but do not expect to rename its locked primary field.

### Reorder Questions

Use the up and down arrow buttons on each question card.

The voice agent asks active questions in this order, except for conditional questions that are skipped.

After reordering, click `Save All Changes`.

### Disable A Question Without Deleting It

1. Click `Edit`.
2. Uncheck `Active in flow`.
3. Click `Done`.
4. Click `Save All Changes`.

Inactive questions are not asked. Their fields are not requested from the caller. If a disabled question had scoring enabled, its points are removed from future scoring.

Use this when you may want the question back later.

### Delete A Question

1. Click `Delete`.
2. Confirm the warning.
3. Click `Save All Changes`.

Deleting removes the question from future calls. If the question had scoring enabled, its points are removed from scoring.

Be careful deleting questions that collect contact information:

- `full_name`
- `contact_phone`
- `email`

If no active question collects these fields, result emails and CRM exports may be missing applicant contact details.

### Reset To Defaults

Click `Reset to Defaults` only if you want to replace the whole question setup with the system default screening flow.

This is useful if the flow becomes confusing, but it will remove custom questions and custom wording.

## Field Keys And Labels

Each question saves the caller's answer into one or more fields.

### Primary Field Key

The primary field key is the main place the answer is saved.

Good custom field keys:

- `smokes_inside`
- `desired_bedrooms`
- `has_section8_voucher`
- `preferred_contact_time`
- `parking_needed`

Use lowercase letters, numbers, and underscores. Avoid spaces and punctuation.

### Field Label

The field label is the human-friendly meaning of the field. The voice agent uses it to understand and confirm answers.

Examples:

- Field key: `desired_bedrooms`
- Field label: `desired number of bedrooms`

- Field key: `has_section8_voucher`
- Field label: `housing voucher or Section 8`

### Additional Fields

Use additional fields when one question collects multiple pieces of information.

Example question:

`What type of pet do you have, and about how much does it weigh?`

Fields:

- Primary field key: `pet_type`
- Additional field: `pet_weight`
- Additional field: `pets_raw`

For most simple questions, one field is enough.

## Answer Types

Choose the answer type that best matches what the caller should say.

### `text`

Use for short open-ended answers.

Good for:

- Employer
- Current residence
- Preferred neighborhood

### `long_text`

Use when the caller may explain in a sentence or two.

Good for:

- Reason for moving
- Eviction circumstances
- Final notes

### `yes_no`

Use when the answer should be yes or no.

Good for:

- Pets
- Eviction history
- Smoking
- Parking needed

The system stores this as true or false.

### `number`

Use for counts.

Good for:

- Number of occupants
- Number of bedrooms
- Number of vehicles

### `currency`

Use for money amounts.

Good for:

- Monthly income
- Budget
- Deposit amount

For income questions, be specific in the question text, such as `monthly household income before taxes`.

### `date`

Use for a date or timeframe.

Good for:

- Move-in date
- Lease start date
- When the applicant wants to move

The system accepts exact dates and softer answers such as `next month`, but exact answers are better for scoring.

### `phone`

Use for phone numbers. Enable read-back confirmation for this.

### `email`

Use for email addresses. Enable read-back confirmation for this.

## Understanding Guide

The understanding guide is an optional note for the voice agent. It is not spoken to the caller.

Use it when the question needs extra interpretation.

Good examples:

- `If they say they have a voucher or Section 8, store yes. Vouchers are accepted and should not be treated negatively.`
- `If they say they smoke only outside, count smokes_inside as no.`
- `Accept approximate dates like next month, but ask once for a specific date if needed.`
- `For bedroom count, extract the number only.`

Keep it short. One or two sentences is enough.

Avoid instructions that conflict with the question. For example, do not ask a yes/no question but write a guide that expects a long explanation.

## Retry Prompts

Retry prompts are used when the caller does not answer clearly.

Use simple, friendly wording.

Good examples:

- First retry: `Could you say that another way?`
- Second retry: `Just a yes or no is perfect.`
- Third retry: `No problem, we can note this for follow-up if you are not sure.`

For phone and email questions, retry prompts should ask the caller to say it slowly or spell it.

## Read-Back Confirmation

Read-back confirmation makes the assistant repeat an answer and ask if it is correct.

Use it for high-risk fields:

- Name
- Phone number
- Email

Example:

The caller says their phone number. The assistant says:

`Let me read that back to make sure I have it right - 5 5 5 1 2 3 4 5 6 7. Is that correct?`

Do not enable read-back on every question. It slows the call down.

## Conditional Questions

Conditional questions are only asked when a previous answer matches a rule.

Example:

Ask pet details only if the caller has pets.

Setup:

- First question field: `has_pets`
- Follow-up question: `What type, breed, and approximate weight are they?`
- Conditional field: `has_pets`
- Operator: `eq`
- Value: `true`

Operators:

- `eq`: ask if the earlier field equals the value
- `ne`: ask if the earlier field does not equal the value
- `truthy`: ask if the earlier field has a positive/yes-like value
- `falsy`: ask if the earlier field has a no/empty-like value

Important: a conditional question must reference a field from an earlier question. If it references a later question, saving will fail.

Avoid making every active question conditional. At least one active question must be askable at the start of the call.

## Scoring

Scoring decides whether the applicant is `qualified`, `review`, or `unqualified`.

Only questions with scoring enabled count toward the score. The system normalizes the final score to 0-100 based on the total possible points from enabled scoring questions.

If you delete or disable a scored question, its points are removed from future scoring.

If no question has scoring enabled, applicants cannot be scored automatically. Calls will route to manual review.

### Scoring Rules

#### `any_answer`

Gives full points when any answer is captured.

Use for:

- Reason for moving
- Current residence
- Occupants count

#### `required_field`

Gives full points when the primary field is captured.

Use when the exact field must be present.

#### `yes_no`

Gives different points for yes and no.

Example for eviction:

- Points if Yes: `0`
- Points if No: `15`

You can also choose `Disqualify on Yes` or `Disqualify on No`.

Use disqualification carefully. A disqualifying answer immediately makes the applicant unqualified.

#### `numeric_range`

Scores based on a minimum and optional maximum.

Example:

- Question: `What is your monthly household income before taxes?`
- Minimum value: `3000`
- Max points: `35`

If the number is below the minimum, the applicant gets partial points and a review reason is recorded.

#### `date_within`

Scores based on how soon the date is.

Example:

- Question: `What move-in date are you hoping for?`
- Max days ahead: `120`
- Max points: `15`

If the move-in date is farther out, the applicant gets partial points and a reason is recorded.

### Recommended Scoring Practice

Keep scoring simple:

- Score only the questions that truly matter.
- Keep the total enabled max points around 100.
- Do not score contact questions such as name, phone, or email.
- Use manual review instead of automatic disqualification when a situation needs human judgment.

The page warns you if enabled scoring points go above 100. This is allowed, but simpler totals are easier to understand.

## Default Question Flow

The default flow is:

1. Full name
2. Phone number
3. Email address
4. Desired move-in date
5. Number of occupants
6. Pets
7. Pet details, only if pets are yes
8. Current residence
9. Residence duration
10. Reason for moving
11. Move timing
12. Eviction or landlord-tenant court filing
13. Eviction details, only if eviction is yes
14. Monthly household income before taxes
15. Employer or income source
16. Employment duration
17. Final rental, credit, or background notes

The default scoring is mostly on move-in timing, occupants, pets, residence, moving reason, eviction, and income. Contact fields are collected but not scored.

## Save Warnings

The page may warn you before saving.

Common warnings:

- Missing contact fields: no active question collects name, phone, or email.
- Scoring total above 100: the score will still work, but it may be harder to reason about.
- No question can start the call: every active question is conditional.
- No scoring enabled: applicants will go to manual review.

Warnings do not always block saving, but read them carefully.

Errors that block saving include:

- No questions at all
- No active questions
- Duplicate question IDs or states
- A question with no extract field
- Invalid answer type
- Conditional question references a field that appears later in the flow
- Built-in question primary field was changed

## Testing Changes

After making changes:

1. Click `Save All Changes`.
2. Click `Preview Flow`.
3. Start a test call from the test console if available.
4. Answer the new or edited question naturally.
5. Open the call detail page and confirm:
   - The transcript sounds right.
   - The field was saved.
   - The question row says Answered, Skipped, or Declined correctly.
   - The score and status make sense.

## Safe Editing Checklist

Before saving, check:

- At least one active question starts the call.
- Name, phone, and email are still collected unless you intentionally removed them.
- Custom field keys are lowercase with underscores.
- Conditional questions point to earlier fields.
- Read-back is enabled only for high-risk fields.
- Scoring is enabled only where it matters.
- The preview flow looks right.

## Examples

### Add A Parking Question

Question text:

`Will you need parking?`

Recommended setup:

- Answer type: `yes_no`
- Primary field key: `needs_parking`
- Field label: `needs parking`
- Scoring: disabled, unless parking is a qualification rule

### Add A Bedroom Count Question

Question text:

`How many bedrooms are you looking for?`

Recommended setup:

- Answer type: `number`
- Primary field key: `desired_bedrooms`
- Field label: `desired bedrooms`
- Understanding guide: `Extract the number of bedrooms only.`

### Add A Voucher Question

Question text:

`Will you be using a housing voucher or Section 8?`

Recommended setup:

- Answer type: `yes_no`
- Primary field key: `has_housing_voucher`
- Field label: `housing voucher or Section 8`
- Understanding guide: `Vouchers are accepted. Extract yes or no only and do not treat this negatively.`
- Scoring: usually disabled

### Add A Conditional Follow-Up

First question:

`Do you have pets?`

Field:

`has_pets`

Follow-up question:

`What type, breed, and approximate weight are they?`

Conditional setup on the follow-up:

- Field: `has_pets`
- Operator: `eq`
- Value: `true`

This follow-up is skipped when the caller says they do not have pets.

## Best Practices

- Keep questions short and conversational.
- Ask one thing at a time when possible.
- Use additional fields only when one question naturally collects multiple details.
- Do not turn every detail into a scored rule.
- Prefer manual review for sensitive or complex situations.
- Use `Active in flow` instead of deleting when you are unsure.
- Use `Reset to Defaults` only when you want to replace the full flow.

