/* Shared reviewer workflow actions (call + applicant detail pages) */

function initReviewerWorkflow(callId) {
  window._reviewerCallId = callId;
}

async function patchCall(path, body) {
  const opts = { method: 'PATCH' };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  return apiFetch('/admin/api/calls/' + window._reviewerCallId + path, opts);
}

async function resendEmail() {
  try {
    await apiFetch('/admin/api/calls/' + window._reviewerCallId + '/resend-email', { method: 'POST' });
    showMessage('Result email queued — it will arrive shortly.');
  } catch (e) { showMessage(e.message, true); }
}

async function deleteCall() {
  try {
    const ok = await confirmAction({
      title: 'Delete this call?',
      message: 'This permanently removes the call, applicant profile, and any recording. This cannot be undone.',
      typedConfirm: 'DELETE',
      confirmLabel: 'Delete permanently',
      danger: true,
    });
    if (!ok) return;
    await apiFetch('/admin/api/calls/' + window._reviewerCallId, { method: 'DELETE' });
    showMessage('Call deleted. Redirecting…');
    setTimeout(() => { location.href = '/admin/calls'; }, 1200);
  } catch (e) { showMessage(e.message, true); }
}

async function saveQualification() {
  try {
    const sel = document.getElementById('qualOverride');
    const status = sel.value;
    const label = sel.options[sel.selectedIndex].text;
    const reasonEl = document.getElementById('qualOverrideReason');
    const reason = reasonEl ? reasonEl.value.trim() : '';
    const prior = sel.dataset.current || '';
    if (prior && prior !== status && !reason) {
      const proceed = await confirmAction({
        title: 'Save without a reason?',
        message: 'Adding a short reason helps if you need to explain this decision later. Save anyway?',
        confirmLabel: 'Save anyway',
        danger: false,
      });
      if (!proceed) return;
    }
    await patchCall('/qualification', { status, reason: reason || undefined });
    showMessage('Result updated to ' + label + '.');
    if (sel) sel.dataset.current = status;
  } catch (e) { showMessage(e.message, true); }
}

async function toggleReviewed() {
  try {
    const data = await patchCall('/review');
    const btn = document.getElementById('reviewBtn');
    const pill = document.getElementById('reviewStatePill');
    const reviewed = !!data.reviewed;
    if (btn) btn.textContent = reviewed ? 'Mark unreviewed' : 'Mark as reviewed';
    if (pill) {
      pill.className = 'review-chip ' + (reviewed ? 'reviewed' : 'needs-review');
      pill.innerHTML = reviewed
        ? '<span class="dot ok" aria-hidden="true"></span> Reviewed'
        : '<span class="dot warn" aria-hidden="true"></span> Needs your review';
    }
    showMessage(reviewed ? 'Marked as reviewed.' : 'Marked as not reviewed.');
  } catch (e) { showMessage(e.message, true); }
}

async function markTenantReviewed(tenantId, reviewed) {
  try {
    await apiFetch('/admin/api/tenants/' + tenantId + '/review', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewed }),
    });
    showMessage(reviewed ? 'Marked as reviewed.' : 'Marked as not reviewed.');
    setTimeout(() => location.reload(), 600);
  } catch (e) { showMessage(e.message, true); }
}

async function bulkMarkReviewed(tenantIds) {
  try {
    if (!tenantIds.length) {
      showMessage('Select at least one applicant.', true);
      return;
    }
    const ok = await confirmAction({
      title: 'Mark selected as reviewed?',
      message: 'Mark ' + tenantIds.length + ' applicant(s) as reviewed.',
      confirmLabel: 'Mark reviewed',
      danger: false,
    });
    if (!ok) return;
    await apiFetch('/admin/api/tenants/bulk-review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tenant_ids: tenantIds, reviewed: true }),
    });
    showMessage('Marked ' + tenantIds.length + ' applicant(s) as reviewed.');
    setTimeout(() => location.reload(), 700);
  } catch (e) { showMessage(e.message, true); }
}
