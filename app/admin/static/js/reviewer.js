/* Shared reviewer workflow actions (lists, call + applicant detail pages) */

function initReviewerWorkflow(callId) {
  window._reviewerCallId = callId;
}

function initTenantDetail(tenantId) {
  window._tenantDetailId = tenantId;
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

function toggleAllTenants(on) {
  document.querySelectorAll('.tenant-select').forEach((cb) => { cb.checked = on; });
}

function bulkMarkSelected() {
  const ids = Array.from(document.querySelectorAll('.tenant-select:checked')).map((cb) => cb.value);
  withBtnLoading(document.getElementById('bulkReviewBtn'), () => bulkMarkReviewed(ids), 'Saving\u2026');
}

async function blacklistTenant() {
  const tenantId = window._tenantDetailId;
  if (!tenantId) return;
  try {
    const ok = await confirmAction({
      title: 'Add to do-not-call list?',
      message: 'Future calls from this number will be declined automatically.',
      confirmLabel: 'Add to list',
      danger: true,
    });
    if (!ok) return;
    await apiFetch('/admin/api/tenants/' + tenantId + '/blacklist', { method: 'POST' });
    showMessage('Added to do-not-call list.');
    setTimeout(() => location.reload(), 900);
  } catch (e) { showMessage(e.message, true); }
}

async function saveTenant(e) {
  e.preventDefault();
  const tenantId = window._tenantDetailId;
  if (!tenantId) return;
  const petsVal = document.getElementById('hasPets').value;
  const data = {
    full_name: document.getElementById('fullName').value.trim(),
    contact_phone: document.getElementById('contactPhone').value.trim(),
    email: document.getElementById('email').value.trim(),
    monthly_income: parseInt(document.getElementById('income').value, 10) || 0,
    adults_count: parseInt(document.getElementById('adults').value, 10) || 0,
    children_count: parseInt(document.getElementById('children').value, 10) || 0,
    move_in_raw: document.getElementById('moveDate').value,
    move_timing: document.getElementById('moveTiming').value,
    current_residence: document.getElementById('currentResidence').value,
    residence_duration: document.getElementById('residenceDuration').value,
    move_reason: document.getElementById('moveReason').value,
    employer: document.getElementById('employer').value,
    employment_duration: document.getElementById('employmentDuration').value,
    general_notes: document.getElementById('generalNotes').value,
    notes: document.getElementById('notes').value,
    has_eviction: document.getElementById('hasEviction').value === 'true',
  };
  if (petsVal !== '') {
    data.has_pets = petsVal === 'true';
  }
  try {
    await apiFetch('/admin/api/tenants/' + tenantId, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    showMessage('Applicant updated successfully');
    setTimeout(() => location.reload(), 1500);
  } catch (err) {
    showMessage('Error: ' + err.message, true);
  }
}
