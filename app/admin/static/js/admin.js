/* Shared admin UI utilities: API fetch + confirm modal */

async function apiFetch(url, opts) {
  opts = opts || {};
  const r = await fetch(url, opts);
  if (r.status === 401) {
    window.location.href = '/admin/login';
    throw new Error('Session expired');
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText || 'Request failed');
  }
  const ct = r.headers.get('content-type') || '';
  if (ct.includes('application/json')) return r.json().catch(() => ({}));
  return r;
}

function confirmAction(options) {
  const opts = options || {};
  return new Promise((resolve) => {
    const modal = document.getElementById('confirmModal');
    const titleEl = document.getElementById('confirmTitle');
    const msgEl = document.getElementById('confirmMessage');
    const inputEl = document.getElementById('confirmInput');
    const okBtn = document.getElementById('confirmOk');
    const cancelBtn = document.getElementById('confirmCancel');
    if (!modal || !okBtn || !cancelBtn) {
      resolve(window.confirm(opts.message || 'Are you sure?'));
      return;
    }
    titleEl.textContent = opts.title || 'Confirm';
    msgEl.textContent = opts.message || '';
    const typed = opts.typedConfirm || null;
    if (typed && inputEl) {
      inputEl.hidden = false;
      inputEl.value = '';
      inputEl.placeholder = 'Type ' + typed + ' to confirm';
      inputEl.setAttribute('aria-label', 'Type ' + typed + ' to confirm');
    } else if (inputEl) {
      inputEl.hidden = true;
      inputEl.value = '';
    }
    okBtn.className = opts.danger === false ? 'btn btn-primary' : 'btn btn-danger';
    okBtn.textContent = opts.confirmLabel || 'Confirm';
    cancelBtn.textContent = opts.cancelLabel || 'Cancel';
    modal.hidden = false;
    document.body.classList.add('modal-open');
    function cleanup(result) {
      modal.hidden = true;
      document.body.classList.remove('modal-open');
      okBtn.removeEventListener('click', onOk);
      cancelBtn.removeEventListener('click', onCancel);
      modal.querySelector('.confirm-backdrop')?.removeEventListener('click', onCancel);
      document.removeEventListener('keydown', onKey);
      resolve(result);
    }
    function onOk() {
      if (typed && inputEl && inputEl.value.trim() !== typed) {
        showMessage('Please type ' + typed + ' to confirm.', true);
        inputEl.focus();
        return;
      }
      cleanup(true);
    }
    function onCancel() { cleanup(false); }
    function onKey(e) {
      if (e.key === 'Escape') onCancel();
    }
    okBtn.addEventListener('click', onOk);
    cancelBtn.addEventListener('click', onCancel);
    modal.querySelector('.confirm-backdrop')?.addEventListener('click', onCancel);
    document.addEventListener('keydown', onKey);
    (typed && inputEl ? inputEl : cancelBtn).focus();
  });
}
