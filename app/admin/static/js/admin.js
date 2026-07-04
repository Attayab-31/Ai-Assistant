/* Shared admin UI utilities */

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

function toggleNav(open) {
  document.getElementById('sidebar').classList.toggle('open', open);
  document.getElementById('scrim').classList.toggle('show', open);
  const btn = document.getElementById('menuBtn');
  if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function showToast(message, kind) {
  const wrap = document.getElementById('toastWrap');
  const el = document.createElement('div');
  el.className = 'toast ' + (kind || '');
  el.textContent = message;
  wrap.appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 250); }, 3200);
}

async function adminLogout(e) {
  if (e) e.preventDefault();
  try {
    await fetch('/auth/logout', {
      method: 'POST',
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
    });
  } catch (err) { /* ignore */ }
  window.location.href = '/admin/login';
}

function showMessage(text, isError) {
  const cleaned = String(text || '').replace(/^[\u2713\u2717✓✗]\s*/, '').trim();
  showToast(cleaned || String(text), isError ? 'bad' : 'ok');
}

function setBtnLoading(btn, loading, text) {
  if (!btn) return;
  if (loading) {
    if (btn.dataset.loading === '1') return;
    btn.dataset.loading = '1';
    btn.dataset.originalHtml = btn.innerHTML;
    if (!btn.style.minWidth) btn.style.minWidth = btn.offsetWidth + 'px';
    btn.classList.add('is-loading');
    btn.setAttribute('aria-busy', 'true');
    btn.disabled = true;
    const label = text || btn.dataset.loadingText || 'Working\u2026';
    btn.innerHTML = '<span class="btn-spinner" aria-hidden="true"></span><span>' + label + '</span>';
  } else {
    if (btn.dataset.loading !== '1') return;
    btn.disabled = false;
    btn.classList.remove('is-loading');
    btn.removeAttribute('aria-busy');
    if (btn.dataset.originalHtml !== undefined) btn.innerHTML = btn.dataset.originalHtml;
    delete btn.dataset.loading;
    delete btn.dataset.originalHtml;
  }
}

async function withBtnLoading(btn, fn, text) {
  setBtnLoading(btn, true, text);
  try { return await fn(); }
  finally { setBtnLoading(btn, false); }
}

async function withFormLoading(event, fn, text) {
  const form = event.currentTarget || event.target;
  const btn = event.submitter
    || (form.querySelector && form.querySelector('button[type="submit"], button:not([type])'));
  setBtnLoading(btn, true, text);
  try { return await fn(event); }
  finally { setBtnLoading(btn, false); }
}

(function () {
  const origFetch = window.fetch;
  window.fetch = function (url, opts) {
    opts = opts || {};
    const headers = new Headers(opts.headers || {});
    if (!headers.has('X-Requested-With')) {
      headers.set('X-Requested-With', 'XMLHttpRequest');
    }
    opts.headers = headers;
    return origFetch(url, opts);
  };
})();

(function () {
  const prefetched = new Set();
  function isInternalNav(a) {
    if (!a || !a.href) return false;
    if (a.target === '_blank' || a.hasAttribute('download')) return false;
    if (a.getAttribute('href').startsWith('#')) return false;
    if (a.dataset.noPrefetch !== undefined) return false;
    try {
      const u = new URL(a.href, location.href);
      return u.origin === location.origin && u.pathname.startsWith('/admin/') && u.pathname !== location.pathname;
    } catch (_) { return false; }
  }
  function prefetch(a) {
    if (!isInternalNav(a) || prefetched.has(a.href)) return;
    prefetched.add(a.href);
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = a.href;
    link.as = 'document';
    document.head.appendChild(link);
  }
  document.addEventListener('pointerenter', (e) => {
    const a = e.target.closest && e.target.closest('a');
    if (a) prefetch(a);
  }, { capture: true, passive: true });
  document.addEventListener('touchstart', (e) => {
    const a = e.target.closest && e.target.closest('a');
    if (a) prefetch(a);
  }, { capture: true, passive: true });

  const bar = document.getElementById('navbar');
  let timer = null;
  function startBar() {
    if (!bar) return;
    bar.classList.add('go');
    bar.style.width = '18%';
    clearInterval(timer);
    let w = 18;
    timer = setInterval(() => { w = Math.min(w + (90 - w) * 0.18, 90); bar.style.width = w + '%'; }, 240);
  }
  document.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest && e.target.closest('a');
    if (isInternalNav(a)) startBar();
  }, true);
  window.addEventListener('pageshow', () => {
    if (bar) {
      clearInterval(timer);
      bar.style.width = '0';
      bar.classList.remove('go');
    }
  });
})();
