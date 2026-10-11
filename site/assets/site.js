/* Small and framework-free: nav toggle, click-to-zoom images, lazily rendered Plotly charts. */
(function () {
  document.documentElement.classList.add('js');

  var nav = document.getElementById('site-nav');
  var toggle = nav && nav.querySelector('.nav-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  document.addEventListener('click', function (e) {
    var img = e.target.closest && e.target.closest('img.zoom');
    if (!img) return;
    var box = document.createElement('div');
    box.className = 'lightbox';
    var big = document.createElement('img');
    big.src = img.currentSrc || img.src;
    big.alt = img.alt;
    box.appendChild(big);
    var close = function () { box.remove(); document.removeEventListener('keydown', onKey); };
    var onKey = function (ev) { if (ev.key === 'Escape') close(); };
    box.addEventListener('click', close);
    document.addEventListener('keydown', onKey);
    document.body.appendChild(box);
  });

  // If a chart can't be drawn (CDN blocked, bad JSON), show the PNG kept in <noscript> instead.
  function fallback(fig) {
    fig.dataset.state = 'failed';
    var ns = fig.querySelector('noscript');
    if (!ns) return;
    var holder = document.createElement('div');
    holder.innerHTML = ns.textContent;
    var img = holder.firstChild;
    if (img) { img.className = 'zoom'; fig.insertBefore(img, fig.firstChild); }
  }

  function render(fig) {
    if (fig.dataset.state) return;
    fig.dataset.state = 'loading';
    fetch(fig.dataset.src).then(function (r) { return r.json(); }).then(function (spec) {
      var host = document.createElement('div');
      host.className = 'plot';
      // multi-panel charts stay readable on narrow screens by scrolling the figure sideways
      if (spec.layout && spec.layout.xaxis2) host.style.minWidth = '640px';
      fig.insertBefore(host, fig.firstChild);
      var draw = function () {
        if (!window.Plotly) { host.remove(); fallback(fig); return; }
        try {
          window.Plotly.newPlot(host, spec.data, spec.layout || {}, { responsive: true, displaylogo: false });
          fig.dataset.state = 'done';
        } catch (e) { host.remove(); fallback(fig); }
      };
      if (window.Plotly) draw(); else window.addEventListener('load', draw);
    }).catch(function () { fallback(fig); });
  }

  // The demo video starts only when the visitor hasn't asked for reduced motion; otherwise the poster stays.
  var demo = document.querySelector('video.demo');
  if (demo && !(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) {
    demo.play().catch(function () { /* autoplay blocked: the poster stays */ });
  }

  var figs = document.querySelectorAll('figure.chart[data-chart]');
  if (!figs.length) return;
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { io.unobserve(en.target); render(en.target); } });
    }, { rootMargin: '300px' });
    figs.forEach(function (f) { io.observe(f); });
  } else {
    figs.forEach(render);
  }
})();
