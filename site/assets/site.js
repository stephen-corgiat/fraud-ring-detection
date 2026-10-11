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

  function render(fig) {
    if (fig.dataset.state) return;
    fig.dataset.state = 'loading';
    fetch(fig.dataset.src).then(function (r) { return r.json(); }).then(function (spec) {
      var host = document.createElement('div');
      host.className = 'plot';
      fig.insertBefore(host, fig.firstChild);
      var draw = function () {
        window.Plotly.newPlot(host, spec.data, spec.layout || {}, { responsive: true, displaylogo: false });
        fig.dataset.state = 'done';
      };
      if (window.Plotly) draw(); else window.addEventListener('load', draw);
    }).catch(function () { fig.dataset.state = 'failed'; });
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
