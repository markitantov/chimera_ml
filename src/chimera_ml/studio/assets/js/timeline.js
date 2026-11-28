
    () => {
      const videoRoot = document.querySelector('#studio-video');
      if (!videoRoot) return;
      const roots = [...document.querySelectorAll('#studio-timeline, #studio-xai-timeline')];
      const bindRoot = (root) => {
        const state = root.__chimeraStudioSync || (root.__chimeraStudioSync = {});
        const outputSelector = '[data-studio-output], [data-studio-xai-output]';
        const seriesSelector = '[data-studio-series], [data-studio-xai-series]';
        const applyVisibility = () => {
          const output = root.querySelector(outputSelector)?.value;
          const active = new Set([...root.querySelectorAll('[data-studio-source]:checked')].map(
            (box) => box.dataset.studioSource
          ));
          root.querySelectorAll(seriesSelector).forEach((series) => {
            const source = series.dataset.source;
            const sourceAllowed = !root.querySelector('[data-studio-source]') || active.has(source);
            series.style.display = series.dataset.output === output && sourceAllowed ? '' : 'none';
          });
          root.querySelectorAll('[data-studio-axis-output]').forEach((axis) => {
            axis.style.display = axis.dataset.studioAxisOutput === output ? '' : 'none';
          });
        };
        const bind = () => {
          const video = videoRoot.querySelector('video');
          const svg = root.querySelector('[data-studio-timeline], [data-studio-xai-timeline]');
          if (!video || !svg) return;
          if (state.video !== video) {
            state.video?.removeEventListener('timeupdate', state.updateCursor);
            state.video?.removeEventListener('seeking', state.updateCursor);
            state.video = video;
            state.updateCursor = () => {
              const currentSvg = state.svg || svg;
              const start = Number(currentSvg.dataset.start), end = Number(currentSvg.dataset.end);
              const duration = Number(video.duration);
              const limit = Number.isFinite(duration) && duration > 0 ? duration : end;
              const time = Math.max(0, Math.min(limit, Number(video.currentTime || 0)));
              const plotLeft = Number(currentSvg.dataset.plotLeft || 0);
              const plotWidth = Number(currentSvg.dataset.plotWidth || currentSvg.viewBox.baseVal.width);
              const progress = Math.max(0, Math.min(1, (time - start) / Math.max(end - start, 1e-9)));
              const x = plotLeft + progress * plotWidth;
              const cursor = currentSvg.querySelector('[data-studio-cursor], [data-studio-xai-cursor]');
              cursor?.setAttribute('x1', String(x)); cursor?.setAttribute('x2', String(x));
            };
            video.addEventListener('timeupdate', state.updateCursor);
            video.addEventListener('seeking', state.updateCursor);
          }
          if (state.svg === svg && state.clickVideo === video) {
            applyVisibility(); state.updateCursor(); return;
          }
          state.svg?.removeEventListener('click', state.clickSvg);
          state.svg = svg; state.clickVideo = video;
          state.clickSvg = (event) => {
            const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
            const matrix = svg.getScreenCTM(); if (!matrix) return;
            const transformed = point.matrixTransform(matrix.inverse());
            const start = Number(svg.dataset.start), end = Number(svg.dataset.end);
            const duration = Number(video.duration);
            const limit = Number.isFinite(duration) && duration > 0 ? duration : end;
            const plotLeft = Number(svg.dataset.plotLeft || 0);
            const plotWidth = Number(svg.dataset.plotWidth || svg.viewBox.baseVal.width);
            const progress = (transformed.x - plotLeft) / plotWidth;
            const target = start + progress * (end - start);
            video.currentTime = Math.max(0, Math.min(limit, target));
          };
          svg.addEventListener('click', state.clickSvg);
          root.querySelector(outputSelector)?.addEventListener('change', applyVisibility);
          root.querySelectorAll('[data-studio-source]').forEach((box) =>
            box.addEventListener('change', applyVisibility)
          );
          applyVisibility(); state.updateCursor();
        };
        bind();
        if (!state.observer) {
          state.observer = new MutationObserver(bind);
          state.observer.observe(root, {childList: true, subtree: true});
        }
      };
      roots.forEach(bindRoot);
    }
