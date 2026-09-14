/* ==========================================================================
   Kotha — the backend, faked, so the pill can be worked on in a browser.

   Open app/ui/index.html directly. No Rust, no model, no build step, no
   server. This file notices there is no Tauri around it and takes over,
   driving exactly the same two entry points the real app drives.

   That matters more than it looks: the real app is a ten-minute CTranslate2
   compile away, and a CSS change should not cost ten minutes. Anything you
   can see here is what you will get there.

     space        cycle idle → listening → thinking → done → error
     1 2 3 4 5    jump straight to a state
     m            toggle between fake speech and silence while listening,
                  which is the difference between a lit glyph and the five
                  resting bars an armed, silent pill shows
     t            cycle the theme: system → dark → light. Stands in for the
                  kotha://theme event, which is the only way the real pill
                  ever learns this — it cannot ask.

   It also runs a scripted loop on load so the pill is doing something the
   moment the file opens.

   This file DOES ship inside the app — Tauri embeds `app/ui/` wholesale and
   there is no bundler to strip it. It stands down at runtime instead, on the
   first line below. That is the cheaper trade: one branch here, against a
   build step that would exist only to delete one file.
   ========================================================================== */

if (window.__TAURI__) {
  console.info("kotha: real backend present, mock stood down");
} else {
  const { setState, push, STATES } = window.kotha;

  let speaking = true;
  let t = 0;

  /* Same three the settings window offers, in the same order. `system` first
     because it is the default and the one most likely to be wrong-looking on
     a given desktop. */
  const THEMES = ["system", "dark", "light"];
  let theme = 0;

  /* Fake RMS that reads like a voice rather than a sine wave: a slow syllable
     envelope, a faster tremor on top, and a noise floor. The point is only to
     exercise attack and release, so it does not need to be more honest than
     that — and it never pretends to be a real measurement anywhere else. */
  function fakeLevel() {
    if (!speaking) return Math.random() * 0.006;
    t += 1 / 30;
    const syllable = Math.max(0, Math.sin(t * 5.5) ** 2 - 0.08);
    const tremor = 0.75 + 0.25 * Math.sin(t * 21);
    const pause = t % 4.2 < 0.55 ? 0.06 : 1;      // breathe every few seconds
    /* 0.09 peak RMS is about -21 dBFS, which is ordinary speech in the middle
       of the range pill.js was calibrated against. It used to be 0.22 — a
       shout — and the strip sat pinned at full depth, so the browser showed a
       solid block where the app shows writing. */
    return syllable * tremor * pause * 0.09 + Math.random() * 0.01;
  }

  /* Rust meters the microphone whatever the pill is showing — the user may
     talk straight through a decode — so the mock feeds `thinking` too. */
  setInterval(() => {
    const state = document.documentElement.dataset.state;
    if (state === "listening" || state === "thinking") push(fakeLevel());
  }, 1000 / 30);

  /* The scripted loop, so the page is never just sitting there. Any keypress
     stops it and hands control over. */
  let scripted = true;
  const script = [
    ["listening", 4200],
    ["thinking", 2400],
    ["done", 1400],
    ["idle", 700],
    // The failure is in the loop because it is a state someone has to be able
    // to look at, and it is the one state a browser cannot provoke for real.
    ["error", 2600],
    ["idle", 900],
  ];
  (function run(i = 0) {
    if (!scripted) return;
    const [state, hold] = script[i % script.length];
    setState(state);
    setTimeout(() => run(i + 1), hold);
  })();

  addEventListener("keydown", (e) => {
    scripted = false;
    const at = STATES.indexOf(document.documentElement.dataset.state);
    if (e.key === " ") {
      e.preventDefault();
      setState(STATES[(at + 1) % STATES.length]);
    } else if (["1", "2", "3", "4", "5"].includes(e.key)) {
      setState(STATES[Number(e.key) - 1]);
    } else if (e.key.toLowerCase() === "m") {
      speaking = !speaking;
    } else if (e.key.toLowerCase() === "t") {
      theme = (theme + 1) % THEMES.length;
      kothaTheme(THEMES[theme]);
      console.info(`kotha: theme ${THEMES[theme]}`);
    }
  });

  /* A checkerboard behind the pill, only in the browser. The Tauri window is
     transparent, so this stands in for "some arbitrary thing the user was
     looking at" — and it is the only way to see whether the blur, the border
     and the shadow actually hold up over light content. */
  document.body.style.background = `
    conic-gradient(from 90deg at 1px 1px, #0000 25%, #8883 0) 0 0/22px 22px,
    linear-gradient(120deg, #f3f4f6, #cbd5e1 45%, #475569 46%, #1e293b)`;

  /* It runs from white to near-black on purpose, and it is the only way to
     check the thing both themes have to survive: the pill floats over content
     it does not control. A light pill has to stay legible on the dark end of
     that gradient and a dark one on the light end. Press `t` and drag the
     window. */
}
