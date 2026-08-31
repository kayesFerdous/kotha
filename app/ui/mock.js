/* ==========================================================================
   Kotha — the backend, faked, so the pill can be worked on in a browser.

   Open app/ui/index.html directly. No Rust, no model, no build step, no
   server. This file notices there is no Tauri around it and takes over,
   driving exactly the same two entry points the real app drives.

   That matters more than it looks: the real app is a ten-minute CTranslate2
   compile away, and a CSS change should not cost ten minutes. Anything you
   can see here is what you will get there.

     space        cycle idle → listening → thinking → done
     1 2 3 4      jump straight to a state
     m            toggle between fake speech and silence while listening

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
    return syllable * tremor * pause * 0.22 + Math.random() * 0.01;
  }

  setInterval(() => {
    if (document.documentElement.dataset.state === "listening") push(fakeLevel());
  }, 1000 / 30);

  /* The scripted loop, so the page is never just sitting there. Any keypress
     stops it and hands control over. */
  let scripted = true;
  const script = [
    ["listening", 4200],
    ["thinking", 2400],
    ["done", 1600],
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
    } else if (["1", "2", "3", "4"].includes(e.key)) {
      setState(STATES[Number(e.key) - 1]);
    } else if (e.key.toLowerCase() === "m") {
      speaking = !speaking;
    }
  });

  /* A checkerboard behind the pill, only in the browser. The Tauri window is
     transparent, so this stands in for "some arbitrary thing the user was
     looking at" — and it is the only way to see whether the blur, the border
     and the shadow actually hold up over light content. */
  document.body.style.background = `
    conic-gradient(from 90deg at 1px 1px, #0000 25%, #8883 0) 0 0/22px 22px,
    linear-gradient(120deg, #f3f4f6, #cbd5e1 45%, #475569 46%, #1e293b)`;
}
