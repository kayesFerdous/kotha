/* ==========================================================================
   Kotha — the settings window's behaviour.

   THE CONTRACT, WHICH IS TWO CALLS
   --------------------------------
     invoke("settings_get")  -> { every value, and every list to draw it from }
     invoke("settings_set", { key, value })  -> ok, or a rejection in words

   The reply to settings_get carries the OPTION LISTS as well as the values:
   the hotkeys, finishes, paste routes, themes and login choices are Rust
   constants, and a
   copy of any of them here is a copy that drifts from what the app accepts.

   Every change is saved the moment it is made, and nothing says "Saved" —
   the control already shows the new value, and a toast repeating it is noise.
   What does get said is a refusal: Rust validates and Rust applies, so the
   hotkey you picked may belong to another application. Then the control goes
   BACK to what the app is actually using, and the row says why in red, until
   the next change. A control left on a value the app refused is a window that
   lies about what it is doing.

   WHAT EACH ROW SAYS
   ------------------
   One sentence, about the current choice only, rewritten when it changes —
   see SAYS. That is the whole of the page's explaining.

   MOCK
   ----
   The bottom of this file serves the same payload from memory when there is
   no Tauri, so settings.html opens in a browser. It ships inside the app and
   stands down at runtime.
   ========================================================================== */

const MAC = /Mac/.test(navigator.platform);

/** `Ctrl+Shift+Space` as a person would write it: ⌃⇧Space on a Mac,
    Ctrl + Shift + Space elsewhere. */
function keyName(k) {
  if (!MAC) return k.split("+").join(" + ");
  const sym = { Ctrl: "⌃", Control: "⌃", Alt: "⌥", Option: "⌥", Shift: "⇧", Cmd: "⌘", Super: "⌘" };
  return k.split("+").map((p) => sym[p] || p).join("");
}

/** The sentence under each row, for the value it currently has. `s` is the
    whole settings payload, because one row's sentence can name another's
    value — the finish row tells you which key ends a dictation. */
const SAYS = {
  hotkey: (v) => `Press ${keyName(v)} in any app to start dictating.`,
  finish: (v, s) =>
    ({
      pause: "Text appears each time you pause. Kotha stops after 3 seconds of quiet.",
      confirm: `Talk as long as you like. Press ✓ or ${keyName(s.hotkey)} to insert it all, or Esc to throw it away.`,
    })[v],
  paste: (v) =>
    ({
      paste: "Kotha types the text wherever your cursor is.",
      portal: "Types at your cursor through the desktop’s permission prompt. It asks once.",
      copy: `The text is copied. Paste it yourself with ${MAC ? "⌘V" : "Ctrl+V"}.`,
    })[v],
  autostart: (v) =>
    v === "on"
      ? "Kotha starts when you log in and waits in the tray."
      : "Kotha starts only when you open it.",
};

let s = {}; // what the app is actually using

/* -------------------------------------------------------------- drawing */

/** Radios inside labels, one per option. Both controls on this page are
    built from these, so arrow keys, focus and screen readers all work
    without being given anything by hand. */
function radios(name, options, chosen, disabled = false) {
  return options.map(([value, label]) => {
    const opt = document.createElement("label");
    const input = Object.assign(document.createElement("input"), {
      type: "radio", name, value, checked: value === chosen, disabled,
    });
    opt.append(input, label);
    return opt;
  });
}

/** A segmented switch: the radios side by side. */
function segmented(id, options, chosen) {
  document.getElementById(id).replaceChildren(...radios(id, options, chosen));
}

/** A dropdown: a <details> whose summary shows the choice and whose body is
    the radios stacked. See the head of settings.html for why not <select>. */
function menu(id, options, chosen, disabled) {
  const el = document.getElementById(id);
  el.open = false;
  el.classList.toggle("disabled", disabled);
  el.querySelector("summary").textContent = options.find(([v]) => v === chosen)?.[1] ?? chosen;
  el.querySelector(".list").replaceChildren(...radios(id, options, chosen, disabled));
}

/** Write a row's sentence: what its value does, or — `bad` — why it was refused. */
function say(key, text = SAYS[key]?.(s[key], s), bad = false) {
  const p = document.querySelector(`[data-key="${key}"] .says`);
  if (!p) return;
  p.textContent = text || "";
  p.classList.toggle("bad", bad);
}

function render() {
  kothaTheme(s.theme);

  // The bound key plus any hand-edited one already arrive merged in `hotkeys`.
  menu("hotkey", s.hotkeys.map((k) => [k, keyName(k)]), s.hotkey, false);
  segmented("finish", s.finishes, s.finish);
  menu("paste", s.pasteModes, s.paste, s.pasteForced);
  segmented("theme", s.themes, s.theme);
  segmented("autostart", s.autostarts, s.autostart);

  for (const key of ["hotkey", "finish", "paste", "autostart"]) say(key);
  if (!s.hotkeyBound) {
    say("hotkey", "Another app already uses this key, so it does nothing. Pick a different one.", true);
  }
  if (s.pasteForced) say("paste", "Set by KOTHA_PASTE in the environment. Unset it to choose here.", true);

  document.getElementById("about").textContent =
    `Kotha ${s.version} · Runs entirely on this computer.`;
}

/* --------------------------------------------------------------- saving */

async function save(key, value) {
  if (value === s[key]) return;
  try {
    await set(key, value);
    s[key] = value;
    if (key === "hotkey") s.hotkeyBound = true;
    render(); // closes the menu, and the finish sentence names the hotkey
  } catch (message) {
    render(); // back to what the app is using…
    say(key, String(message), true); // …and why
  }
}

/* ---------------------------------------------------------------- Tauri */

let get, set;

if (window.__TAURI__) {
  const { invoke } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;

  get = () => invoke("settings_get");
  set = (key, value) => invoke("settings_set", { key, value });

  // Another window, or a hand-edited settings.json, can change the theme.
  listen("kotha://theme", (e) => kothaTheme(e.payload));
} else {
  /* ----------------------------------------------------------------- mock */
  console.info("kotha: no backend, serving settings from memory");

  const state = {
    hotkey: "F9",
    hotkeys: ["F9", "Ctrl+Shift+Space", "Alt+Shift+D", "Ctrl+Alt+Space"],
    hotkeyBound: true,
    finish: "confirm",
    finishes: [["pause", "When I go quiet"], ["confirm", "When I press ✓"]],
    paste: "paste",
    pasteModes: [
      ["paste", "Type at cursor"],
      ["portal", "Type at cursor (portal)"],
      ["copy", "Copy to clipboard"],
    ],
    pasteForced: false,
    theme: "system",
    themes: [["system", "System"], ["light", "Light"], ["dark", "Dark"]],
    autostart: "off",
    autostarts: [["on", "On"], ["off", "Off"]],
    version: "0.2.0",
  };

  get = async () => structuredClone(state);
  set = async (key, value) => {
    // One refusal, so the red path can be looked at without a build.
    if (key === "hotkey" && value === "Ctrl+Alt+Space") {
      throw `Another app already uses ${keyName(value)}. Still using ${keyName(state.hotkey)}.`;
    }
    state[key] = value;
  };
}

/* One listener per control, on the container: the options inside are
   replaced on every render and a listener up here survives that. */
for (const key of ["hotkey", "finish", "paste", "theme", "autostart"]) {
  document.getElementById(key).addEventListener("change", (e) => save(key, e.target.value));
}

/* A dropdown closes on a click anywhere else, and on Esc — back to its
   summary, so the keyboard does not lose its place. */
addEventListener("click", (e) => {
  for (const m of document.querySelectorAll(".menu[open]")) if (!m.contains(e.target)) m.open = false;
});
addEventListener("keydown", (e) => {
  const m = e.key === "Escape" && document.querySelector(".menu[open]");
  if (m) { m.open = false; m.querySelector("summary").focus(); }
});
// A forced setting shows its value but does not open.
for (const m of document.querySelectorAll(".menu")) {
  m.querySelector("summary").addEventListener("click", (e) => {
    if (m.classList.contains("disabled")) e.preventDefault();
  });
  // Opening moves focus onto the chosen option, so arrows work at once.
  m.addEventListener("toggle", () => m.open && m.querySelector(":checked")?.focus());
}

get()
  .then((reply) => { s = reply; render(); })
  .catch((e) => {
    document.getElementById("about").textContent = `Could not read the settings: ${e}`;
  });
