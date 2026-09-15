/* ==========================================================================
   Kotha — the settings window's behaviour.

   THE CONTRACT, WHICH IS TWO CALLS
   --------------------------------
   The pill's contract is one-way and stays that way (see the head of
   pill.js). This window, like the first-run one, calls into Rust — and it is
   the only other place that does.

     invoke("settings_get")  -> { every value, and every list to draw it from }
     invoke("settings_set", { key, value })  -> ok, or a rejection in words

   `settings_get` answers once, on load, and the reply carries the OPTION
   LISTS as well as the values. That is the important half: the hotkeys, the
   text-output routes and the themes are all Rust constants, and a copy of any
   of them written into this file is a copy that drifts from what the app will
   accept. Nothing in app/ui/ knows what a paste route is.

   `settings_set` is per-change rather than a Save button, because every
   setting here is either read at the top of the next dictation or applied the
   instant it changes. There is nothing to batch and nothing to cancel.

   WHAT A REJECTION MEANS
   ----------------------
   Rust validates and Rust applies, so a rejection is a real outcome and not a
   bug: the hotkey you picked belongs to another application, or an
   environment variable has taken the setting over. Every one is a sentence
   meant to be read. When one arrives the radio is put BACK to what the app is
   actually using — a control left sitting on a value the app rejected is a
   window that lies about what it is doing.

   WHAT THIS FILE IS ALLOWED TO DO
   -------------------------------
   Build the option rows, call kothaTheme, and write the footer. Which row is
   tinted, how a group is greyed out, what the focus ring looks like — all of
   that is pill.css under `body.settings`, keyed off `:checked` and
   `:disabled`. If you find yourself setting a colour from here, put it back
   in the stylesheet.

   MOCK
   ----
   The bottom of this file notices there is no Tauri and serves the same
   payload from memory, so app/ui/settings.html opens in a browser, renders,
   and themes. Same trade as mock.js: it ships inside the app and stands down
   at runtime, which is cheaper than a build step that exists to delete it.
   ========================================================================== */

const status = document.getElementById("status");

/** How long a "Saved" lingers. Long enough to notice, short enough that the
    footer is empty again before the next change — see the note in pill.css
    about a page that permanently reads "Saved". */
const SAID = 2400;
let saidTimer = null;

function say(text, bad = false) {
  clearTimeout(saidTimer);
  status.textContent = text;
  status.classList.toggle("bad", bad);
  // An error stays until the next thing happens. It is the only message on
  // this page the user has to act on, and 2.4 seconds is not long enough to
  // read a sentence you were not expecting.
  if (!bad) saidTimer = setTimeout(() => (status.textContent = ""), SAID);
}

/** What the app is actually using, per group.

    Kept here rather than read back out of the DOM, because the DOM is where
    the *attempt* lives: by the time a rejection arrives the radio already
    shows what was clicked, which is exactly the value we need to undo. */
const current = {};

/** Markup-safe, for the one place a value reaches innerHTML.

    The hotkeys are Rust constants except for one case: `hotkey_choice` accepts
    anything Tauri can parse out of a hand-edited settings.json, and shows it
    alongside the offered four. That is a value from a file, so it is escaped
    before it becomes a <kbd>. Everything else on this page is set with
    textContent. */
const esc = (t) =>
  t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** A shortcut as keys: `Ctrl+Shift+Space` -> Ctrl + Shift + Space. */
const keys = (k) => k.split("+").map((x) => `<kbd>${esc(x)}</kbd>`).join(" + ");

/**
 * Draw one group of radios.
 *
 * `options` is a list of [value, label] pairs exactly as Rust sent it, so the
 * order on screen is the order in the Rust constant — which for the hotkeys
 * means the default is first, and that is not an accident.
 *
 * `html` says the labels carry markup. Only the hotkey group does, and only
 * because a key is drawn as <kbd>.
 */
function group(id, options, chosen, { disabled = false, html = false } = {}) {
  const box = document.getElementById(id);
  current[id] = chosen;

  box.replaceChildren(
    ...options.map(([value, label]) => {
      const row = document.createElement("label");
      row.className = "opt";

      const input = document.createElement("input");
      input.type = "radio";
      input.name = id;
      input.value = value;
      // The property, not the attribute: this is the live selection.
      input.checked = value === chosen;
      input.disabled = disabled;

      const name = document.createElement("span");
      name.className = "opt-name";
      name[html ? "innerHTML" : "textContent"] = label;

      row.append(input, name);
      return row;
    })
  );
}

/**
 * Save one change, and put the radio back if Rust would not have it.
 *
 * One listener per group, added once at the bottom of this file and never
 * removed — `current` carries the value across saves, so there is nothing to
 * rewire and no cloning. An earlier draft re-cloned the group after every
 * save and silently lost the selection: `cloneNode` copies the `checked`
 * ATTRIBUTE, and this sets the property.
 */
async function save(key, value) {
  const previous = current[key];
  if (value === previous) return;

  try {
    await set(key, value);
    current[key] = value;
    if (key === "theme") kothaTheme(value);
    // A key that bound is a key that is no longer unbound.
    if (key === "hotkey") document.getElementById("hotkey-unbound").hidden = true;
    say("Saved");
  } catch (message) {
    say(String(message), true);
    const back = document.querySelector(`input[name="${key}"][value="${previous}"]`);
    if (back) back.checked = true;
  }
}

function render(s) {
  kothaTheme(s.theme);

  group("hotkey", s.hotkeys.map((k) => [k, keys(k)]), s.hotkey, { html: true });
  const unbound = document.getElementById("hotkey-unbound");
  unbound.hidden = s.hotkeyBound;
  unbound.textContent =
    "Another application already has this key, so nothing is bound. " +
    "Pick a different one — until then, the tray icon's Dictate starts a dictation.";

  group("paste", s.pasteModes, s.paste, { disabled: s.pasteForced });
  document.getElementById("paste-forced").hidden = !s.pasteForced;
  document.getElementById("paste-forced").textContent =
    "KOTHA_PASTE is set in the environment, so it is deciding this. Unset it to choose here.";

  group("theme", s.themes, s.theme);
}

/* ------------------------------------------------------------------ Tauri */

let get, set;

if (window.__TAURI__) {
  const { invoke } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;

  get = () => invoke("settings_get");
  set = (key, value) => invoke("settings_set", { key, value });

  // Another window — or a hand-edited settings.json picked up on the next
  // show — can change the theme while this one is open. Cheap to follow.
  listen("kotha://theme", (e) => kothaTheme(e.payload));
} else {
  /* ----------------------------------------------------------------- mock */
  console.info("kotha: no backend, serving settings from memory");

  const state = {
    hotkey: "F9",
    hotkeys: ["F9", "Ctrl+Shift+Space", "Alt+Shift+D", "Ctrl+Alt+Space"],
    hotkeyBound: true,
    paste: "copy",
    pasteModes: [
      ["copy", "Clipboard only"],
      ["paste", "Paste at the cursor"],
      ["portal", "Paste at the cursor (portal)"],
    ],
    pasteForced: false,
    theme: "system",
    themes: [["system", "Match the system"], ["dark", "Dark"], ["light", "Light"]],
  };

  get = async () => state;
  set = async (key, value) => {
    // One refusal, so the rejection path is reachable without a build: this is
    // the shape of "another application already has that key".
    if (key === "hotkey" && value === "Ctrl+Alt+Space") {
      throw `${value} is already taken by another application. Still using ${state.hotkey}.`;
    }
    state[key] = value;
  };
}

/* One listener per group, on the container rather than on each radio: the
   rows are replaced whenever a group is drawn, and a listener on the box
   survives that. Added before the first render, so nothing has to be rewired
   afterwards. */
for (const id of ["hotkey", "paste", "theme"]) {
  document.getElementById(id).addEventListener("change", (e) => save(id, e.target.value));
}

get().then(render).catch((e) => say(`Could not read the settings: ${e}`, true));
