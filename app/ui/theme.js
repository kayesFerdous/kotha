/* ==========================================================================
   Kotha — what "system" means, in one place.

   All three windows load this, from <head>, before their own script and
   before anything is painted. It is the only file in app/ui/ that is shared,
   and it is shared for one reason: "system" is a definition, and three copies
   of a definition is three chances for the pill to be dark while the settings
   window next to it is light.

   THE WHOLE JOB
   -------------
   The setting has three values — system, dark, light. The stylesheet has two
   palettes. This is the thing in between: it resolves `system` against the
   desktop and writes a concrete `data-theme="dark"` or `data-theme="light"`
   onto <html>. pill.css therefore never has to ask what the OS is doing, and
   the light palette is written once instead of once per way of reaching it.

     kothaTheme("system" | "dark" | "light")

   Call it whenever the answer changes. It is idempotent and it is cheap —
   one attribute write — so there is no need to check whether it changed
   first.

   WHY IT RUNS IN <head>
   ---------------------
   :root is the dark palette, so a page that resolved its theme after paint
   would show one frame of dark before going light. This runs during head
   parsing, when <html> already exists and nothing has been painted yet.

   THE DEFAULT IS `system`, NOT `dark`
   -----------------------------------
   It matches THEMES[0] in main.rs, which is what an install with no
   settings.json is actually using. Defaulting to dark here instead would mean
   every window on a light desktop started wrong and corrected itself a
   moment later, which is the flash this file exists to prevent.
   ========================================================================== */

(() => {
  const media = window.matchMedia("(prefers-color-scheme: light)");
  let wanted = "system";

  function apply() {
    document.documentElement.dataset.theme =
      wanted === "system" ? (media.matches ? "light" : "dark") : wanted;
  }

  /** Set the theme. `system` follows the desktop from here on. */
  window.kothaTheme = (name) => {
    wanted = ["system", "dark", "light"].includes(name) ? name : "system";
    apply();
  };

  /** Only while we are following it. An explicit dark or light outranks the
      desktop changing its mind, which is the entire point of choosing one. */
  media.addEventListener("change", () => {
    if (wanted === "system") apply();
  });

  apply();
})();
