// This must run *before* Mermaid tries to render anything.
// It sets securityLevel:'loose' so that <img> or raw HTML is allowed in node labels.
//
// NOTE: The exact syntax depends on your version of Mermaid.
// For newer releases (>= 10.x), you may need:
mermaid.initialize({
  startOnLoad: true,
  securityLevel: 'loose',    // allow raw HTML inside nodes
  theme: 'default'           // (or 'base', 'forest', etc.)
});

// If using an older Mermaid, you might need something like:
// mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
