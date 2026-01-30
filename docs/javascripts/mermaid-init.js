// Initialize Mermaid.js for diagram rendering
// This works with Material for MkDocs instant loading feature
if (typeof document$ !== 'undefined') {
  // Material for MkDocs with instant loading
  document$.subscribe(function() {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default'
    });
    mermaid.contentLoaded();
  });
} else {
  // Fallback for standard page loads
  document.addEventListener('DOMContentLoaded', function() {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default'
    });
  });
}
