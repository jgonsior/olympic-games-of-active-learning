// Initialize Mermaid.js for diagram rendering
// This works with Material for MkDocs instant loading feature
if (typeof document$ !== 'undefined') {
  // Material for MkDocs with instant loading
  document$.subscribe(function() {
    if (typeof mermaid !== 'undefined') {
      mermaid.initialize({
        startOnLoad: true,
        theme: 'default'
      });
    }
  });
} else {
  // Fallback for standard page loads
  document.addEventListener('DOMContentLoaded', function() {
    if (typeof mermaid !== 'undefined') {
      mermaid.initialize({
        startOnLoad: true,
        theme: 'default'
      });
    }
  });
}
