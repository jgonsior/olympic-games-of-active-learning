// Initialize Mermaid.js for diagram rendering
// Wait for Mermaid library to be available
(function initMermaid(retryCount) {
  retryCount = retryCount || 0;
  const maxRetries = 50;  // 5 seconds total (50 * 100ms)
  
  if (typeof mermaid !== 'undefined') {
    // Mermaid is available, initialize it
    mermaid.initialize({
      startOnLoad: false,  // We'll manually trigger rendering
      theme: 'default'
    });
    
    // Function to render mermaid diagrams
    function renderMermaid() {
      if (typeof mermaid !== 'undefined') {
        // Find all mermaid divs and render them
        const mermaidElements = document.querySelectorAll('.mermaid');
        if (mermaidElements.length > 0) {
          mermaid.run({ querySelector: '.mermaid' });
        }
      }
    }
    
    // For Material for MkDocs with instant loading
    if (typeof document$ !== 'undefined') {
      document$.subscribe(renderMermaid);
    } else {
      // For standard page loads
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', renderMermaid);
      } else {
        // DOM already loaded
        renderMermaid();
      }
    }
  } else if (retryCount < maxRetries) {
    // Mermaid not yet loaded, retry after a short delay
    setTimeout(function() { initMermaid(retryCount + 1); }, 100);
  } else {
    // Max retries reached, log error but don't break the page
    console.warn('Mermaid library failed to load after ' + maxRetries + ' attempts');
  }
})();
