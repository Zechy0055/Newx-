// demo_vis/front/__mocks__/marked-react.js
import React from 'react';

const MockMarkdown = ({ value, baseURL, renderer, gfm }) => {
  // Render the raw Markdown value or a placeholder for testing purposes.
  // This helps verify that the correct Markdown content is being passed to it.
  return (
    <div data-testid="mock-markdown-render" data-baseurl={baseURL} data-gfm={gfm}>
      <pre>{value}</pre>
    </div>
  );
};

export default MockMarkdown;
