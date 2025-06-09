// demo_vis/front/__mocks__/react-syntax-highlighter.js
import React from 'react';

// The component imported in page.tsx is `Prism as SyntaxHighlighter`
// So we need to export a component named Prism.
export const Prism = ({ children, language, style, ...props }) => {
  return (
    <pre data-testid="mock-syntax-highlighter" data-language={language} {...props}>
      <code>{children}</code>
    </pre>
  );
};

// If the default export was used like `import SyntaxHighlighter from 'react-syntax-highlighter'`,
// then we would do:
// export default Prism;

// If other named exports from this library are used, mock them as needed.
// For example, if styles were imported directly:
// export const vs = {}; // Mocking the style object if it's imported and used elsewhere
// However, in page.tsx, `vs` is imported from `react-syntax-highlighter/dist/esm/styles/prism`.
// So a specific mock for that path might be needed if this generic one doesn't cover it,
// or if the style import itself causes issues in tests.
// Jest's moduleNameMapper can also handle specific path mocks for styles if necessary.
// For now, this component mock should suffice for `Prism as SyntaxHighlighter`.
