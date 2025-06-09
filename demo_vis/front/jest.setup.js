// jest.setup.js
import '@testing-library/jest-dom'

// Mock global fetch
global.fetch = jest.fn();

// You can add other global setup configurations here if needed.

// Example: Mock a global function or object if necessary for all tests
// global.someGlobalFunction = jest.fn();

// Suppress console.error and console.log for cleaner test output, if desired
// However, be cautious as this might hide important warnings or errors.
// You can enable this selectively if test output is too noisy.
/*
let originalError;
let originalLog;

beforeAll(() => {
  originalError = console.error;
  originalLog = console.log;
  console.error = (...args) => {
    if (/Warning: ReactDOM.render is no longer supported in React 18./.test(args[0])) {
      // Suppress this specific warning if it's unavoidable and understood
      return;
    }
    originalError.apply(console, args);
  };
  // console.log = jest.fn(); // Suppress all console.log
});

afterAll(() => {
  console.error = originalError;
  console.log = originalLog;
});
*/
