// demo_vis/front/__mocks__/react-hot-toast.js

// Mock the toast object and its methods
const toast = jest.fn();
toast.loading = jest.fn();
toast.success = jest.fn();
toast.error = jest.fn();
toast.dismiss = jest.fn();

// Mock the Toaster component (if it's ever rendered directly in tests, though usually not needed)
export const Toaster = () => <div data-testid="toaster-mock"></div>;

// Default export is the toast function itself
export default toast;
