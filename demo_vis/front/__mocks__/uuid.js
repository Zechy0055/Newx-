// demo_vis/front/__mocks__/uuid.js
let count = 0;
export const v4 = () => {
  count++;
  return `mock-uuid-${count}`;
};

// If there are other functions exported by uuid that are used, mock them too.
// For example, if v1 was used:
// export const v1 = () => 'mock-uuid-v1';
// Default export if the module is imported as `import uuid from 'uuid'`
// export default { v4 };
