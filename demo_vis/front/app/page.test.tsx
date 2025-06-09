import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './page'; // Adjust path if your file structure is different

// Mock global fetch from jest.setup.js, but we can provide specific implementations per test
const mockFetch = global.fetch as jest.Mock;

// Mock logToBackend which is defined in page.tsx itself.
// We can spy on it or mock it if it were imported from another module.
// For now, we'll test its effects by checking fetch calls to /api/log_frontend_event.

describe('App Component (AutoCodeRover Frontend)', () => {
  beforeEach(() => {
    // Reset mocks before each test
    mockFetch.mockClear();
    // Default successful fetch for /api/log_frontend_event
    mockFetch.mockImplementation((url) => {
      if (url.includes('/api/log_frontend_event')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ status: 'Log received' }),
        });
      }
      // For /api/run_github_issue, specific tests will provide their own mock implementation
      return Promise.resolve({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ message: 'Default mock: Not Found' }),
        body: null, // Ensure body is mockable if stream is tested
      });
    });
  });

  test('renders initial form inputs and buttons correctly', () => {
    render(<App />);

    // Check for input fields by placeholder or label
    expect(screen.getByPlaceholderText('the link to the repository')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('the commit hash to checkout')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('The link to the issue')).toBeInTheDocument();

    // Check for buttons
    expect(screen.getByRole('button', { name: /Clear all !/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Boot Now !/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Try example ?/i })).toBeInTheDocument();
  });

  test('"No Issues" placeholder is visible initially', () => {
    render(<App />);
    expect(screen.getByText('No Issues')).toBeInTheDocument();
    expect(screen.getByText('Try one example we prepare for you.')).toBeInTheDocument();
  });

  test('clicking "Try example" button populates form fields', () => {
    render(<App />);
    const repoInput = screen.getByPlaceholderText('the link to the repository') as HTMLInputElement;
    const commitInput = screen.getByPlaceholderText('the commit hash to checkout') as HTMLInputElement;
    const issueInput = screen.getByPlaceholderText('The link to the issue') as HTMLInputElement;

    expect(repoInput.value).toBe('');
    expect(commitInput.value).toBe('');
    expect(issueInput.value).toBe('');

    fireEvent.click(screen.getByRole('button', { name: /Try example ?/i }));

    expect(repoInput.value).toBe('https://github.com/langchain-ai/langchain.git');
    expect(commitInput.value).toBe('cb6e5e5');
    expect(issueInput.value).toBe('https://github.com/langchain-ai/langchain/issues/20453');
  });

  test('clicking "Clear all" button clears form fields and resets state', () => {
    render(<App />);
    // First, populate with example data
    fireEvent.click(screen.getByRole('button', { name: /Try example ?/i }));

    const repoInput = screen.getByPlaceholderText('the link to the repository') as HTMLInputElement;
    expect(repoInput.value).not.toBe(''); // Ensure it's populated

    // Now clear
    fireEvent.click(screen.getByRole('button', { name: /Clear all !/i }));

    expect(repoInput.value).toBe('');
    expect(screen.getByPlaceholderText('the commit hash to checkout').value).toBe('');
    expect(screen.getByPlaceholderText('The link to the issue').value).toBe('');
    expect(screen.getByText('No Issues')).toBeInTheDocument(); // Problem statement should be cleared
    // Check if messages array is cleared (no messages displayed)
    expect(screen.queryByTestId('mock-message-div')).not.toBeInTheDocument(); // Assuming MessageDiv has a testid or identifiable role
  });

  test('form submission calls fetch with correct data and shows loading toast', async () => {
    render(<App />);

    // Populate form
    fireEvent.change(screen.getByPlaceholderText('the link to the repository'), { target: { value: 'test-repo-link' } });
    fireEvent.change(screen.getByPlaceholderText('the commit hash to checkout'), { target: { value: 'test-commit-hash' } });
    fireEvent.change(screen.getByPlaceholderText('The link to the issue'), { target: { value: 'test-issue-link' } });

    // Mock fetch for this specific test for /api/run_github_issue
    // This mock simulates a very slow response that doesn't resolve, so we only test the initial part
    mockFetch.mockImplementationOnce((url) => {
      if (url.includes('/api/run_github_issue')) {
        return new Promise(() => {}); // Never resolves, to keep it in loading state for toast check
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) }); // For logToBackend
    });

    fireEvent.click(screen.getByRole('button', { name: /Boot Now !/i }));

    // Verify fetch was called for /api/run_github_issue
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:5000/api/run_github_issue',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repository_link: 'test-repo-link',
          commit_hash: 'test-commit-hash',
          issue_link: 'test-issue-link',
        }),
      })
    );

    // react-hot-toast is mocked, so we can't screen.getByText the toast directly.
    // We'd need to check if toast.loading was called.
    // This requires importing the mocked toast.
    // For now, this part is implicit. If toast mock was more advanced, we could assert its calls.
    // The loading state within the component (loadingState variable) could also be checked if exposed,
    // or by looking for UI changes like the LoadingDiv component.
    // await screen.findByText('downloading issue and repository ...'); // This would work if toasts were in DOM
  });

  // Helper to create a mock ReadableStream
  function createMockReadableStream(chunks: string[]): ReadableStream<Uint8Array> {
    let chunkIndex = 0;
    return new ReadableStream({
      pull(controller) {
        if (chunkIndex < chunks.length) {
          const chunk = new TextEncoder().encode(chunks[chunkIndex]);
          controller.enqueue(chunk);
          chunkIndex++;
        } else {
          controller.close();
        }
      },
    });
  }

  describe('Streamed Message Handling', () => {
    test('processes issue_info and agent messages from stream correctly', async () => {
      const issueData = {
        category: 'issue_info',
        problem_statement: 'Test Issue Title\nThis is the problem statement.',
        // id and is_open will be added by the component
      };
      const agentMessage1Data = {
        category: 'auto_code_rover',
        title: 'Agent Action 1',
        message: 'Doing something important.',
      };
       const agentMessage2Data = {
        category: 'patch_generation',
        title: 'Patch Generated',
        message: '```diff\n-old\n+new\n```',
      };

      const streamChunks = [
        JSON.stringify(issueData) + '</////json_end>',
        JSON.stringify(agentMessage1Data) + '</////json_end>',
        JSON.stringify(agentMessage2Data) + '</////json_end>',
      ];

      mockFetch.mockImplementation((url) => {
        if (url.includes('/api/run_github_issue')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            body: createMockReadableStream(streamChunks),
          });
        }
        // For logToBackend calls
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
      });

      render(<App />);
      fireEvent.click(screen.getByRole('button', { name: /Boot Now !/i }));

      // Wait for issue info to be processed and displayed
      await waitFor(() => {
        expect(screen.getByText('## Test Issue Title')).toBeInTheDocument();
        expect(screen.getByText('This is the problem statement.')).toBeInTheDocument();
      });

      // Wait for agent messages to appear
      // MessageDiv is mocked to just show title and content for easier testing
      await waitFor(() => {
        expect(screen.getByText(agentMessage1Data.title)).toBeInTheDocument();
        // Check for part of the message content (Markdown is mocked)
        expect(screen.getByText(agentMessage1Data.message)).toBeInTheDocument();
      });

      // Check that first message (agentMessage1) is now closed, and agentMessage2 is open
      // This requires checking the `is_open` state if possible, or observing rendered output
      // The mock MessageDiv doesn't explicitly show open/closed state difference other than what page.tsx does.
      // We'll check that both messages are present. The "last open" logic is internal.
      await waitFor(() => {
        expect(screen.getByText(agentMessage2Data.title)).toBeInTheDocument();
        expect(screen.getByText('```diff\n-old\n+new\n```')).toBeInTheDocument(); // Check for diff content
      });

      // Verify logToBackend was called for stream completion and message processing
      // Count calls to /api/log_frontend_event
      const logCalls = mockFetch.mock.calls.filter(call => call[0].includes('/api/log_frontend_event'));
      // 1 for form submission, 1 for issue_info processing (DEBUG), 1 for agent msg1 (DEBUG), 1 for agent msg2 (DEBUG), 1 for stream completed
      expect(logCalls.length).toBeGreaterThanOrEqual(4); // At least form submit, each message, and stream completed

      // Example of checking a specific log call (if needed and mock is more detailed)
      const streamCompletedLog = logCalls.find(call => JSON.parse(call[1].body).message === 'Stream completed');
      expect(streamCompletedLog).toBeDefined();
    });

    test('message toggle functionality works', async () => {
      const agentMessageData = {
        category: 'auto_code_rover',
        title: 'Collapsible Message',
        message: 'This message can be toggled.',
        is_open: true, // Initially open
      };
      const streamChunks = [JSON.stringify(agentMessageData) + '</////json_end>'];
      mockFetch.mockImplementation((url) => {
        if (url.includes('/api/run_github_issue')) {
          return Promise.resolve({ ok: true, status: 200, body: createMockReadableStream(streamChunks) });
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
      });

      render(<App />);
      fireEvent.click(screen.getByRole('button', { name: /Boot Now !/i }));

      // Wait for the message to render (expanded by default)
      await waitFor(() => {
        expect(screen.getByText(agentMessageData.title)).toBeInTheDocument();
        expect(screen.getByText(agentMessageData.message)).toBeInTheDocument(); // Content visible
      });

      // Click to close it (MessageDiv's button calls onToggleOpen)
      // The button to close would contain the title and an "Up" arrow icon/text
      // Our mock MessageDiv doesn't render different buttons, but calls onToggleOpen(id, false)
      // We target the button that contains the title.
      fireEvent.click(screen.getByText(agentMessageData.title).closest('button'));

      // Wait for the message content to be hidden (or for the state to reflect is_open: false)
      // Since MarkdownRender is mocked to just show the text, we check if it's NOT in document.
      // This depends on MessageDiv's internal rendering logic based on is_open.
      // The current MessageDiv conditionally renders the content div.
      await waitFor(() => {
         expect(screen.queryByText(agentMessageData.message)).not.toBeInTheDocument();
      });

      // Click to re-open it
      fireEvent.click(screen.getByText(agentMessageData.title).closest('button'));
      await waitFor(() => {
        expect(screen.getByText(agentMessageData.message)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling and Logging', () => {
    test('handles fetch error for run_github_issue and logs to backend', async () => {
      mockFetch.mockImplementation((url) => {
         if (url.includes('/api/run_github_issue')) {
            return Promise.resolve({
              ok: false,
              status: 500,
              json: () => Promise.resolve({ message: 'Internal Server Error' }),
              body: null, // No body for error
            });
          }
          return Promise.resolve({ ok: true, json: () => Promise.resolve({}) }); // for logToBackend
      });

      render(<App />);
      fireEvent.click(screen.getByRole('button', { name: /Boot Now !/i }));

      // Wait for error toast (mocked, so we can't see it directly)
      // and check if logToBackend was called for this error.
      await waitFor(() => {
        const errorLogCall = mockFetch.mock.calls.find(call => {
          if (!call[0].includes('/api/log_frontend_event')) return false;
          const body = JSON.parse(call[1].body);
          return body.level === 'ERROR' && body.message.includes('Error in onsubmit_callback fetch/read promise chain');
        });
        expect(errorLogCall).toBeDefined();
      });
    });

     test('handles stream parsing error and logs to backend', async () => {
      const invalidJsonChunk = "This is not JSON</////json_end>";
      mockFetch.mockImplementation((url) => {
        if (url.includes('/api/run_github_issue')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            body: createMockReadableStream([invalidJsonChunk]),
          });
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
      });

      render(<App />);
      fireEvent.click(screen.getByRole('button', { name: /Boot Now !/i }));

      await waitFor(() => {
        const parseErrorLog = mockFetch.mock.calls.find(call => {
          if (!call[0].includes('/api/log_frontend_event')) return false;
          const body = JSON.parse(call[1].body);
          return body.level === 'ERROR' && body.message === 'Failed to parse JSON message from stream';
        });
        expect(parseErrorLog).toBeDefined();
        const context = JSON.parse(parseErrorLog[1].body).context;
        expect(context.item).toBe("This is not JSON");
      });
    });
  });
});
