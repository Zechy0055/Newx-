'use client'
import clsx from 'clsx';
import { BsStars } from "react-icons/bs";
import { FaWpforms } from "react-icons/fa6";
import { v4 as uuidv4 } from 'uuid';
import React, { useState, useEffect, useRef } from 'react'
import { FaHandsHelping } from "react-icons/fa";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vs } from "react-syntax-highlighter/dist/esm/styles/prism";
import { PiWheelchair } from "react-icons/pi";
import toast, { Toaster } from 'react-hot-toast';
import { FieldValues, useForm } from 'react-hook-form';
import { FaAnglesDown, FaAnglesUp } from "react-icons/fa6";
import Markdown from 'marked-react';

// Types for frontend logging
/**
 * Defines the severity levels for log messages.
 * @typedef {'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'} LogSeverity
 */
type LogSeverity = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';

/**
 * Interface for the payload sent to the backend logging endpoint.
 * @interface LogPayload
 * @property {LogSeverity} level - The severity level of the log.
 * @property {string} message - The main log message.
 * @property {string} [component] - Optional name of the component originating the log.
 * @property {string} [function] - Optional name of the function originating the log.
 * @property {Record<string, any>} [context] - Optional additional structured data.
 * @property {string} [frontend_timestamp] - Optional ISO string timestamp from the frontend, added by `logToBackend`.
 */
interface LogPayload {
  level: LogSeverity;
  message: string;
  component?: string;
  function?: string;
  context?: Record<string, any>;
  frontend_timestamp?: string;
}

/**
 * Sends a log message to the backend API.
 * Automatically adds a frontend timestamp to the payload.
 * Falls back to console.error if the backend request fails.
 * @async
 * @function logToBackend
 * @param {Omit<LogPayload, 'frontend_timestamp'>} payload - The log data to send, without the frontend_timestamp.
 */
const logToBackend = async (payload: Omit<LogPayload, 'frontend_timestamp'>) => {
  try {
    const augmentedPayload: LogPayload = {
      ...payload,
      frontend_timestamp: new Date().toISOString()
    };

    const response = await fetch('http://localhost:5000/api/log_frontend_event', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(augmentedPayload),
    });
    if (!response.ok) {
      // Fallback to console.error if backend logging fails
      console.error('Failed to send log to backend:', response.status, response.statusText, await response.text());
    }
  } catch (error) {
    console.error('Error sending log to backend:', error);
  }
};


interface Message {
  id: string
  category: string,
  is_open: boolean,
  // receive_time: Date // Example of a potential future field
};

/**
 * Represents a message specifically from the AutoCodeRover agent.
 * @interface AutoCodeRoverMessage
 * @extends Message
 * @property {string} title - The title of the message.
 * @property {string} message - The content of the message, often Markdown.
 * @property {'auto_code_rover'} category - The category identifier.
 */
interface AutoCodeRoverMessage extends Message {
  title: string,
  message: string,
  category: 'auto_code_rover'
};

/**
 * Represents a message from the Context Retrieval Agent.
 * @interface ContextRetrievalAgentMessage
 * @extends Message
 * @property {string} title - The title of the message.
 * @property {string} message - The content of the message, often Markdown.
 * @property {'context_retrieval_agent'} category - The category identifier.
 */
interface ContextRetrievalAgentMessage extends Message {
  title: string,
  message: string,
  category: 'context_retrieval_agent'
};

/**
 * Represents a message related to Patch Generation.
 * @interface PatchGenerationMessage
 * @extends Message
 * @property {string} title - The title of the message.
 * @property {string} message - The content of the message, often Markdown or a diff.
 * @property {'patch_generation'} category - The category identifier.
 */
interface PatchGenerationMessage extends Message {
  title: string,
  message: string,
  category: 'patch_generation'
};

/**
 * Represents information about the GitHub issue being processed.
 * @interface IssueInfo
 * @extends Message
 * @property {string} problem_statement - The problem statement of the issue.
 * @property {'issue_info'} category - The category identifier.
 */
interface IssueInfo extends Message {
  problem_statement: string,
  category: 'issue_info'
};

/**
 * A union type representing any possible message type in the application.
 * @typedef {AutoCodeRoverMessage | ContextRetrievalAgentMessage | IssueInfo | PatchGenerationMessage} AnyMessage
 */
type AnyMessage = AutoCodeRoverMessage | ContextRetrievalAgentMessage | IssueInfo | PatchGenerationMessage;

/**
 * Renders Markdown content with syntax highlighting for code blocks.
 * @function MarkdownRender
 * @param {object} props - The component's props.
 * @param {string} props.markdown - The Markdown string to render.
 * @param {string} [props.baseURL] - Optional base URL for relative links in the Markdown.
 * @returns {JSX.Element} The rendered Markdown content.
 */
function MarkdownRender({
  markdown,
  baseURL
}: {
  markdown: string,
  baseURL?: string
}) {
  // const uuid = uuidv4(); // uuid was generated but not used. Removed for now.
  const renderer = {
    code(snippet: string, lang: string) {
      return <div>
        {/* monoBlue  */}
        <SyntaxHighlighter language={lang} style={vs}>
          {snippet}
        </SyntaxHighlighter>
      </div>
    },
  };
  return (
    <article className='prose'>
      <Markdown
        value={markdown}
        baseURL={baseURL ? baseURL : ''}
        renderer={renderer}
        gfm
      />
    </article >
  )
}

const LoadingDiv = () => {
  return (
    <div
      className='bg-gray-200 bg-opacity-30 p-4 rounded-2xl flex space-x-3
          group hover:bg-slate-200
          '
    >
      <div className="flex-auto py-0.5 text-lg leading-normal text-gray-800">
        <div
          className="font-medium text-gray-900 flex items-center space-x-2 justify-around"
        >
          <span>Wating for more Agent search and actions ... </span>
        </div>
      </div>
    </div>
  );
};

/**
 * A simple loading indicator component.
 * @function LoadingDiv
 * @returns {JSX.Element} A div displaying a loading message.
 */
const LoadingDiv = () => {
  return (
    <div
      className='bg-gray-200 bg-opacity-30 p-4 rounded-2xl flex space-x-3
          group hover:bg-slate-200
          '
    >
      <div className="flex-auto py-0.5 text-lg leading-normal text-gray-800">
        <div
          className="font-medium text-gray-900 flex items-center space-x-2 justify-around"
        >
          <span>Wating for more Agent search and actions ... </span>
        </div>
      </div>
    </div>
  );
};

/**
 * @typedef MessageDivProps
 * @property {AutoCodeRoverMessage | ContextRetrievalAgentMessage | PatchGenerationMessage} message - The message object to display.
 * @property {(id: string, isOpen: boolean) => void} onToggleOpen - Callback function to toggle the open/closed state of the message.
 * @property {React.ReactNode} icon - The icon to display for the message.
 */

/**
 * Displays an individual agent message, allowing it to be collapsed or expanded.
 * This component is memoized for performance.
 * @function MessageDiv
 * @param {MessageDivProps} props - The component's props.
 * @returns {JSX.Element} The rendered message div.
 */
const MessageDiv = React.memo(({
  message,
  onToggleOpen,
  icon: Icon
}: {
  message: AutoCodeRoverMessage | ContextRetrievalAgentMessage | PatchGenerationMessage,
  onToggleOpen: (id: string, isOpen: boolean) => void,
  icon: React.ReactNode
}) => {
  logToBackend({ level: 'DEBUG', message: 'Rendering MessageDiv', component: 'MessageDiv', context: { messageId: message.id, isOpen: message.is_open } });
  return (
    <>
      {
        message.is_open &&
        <div className='bg-gray-200 bg-opacity-30 p-4 pr-12 rounded-2xl flex space-x-3'>
          <div>
            <div className="relative flex h-6 w-6 flex-none items-center justify-center bg-white">
              {Icon}
            </div>
          </div>
          <div className="flex-auto py-0.5 text-lg leading-normal text-gray-800">
            <button
              className="font-medium text-gray-900 flex items-center space-x-2 
                hover:bg-slate-200 px-2 rounded-md
              "
              onClick={() => onToggleOpen(message.id, false)} // Use onToggleOpen
            >
              <span>{message.title}</span>
              <FaAnglesUp className='h-3 w-3' />

            </button>
            <div
              className='mt-2'
            >
              <MarkdownRender markdown={message.message} />
            </div>
          </div>
          {/* <time dateTime='time' className="flex-none py-0.5 text-xs leading-5 text-gray-950">
            time
          </time> */}
        </div>
      }
      {
        !message.is_open &&
        <button
          className='bg-gray-200 bg-opacity-30 p-4 rounded-2xl flex space-x-3
          group hover:bg-slate-200
          '
          onClick={() => onToggleOpen(message.id, true)} // Use onToggleOpen
        >
          <div>
            <div className="relative flex h-6 w-6 flex-none items-center justify-center bg-white">
              {Icon}
            </div>
          </div>
          <div className="flex-auto py-0.5 text-lg leading-normal text-gray-800">
            <div
              className="font-medium text-gray-900 flex items-center space-x-2
              "
            >
              <span>{message.title}</span>
              <FaAnglesDown className='h-3 w-3' />
            </div>
          </div>
          {/* <time dateTime='time' className="flex-none py-0.5 text-xs leading-5 text-gray-950">
          time
        </time> */}
        </button>
      }

    </>
  );
});
MessageDiv.displayName = 'MessageDiv';

/**
 * The main application component for the AutoCodeRover frontend demo.
 * Manages the overall UI, state (form inputs, messages, loading state),
 * and interactions with the backend API to run AutoCodeRover on a GitHub issue
 * and display the agent's progress.
 * @export
 * @function App
 * @returns {JSX.Element} The main application layout and content.
 */
export default function App() {

  const {
    register,
    handleSubmit,
    setValue
  } = useForm<FieldValues>();
  const [messages, setMessage] = useState<Array<AnyMessage>>([]);
  const [problemStatement, setProblemStatement] = useState<string | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // state indecating the process: 
  // 0: before starting
  // 1: waiting for servering cloning the git issue
  // 2: communicating with server and show the agent actions
  // 3: done
  const [loadingState, setLoadingState] = useState<number>(0);
  // const [toastId, setToastId] = useState<string>(); // Unused state

  /**
   * Populates the form with demo/example data.
   * @function demo_form_callback
   */
  const demo_form_callback = () => {
    logToBackend({level: 'INFO', message: 'Demo form callback triggered', function: 'demo_form_callback'});
    setValue('repository_link', 'https://github.com/langchain-ai/langchain.git');
    setValue('commit_hash', 'cb6e5e5');
    setValue('issue_link', 'https://github.com/langchain-ai/langchain/issues/20453');
  };

  /**
   * Clears all form fields and resets the application state.
   * @function clear_form_callback
   */
  const clear_form_callback = () => {
    logToBackend({level: 'INFO', message: 'Clear form callback triggered', function: 'clear_form_callback'});
    setValue('repository_link', '');
    setValue('commit_hash', '');
    setValue('issue_link', '');
    setLoadingState(0);
    setProblemStatement('');
    setMessage([]);
    toast.dismiss();
  };

  /**
   * Handles the form submission to run AutoCodeRover on the provided issue.
   * It makes a POST request to the backend and processes the streamed response.
   * @async
   * @function onsubmmit_callback
   * @param {FieldValues} data - The form data containing repository link, commit hash, and issue link.
   */
  const onsubmmit_callback = async (data: FieldValues) => {
    logToBackend({level: 'INFO', message: 'Form submitted', function: 'onsubmit_callback', context: {formData: data}});
    const call_server = fetch('http://localhost:5000/api/run_github_issue', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })

    toast.loading('downloading issue and repository ...');

    call_server.then(response => {
      if (response.status !== 200) {
        return response.json().then(data => {
          toast.error(data.message);
          throw new Error(data.message);
        });
      }

      if (!response.body)
        throw Error('???');
      setLoadingState(1);
      const reader = response.body.getReader();

      /**
       * Reads and processes data from the response stream.
       * Updates message state and UI based on incoming data.
       * @async
       * @function read
       * @returns {Promise<void>}
       */
      function read(): Promise<void> {
        return reader.read().then(({ done, value }) => {
          if (done) {
            logToBackend({ level: 'INFO', message: 'Stream completed', function: 'onsubmit_callback.read' });
            toast.dismiss(); // Ensure latest toast is dismissed
            toast.success('Done !');
            setLoadingState(0); // Reset loading state
            return;
          }
          const text = new TextDecoder().decode(value);

          const newMessagesInChunk: AnyMessage[] = [];
          let issueInfoMessage: IssueInfo | null = null;

          text.split('</////json_end>').forEach(item => {
            if (item) {
              try {
                const parsedMessage: AnyMessage = JSON.parse(item);
                // Common processing for all messages from stream before adding to state
                parsedMessage.id = uuidv4();
                // is_open will be handled by the batch update logic for non-issue_info
                // For issue_info, is_open is not directly used in its rendering.
                logToBackend({level: 'DEBUG', message: 'Processing parsed message from stream', function: 'onsubmit_callback.read', context: {category: parsedMessage.category, id: parsedMessage.id}});

                if (parsedMessage.category === 'issue_info') {
                  issueInfoMessage = parsedMessage as IssueInfo;
                } else {
                  // New messages are initially considered open, batch logic will adjust previous last
                  parsedMessage.is_open = true;
                  newMessagesInChunk.push(parsedMessage);
                }
              } catch (e: any) { // Explicitly type e as any or Error
                // Keep console.error for immediate dev visibility, but also log to backend
                console.error("Failed to parse JSON message from stream:", item, e);
                logToBackend({
                  level: 'ERROR',
                  message: 'Failed to parse JSON message from stream',
                  function: 'onsubmit_callback.read',
                  context: { item: item, error: e.toString(), stack: e.stack }
                });
              }
            }
          });

          if (issueInfoMessage) {
            setIssueInfo(issueInfoMessage); // Call function to handle issue info
            setLoadingState(2);
            toast.dismiss(); // Dismiss previous loading toasts
            toast.success('Git issue clone successfully !');
            toast.loading('Agent running'); // New loading toast for agent
          }

          if (newMessagesInChunk.length > 0) {
            // Part 1: Modify setMessage for adding new messages (integrated with batching)
            setMessage(prevMessages => {
              let updatedMessages = [...prevMessages];
              if (updatedMessages.length > 0) {
                // Close the last message among the existing ones
                updatedMessages[updatedMessages.length - 1] = {
                  ...updatedMessages[updatedMessages.length - 1],
                  is_open: false,
                };
              }
              // Add all new messages from the chunk. They are already set to is_open: true.
              // If specific logic is needed for multiple messages within a chunk (e.g. only last of chunk is open)
              // it would be applied to newMessagesInChunk before this concatenation.
              // Current logic: all new messages in chunk are added as open, then next chunk will close the last one.
              return [...updatedMessages, ...newMessagesInChunk];
            });
          }
          return read(); // Continue reading
        });
      }
      return read();
    })
      .catch(error => {
        // Keep console.error for immediate dev visibility
        console.error('Error in onsubmit_callback fetch/read promise chain:', error);
        logToBackend({
          level: 'ERROR',
          message: 'Error in onsubmit_callback fetch/read promise chain',
          function: 'onsubmit_callback',
          context: { error: error.toString(), stack: error.stack }
        });
        toast.dismiss(); // Dismiss any active loading toasts
        toast.error(`Failed to fetch or process data: ${error.message}`);
        setLoadingState(0); // Reset loading state on error
      });
  };

  /**
   * Processes an IssueInfo message to update the problem statement display.
   * @function setIssueInfo
   * @param {IssueInfo} message - The IssueInfo message object.
   */
  const setIssueInfo = (message: IssueInfo) => {
    logToBackend({level: 'INFO', message: 'Setting issue info', function: 'setIssueInfo', context: {issueTitle: message.problem_statement.substring(0, 50)}});
    const s = message.problem_statement;
    const newlineIndex = s.indexOf('\n');
    const firstLine = newlineIndex !== -1 ? s.substring(0, newlineIndex) : s;
    const lastLine = newlineIndex !== -1 ? s.substring(newlineIndex + 1) : s;
    setProblemStatement(`## ${firstLine} \n ${lastLine}`)
  }

  /**
   * Memoized callback to handle toggling the open/closed state of a message.
   * Passed to MessageDiv components.
   * @function handleToggleOpen
   * @param {string} messageId - The ID of the message to toggle.
   * @param {boolean} newIsOpenState - The new desired 'is_open' state.
   */
  const handleToggleOpen = React.useCallback((messageId: string, newIsOpenState: boolean) => {
    logToBackend({level: 'DEBUG', message: 'Toggling message open state', function: 'handleToggleOpen', context: {messageId, newIsOpenState}});
    setMessage(prevMessages =>
      prevMessages.map(msg =>
        msg.id === messageId ? { ...msg, is_open: newIsOpenState } : msg
      )
    );
  }, []);

  return (
    <div className='w-full h-[100vh] flex flex-row'>
      <Toaster
        position="bottom-right"
        reverseOrder={false}
      />

      <div className='
        w-[30%] h-[100vh] lg:p-6 flex flex-col space-y-3
        bg-gray-100
      '>

        <label className='text-2xl'>
          Input the meta infomation of issue here
        </label>

        <form
          className='flex flex-col space-y-3'>

          <div className='flex items-center space-x-3'>
            <label className='w-32'>
              repository
            </label>
            <input
              {...register('repository_link')}
              type="text"
              className='p-2 border w-[80%] rounded-sm'
              placeholder='the link to the repository'
            />
          </div>

          <div className='flex items-center space-x-3'>
            <label className='w-32'>
              commit hash
            </label>
            <input
              {...register('commit_hash')}
              type="text"
              className='p-2 border w-[80%] rounded-sm'
              placeholder='the commit hash to checkout'
            />
          </div>

          <div className='flex items-center space-x-3'>
            <label className='w-32'>
              issue
            </label>
            <input
              {...register('issue_link')}
              type="text"
              className='p-2 border w-[80%] rounded-sm'
              placeholder='The link to the issue'
            />
          </div>

          <div className='flex justify-around text-sm'>

            <button
              className='w-[40%] m-1 py-4 p-2 border rounded-xl bg-slate-200 hover:bg-slate-300 hover:border-slate-50'
              onClick={(event: React.MouseEvent) => {
                event.preventDefault();
                clear_form_callback();
              }}>
              🧹 Clear all ! 🧹
            </button>

            <button
              className='w-[40%] m-1 py-4 p-2 border rounded-xl bg-slate-200 hover:bg-slate-300 hover:border-slate-50'
              type='submit'
              onClick={handleSubmit(onsubmmit_callback)}>
              🚀 Boot Now ! 🚀
            </button>
          </div>
        </form>

        <div className={clsx(
          'py-2 overflow-y-auto border-2 border-t-2 border-gray-400 border-t-gray-400 rounded-xl border-dashed w-full h-full',
          !problemStatement && 'flex items-center justify-around'
        )}>
          {
            !problemStatement && (
              <div className='flex flex-col items-center space-y-2'>
                <FaWpforms className='w-20 h-20' />
                <span className='font-semibold'>No Issues</span>
                <span className='text-gray-500'>
                  Try one example we prepare for you.
                </span>
                <button
                  className='m-1 py-4 p-2 border rounded-xl bg-slate-200 hover:bg-slate-300 hover:border-slate-50'
                  onClick={(event: React.MouseEvent) => {
                    event.preventDefault();
                    demo_form_callback();
                  }}>
                  💡 Try example ? 💡
                </button>
              </div>
            )
          }
          {
            problemStatement && (
              <div className='p-4 py-2'>
                <MarkdownRender
                  markdown={problemStatement}
                >
                </MarkdownRender>
              </div>
            )
          }
        </div>
      </div>

      <div className='w-[70%] h-[100vh] overflow-y-auto'>
        <ul
          role="list"
          className="
      space-y-6
      p-6
      "
        >
          {messages.map((message) => ( // Removed index as it's not used for key
            <li key={message.id} className="relative flex gap-x-4 overflow-x-auto">
              {
                !(message.category === 'issue_info') &&
                <MessageDiv
                  message={message as AutoCodeRoverMessage | ContextRetrievalAgentMessage | PatchGenerationMessage}
                  onToggleOpen={handleToggleOpen} // Pass memoized callback
                  icon={
                    <>
                      {
                        message.category === 'auto_code_rover' && (
                          <FaHandsHelping className='h-6 w-6 text-indigo-600 bg-gray-200 bg-opacity-30'
                            aria-hidden="true"
                          />
                        )
                      }
                      {
                        message.category === 'context_retrieval_agent' && (

                          <PiWheelchair className='h-6 w-6 text-indigo-600 bg-gray-200 bg-opacity-30'
                            aria-hidden="true"
                          />
                        )
                      }
                      {
                        message.category === 'patch_generation' && (
                          <BsStars className='h-6 w-6 text-indigo-600 bg-gray-200 bg-opacity-30'
                            aria-hidden="true"
                          />
                        )
                      }
                    </>
                  }
                />
              }

            </li>
          ))}
          {
            loadingState === 1 &&
            <div ref={messagesEndRef}>
              <LoadingDiv />
            </div>
          }
        </ul>
      </div>
    </div >
  )
}
