/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import { marked } from 'marked';

// --- LLM Abstraction Layer ---

/**
 * The response structure from any LLM provider.
 */
interface ProviderResponse {
  text?: string;
  json?: any;
  sources?: { title: string; uri: string }[];
}

/**
 * Interface for a multimodal LLM provider.
 */
interface LLMProvider {
  generateContent(prompt: string, useSearch: boolean, useJson: boolean): Promise<ProviderResponse>;
}

/**
 * An implementation of LLMProvider for the Google Gemini API.
 */
class GeminiProvider implements LLMProvider {
  private ai: GoogleGenAI;
  private modelName: string;
  private systemInstruction: string;

  constructor(apiKey: string, modelName: string, systemInstruction: string) {
    if (!apiKey) {
      throw new Error("API key is missing. Please ensure it's configured correctly.");
    }
    this.ai = new GoogleGenAI({ apiKey });
    this.modelName = modelName;
    this.systemInstruction = systemInstruction;
  }

  async generateContent(prompt: string, useSearch: boolean, useJson: boolean): Promise<ProviderResponse> {
    const config: any = {
      systemInstruction: this.systemInstruction,
    };

    if (useSearch) {
      config.tools = [{ googleSearch: {} }];
    }
    if (useJson) {
      config.responseMimeType = 'application/json';
    }

    const response = await this.ai.models.generateContent({
      model: this.modelName,
      contents: prompt,
      config: config,
    });
    
    const text = await Promise.resolve(response.text);

    const providerResponse: ProviderResponse = {};

    if (useJson) {
      providerResponse.json = JSON.parse(text);
    } else {
      providerResponse.text = text;
    }
    
    const groundingMetadata = response.candidates?.[0]?.groundingMetadata;
    if (useSearch && groundingMetadata?.groundingChunks) {
      const sources = groundingMetadata.groundingChunks
        .map(chunk => ({ title: chunk.web.title, uri: chunk.web.uri }));
      
      providerResponse.sources = sources;
    }

    return providerResponse;
  }
}

/**
 * Manages the available LLM providers and the currently active one.
 */
class LLMManager {
    private providers: Map<string, LLMProvider> = new Map();
    private currentProviderName: string;

    registerProvider(name: string, provider: LLMProvider) {
        this.providers.set(name, provider);
        if (!this.currentProviderName) {
            this.currentProviderName = name;
        }
    }

    // In a real app, this would be connected to a UI selector.
    setCurrentProvider(name: string) {
        if (!this.providers.has(name)) {
            throw new Error(`Provider "${name}" is not registered.`);
        }
        this.currentProviderName = name;
        console.log(`Switched to LLM provider: ${name}`);
    }

    getCurrentProvider(): LLMProvider {
        if (!this.currentProviderName || !this.providers.has(this.currentProviderName)) {
            throw new Error("No active LLM provider is set or registered.");
        }
        return this.providers.get(this.currentProviderName)!;
    }
}


// --- CONFIGURATION ---
const API_KEY = process.env.API_KEY;
const MODEL_NAME = 'gemini-2.5-flash';
const JOURNAL_CONTEXT_KEY = 'narrativeSculptorJournalContext';

// --- DOM ELEMENTS ---
const journalContextEl = document.getElementById('journal-context') as HTMLTextAreaElement;
const statusIndicatorEl = document.getElementById('status-indicator');
const chatHistoryEl = document.getElementById('chat-history');
const loadingIndicatorEl = document.getElementById('loading-indicator');
const confirmationButtonsEl = document.getElementById('confirmation-buttons');
const chatFormEl = document.getElementById('chat-form') as HTMLFormElement;
const chatInputEl = document.getElementById('chat-input') as HTMLTextAreaElement;
const sendButton = chatFormEl.querySelector('button');
const resetButtonEl = document.getElementById('reset-button');
const confirmYesButtonEl = document.getElementById('confirm-yes');
const confirmNoButtonEl = document.getElementById('confirm-no');
const voiceInputButtonEl = document.getElementById('voice-input-button');
const audioOutputButtonEl = document.getElementById('audio-output-button');


// --- STATE MANAGEMENT ---
const AppState = {
  INTAKE: 'INTAKE',
  REFINING: 'REFINING',
  DRAFTING: 'DRAFTING',
  CONFIRMING: 'CONFIRMING',
  ARCHIVING: 'ARCHIVING',
};

let currentState = AppState.INTAKE;
let chatHistory = [];
let rawThought = '';
let finalDraft = null;
let isAudioOutputEnabled = false;
let isRecording = false;

// --- SPEECH & AUDIO ---
const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
let recognition;
if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        isRecording = true;
        updateUI();
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        chatInputEl.value = transcript;
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        isRecording = false;
        updateUI();
    };
    
    recognition.onend = () => {
        isRecording = false;
        updateUI();
    };

} else {
    voiceInputButtonEl.style.display = 'none';
}


// --- INITIALIZATION ---
const SYSTEM_INSTRUCTION = `You are Narrative Sculptor, a supportive, Socratic, and fact-checking agent. Your goal is to help the user refine their unstructured thoughts into a single, cohesive, and factually-backed narrative entry for their journal.

**Your Persona & Rules:**
1.  **Interaction:** Be friendly, non-judgemental, highly focused, and analytical.
2.  **Context:** You will be given the user's previous journal entries. Always analyze the user's new input against this context to identify thematic overlaps, continuity, and progression of ideas.
3.  **Fact-Grounding:** For any claims requiring external validation, you MUST use the provided Google Search tool. Only use information from the search results. Do not invent facts. If a claim cannot be verified, flag it as speculative.
4.  **Questioning:** Only ask one, specific clarifying question at a time. Your questions should aim to define the core topic/thesis, identify the intended audience or purpose, fill logical gaps, or determine the desired tone or structure.
5.  **State Transitions:** During the 'Refining' phase, after you've gathered enough information, you must respond with the exact string "DRAFTING_READY" and nothing else. This will trigger the next step. Do not say this until you are confident you can produce a high-quality draft.`;

// Setup the LLM Manager and Providers
const llmManager = new LLMManager();
try {
    const geminiProvider = new GeminiProvider(API_KEY, MODEL_NAME, SYSTEM_INSTRUCTION);
    llmManager.registerProvider('gemini', geminiProvider);

    // To add another provider, you would implement the LLMProvider interface and register it here.
    // Example:
    // class OtherProvider implements LLMProvider { /* ... */ }
    // const otherProvider = new OtherProvider(OTHER_API_KEY);
    // llmManager.registerProvider('other', otherProvider);
    // llmManager.setCurrentProvider('other'); // To switch to it
} catch (error) {
    console.error("Failed to initialize LLM provider:", error);
    addMessage('agent', "Error: Could not initialize the AI service. Please check the API key and configuration.", "Error: Could not initialize the AI service.");
}


// --- UI UPDATE FUNCTIONS ---
function updateUI() {
  statusIndicatorEl.textContent = `State: ${currentState}`;
  chatHistoryEl.innerHTML = chatHistory.map(msg => `
    <div class="message ${msg.role}">
      ${msg.content}
    </div>
  `).join('');
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

  const isAgentTurn = currentState === AppState.DRAFTING || currentState === AppState.ARCHIVING;
  chatInputEl.disabled = isAgentTurn;
  sendButton.disabled = isAgentTurn;
  voiceInputButtonEl.classList.toggle('active', isRecording);
  audioOutputButtonEl.classList.toggle('active', isAudioOutputEnabled);

  if (isAgentTurn) {
    loadingIndicatorEl.classList.remove('hidden');
    chatFormEl.classList.add('hidden');
    confirmationButtonsEl.classList.add('hidden');
  } else {
    loadingIndicatorEl.classList.add('hidden');
    if (currentState === AppState.CONFIRMING) {
      chatFormEl.classList.add('hidden');
      confirmationButtonsEl.classList.remove('hidden');
    } else {
      chatFormEl.classList.remove('hidden');
      confirmationButtonsEl.classList.add('hidden');
    }
  }
}

function speak(text: string) {
    if (!isAudioOutputEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel(); // Stop any previous speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
}

function stripHtml(html: string): string {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    return doc.body.textContent || "";
}

function addMessage(role: 'user' | 'agent', content: string, speakableText?: string) {
  // Use marked to parse markdown content for the agent
  const parsedContent = role === 'agent' ? marked.parse(content) : `<p>${content}</p>`;
  chatHistory.push({ role, content: parsedContent });
  
  if (role === 'agent') {
    speak(speakableText || stripHtml(parsedContent as string));
  }
  
  updateUI();
}

function resetState() {
    currentState = AppState.INTAKE;
    chatHistory = [];
    rawThought = '';
    finalDraft = null;
    const defaultContext = getHardcodedDefaultContext();
    journalContextEl.value = defaultContext;
    localStorage.setItem(JOURNAL_CONTEXT_KEY, defaultContext);
    if(recognition && isRecording) recognition.stop();
    window.speechSynthesis.cancel();
    addMessage('agent', "Session reset. Please enter a new thought to begin.", "Session reset.");
    updateUI();
}

// --- CORE LOGIC ---
async function handleUserInput(event: Event) {
  event.preventDefault();
  const userInput = chatInputEl.value.trim();
  if (!userInput) return;

  addMessage('user', userInput);
  chatInputEl.value = '';

  if (currentState === AppState.INTAKE) {
    rawThought = userInput;
  }
  
  await runConversation();
}

async function runConversation() {
  currentState = currentState === AppState.INTAKE ? AppState.REFINING : currentState;
  updateUI();

  try {
    const prompt = buildPrompt();
    const useSearch = currentState === AppState.REFINING;

    const provider = llmManager.getCurrentProvider();
    const response = await provider.generateContent(prompt, useSearch, false);

    if (response.text.trim() === 'DRAFTING_READY') {
      await triggerDrafting();
    } else {
      let agentContent = response.text;
      if (response.sources && response.sources.length > 0) {
        const sourcesHtml = `
            <div class="citations">
                <h4>Sources:</h4>
                <ul>
                    ${response.sources.map(s => `<li><a href="${s.uri}" target="_blank">${s.title}</a></li>`).join('')}
                </ul>
            </div>
        `;
        agentContent += sourcesHtml;
      }
      addMessage('agent', agentContent);
    }
  } catch (error) {
    handleApiError(error, 'conversation');
  }
  
  updateUI();
}

async function triggerDrafting() {
    currentState = AppState.DRAFTING;
    addMessage('agent', 'Sufficient context gathered. Generating the final narrative draft...', 'Generating draft.');
    updateUI();

    try {
        const draftingPrompt = `Based on our entire conversation, generate the final narrative.
        - Original Thought: ${rawThought}
        - Journal Context: ${journalContextEl.value}
        - Conversation History: ${JSON.stringify(chatHistory.map(m => ({role: m.role, content: stripHtml(m.content)})))}
        
        Generate a JSON object with three keys: "title", "narrative", and "sources".
        - The "title" should be a concise and compelling heading that captures the main theme of the narrative entry.
        - The "narrative" is the final, well-structured journal entry. It must include citation markers like [1] corresponding to the sources array if any facts were grounded.
        - The "sources" is an array of objects, each with a "title" and "uri", if you used any external sources.`;

        const provider = llmManager.getCurrentProvider();
        const response = await provider.generateContent(draftingPrompt, false, true);
        
        finalDraft = response.json;

        // If the provider also returned sources (e.g. from a tool call within a JSON response),
        // let's prefer those as they are more reliable than what the model might generate in the JSON.
        if (response.sources && response.sources.length > 0) {
            finalDraft.sources = response.sources;
        }
        
        currentState = AppState.CONFIRMING;
        
        const confirmationMessage = `
            <p>I've generated a draft for you. Are you satisfied with this narrative and title?</p>
            <div class="archived-entry">
                <h3>${finalDraft.title}</h3>
                <div class="narrative">${marked.parse(finalDraft.narrative)}</div>
                ${finalDraft.sources && finalDraft.sources.length > 0 ? `
                <div class="citations">
                    <h4>Sources:</h4>
                    <ul>
                        ${finalDraft.sources.map(s => `<li><a href="${s.uri}" target="_blank">${s.title}</a></li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
        `;
        const speakableText = `I've generated a draft for you. Are you satisfied? The title is "${finalDraft.title}". The narrative is: ${finalDraft.narrative}`;
        addMessage('agent', confirmationMessage, speakableText);

    } catch (error) {
        handleApiError(error, 'drafting');
    }
    updateUI();
}


function handleConfirmation(approved: boolean) {
    if (approved) {
        currentState = AppState.ARCHIVING;
        updateUI();

        const timestamp = new Date().toLocaleString([], {
            year: 'numeric', month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit', second: '2-digit',
            hour12: false
        }).replace(',', '');
        
        const sourcesText = finalDraft.sources && finalDraft.sources.length > 0 
            ? `\n\n### Sources\n${finalDraft.sources.map(s => `- [${s.title}](${s.uri})`).join('\n')}` 
            : '';
            
        const markdownToExport = `# ${finalDraft.title}\n\n*Logged: ${timestamp}*\n\n${finalDraft.narrative}${sourcesText}`;
        const sanitizedMarkdown = markdownToExport.replace(/"/g, '&quot;');
        
        const formattedEntryHTML = `
            <div class="archived-entry">
                <h3>${finalDraft.title}</h3>
                <p class="timestamp">--- Logged: ${timestamp} ---</p>
                <div class="narrative">${marked.parse(finalDraft.narrative)}</div>
                ${finalDraft.sources && finalDraft.sources.length > 0 ? `
                <div class="citations">
                    <h4>Sources:</h4>
                    <ul>
                        ${finalDraft.sources.map(s => `<li><a href="${s.uri}" target="_blank">${s.title}</a></li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
        `;
        
        const exportActionsHTML = `
            <div class="export-actions">
                <button data-action="copy" data-content="${sanitizedMarkdown}">Copy to Clipboard</button>
                <button data-action="download" data-content="${sanitizedMarkdown}" data-title="${finalDraft.title.replace(/"/g, '&quot;')}">Download as Markdown</button>
            </div>
        `;
        
        addMessage(
            'agent', 
            `<p>Excellent! Your entry is ready. You can now copy it or download it to add to your Google Doc.</p>${formattedEntryHTML}${exportActionsHTML}<p>You can start a new thought below.</p>`, 
            'Excellent! The entry has been archived. You can start a new thought.'
        );
        
        // Simulate updating context
        const newEntryText = `Title: ${finalDraft.title}\nTimestamp: ${timestamp}\nNarrative: ${finalDraft.narrative.substring(0, 150)}...\n---\n\n`;
        journalContextEl.value = newEntryText + journalContextEl.value;
        localStorage.setItem(JOURNAL_CONTEXT_KEY, journalContextEl.value);

        currentState = AppState.INTAKE;
    } else {
        addMessage('agent', "No problem. Let's discard this draft. Please enter a new thought to start over.", "Draft discarded. Please start over.");
        resetState();
    }
    updateUI();
}

/**
 * Handles API errors, displaying a user-friendly message and managing state.
 * @param error The error object caught.
 * @param context The context in which the error occurred ('conversation' or 'drafting').
 */
function handleApiError(error: any, context: 'conversation' | 'drafting') {
    console.error(`Error during ${context}:`, error);

    let userMessage = `Sorry, I encountered a technical issue while trying to ${context === 'conversation' ? 'process your thought' : 'draft the narrative'}.`;
    let speakableMessage = "Sorry, I encountered a technical issue.";

    // Attempt to parse a more specific error message from the Gemini API error structure.
    if (error && error.message) {
        if (error.message.includes('API key not valid')) {
            userMessage = "There seems to be an issue with the API configuration. The provided API key is invalid. Please contact the administrator to resolve this.";
            speakableMessage = "There is an API configuration error.";
        } else if (error.message.toLowerCase().includes('rate limit')) {
            userMessage = "The service is currently experiencing high traffic and your request could not be completed. Please wait a moment and try again.";
            speakableMessage = "The service is busy. Please try again later.";
        } else if (error.message.toLowerCase().includes('timed out')) {
            userMessage = "The request timed out. This might be a temporary network issue. Please try sending your message again.";
            speakableMessage = "The request timed out. Please try again.";
        } else {
             // For other known errors, provide a general but helpful message.
             userMessage += " Please try your request again. If the problem continues, resetting the session might help.";
        }
    } else {
        // For unknown errors.
        userMessage += " Please try again. If the issue persists, consider resetting the session.";
    }

    addMessage('agent', userMessage, speakableMessage);

    // Revert state to allow the user to continue or retry.
    if (context === 'drafting') {
        // If drafting fails, it's best to return to the refining state
        // to allow for corrections or another attempt.
        currentState = AppState.REFINING;
    }
    // For 'conversation' errors, the state is already 'REFINING' and the UI will be
    // re-enabled by the updateUI() call in the calling function, so no state change is needed.
}

function buildPrompt(): string {
  const historyText = chatHistory.map(m => `${m.role}: ${stripHtml(m.content)}`).join('\n');

  if (currentState === AppState.INTAKE) {
      return `The user has submitted this initial thought: "${rawThought}".
      Analyze it against their journal context and ask your first clarifying question.
      Journal Context: ${journalContextEl.value}`;
  } else {
      return `Continue the conversation based on the user's last message.
      Your goal is to gather enough information to draft a narrative.
      Use Google Search to verify claims.
      When ready, respond ONLY with "DRAFTING_READY".
      
      Journal Context: ${journalContextEl.value}
      Original Thought: ${rawThought}
      Conversation History: ${historyText}`;
  }
}

// --- EVENT LISTENERS ---
chatFormEl.addEventListener('submit', handleUserInput);
resetButtonEl.addEventListener('click', resetState);
confirmYesButtonEl.addEventListener('click', () => handleConfirmation(true));
confirmNoButtonEl.addEventListener('click', () => handleConfirmation(false));

voiceInputButtonEl.addEventListener('click', () => {
    if (isRecording) {
        recognition.stop();
    } else {
        recognition.start();
    }
});

audioOutputButtonEl.addEventListener('click', () => {
    isAudioOutputEnabled = !isAudioOutputEnabled;
    if (isAudioOutputEnabled) {
        speak("Audio output enabled.");
    } else {
        window.speechSynthesis.cancel();
    }
    updateUI();
});

chatInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatFormEl.requestSubmit();
    }
});

chatHistoryEl.addEventListener('click', async (event) => {
    const target = event.target as HTMLElement;
    const button = target.closest('button[data-action]');

    if (!button) return;

    const action = button.getAttribute('data-action');
    const content = button.getAttribute('data-content');

    if (action === 'copy') {
        try {
            await navigator.clipboard.writeText(content);
            button.textContent = 'Copied!';
            setTimeout(() => { button.textContent = 'Copy to Clipboard'; }, 2000);
        } catch (err) {
            console.error('Failed to copy text: ', err);
            button.textContent = 'Copy Failed';
            setTimeout(() => { button.textContent = 'Copy to Clipboard'; }, 2000);
        }
    }

    if (action === 'download') {
        const title = button.getAttribute('data-title');
        const filename = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.md`;
        const blob = new Blob([content], { type: 'text/markdown;charset=utf-8;' });
        const link = document.createElement('a');
        if (link.href) {
            URL.revokeObjectURL(link.href);
        }
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});


// --- INITIAL SETUP ---
function getHardcodedDefaultContext() {
    return `Title: The Impact of Remote Work on Urban Planning
Timestamp: 2023-10-26 10:00:00 UTC
Narrative: The shift to remote work, accelerated by the pandemic, is forcing cities to rethink their commercial districts. Empty office buildings could be converted into residential units to address housing shortages...

---

Title: AI in Creative Writing
Timestamp: 2023-10-25 15:30:00 UTC
Narrative: Exploring the use of AI as a brainstorming partner rather than an author. Tools can suggest plot points or character traits, but the core emotional narrative still requires a human touch.`;
}

function getInitialJournalContext() {
    const storedContext = localStorage.getItem(JOURNAL_CONTEXT_KEY);
    if (storedContext) {
        return storedContext;
    }
    const defaultContext = getHardcodedDefaultContext();
    localStorage.setItem(JOURNAL_CONTEXT_KEY, defaultContext);
    return defaultContext;
}

journalContextEl.value = getInitialJournalContext();
addMessage('agent', 'Welcome to Narrative Sculptor. Please enter an initial thought or idea below to begin the refinement process.', 'Welcome to Narrative Sculptor.');
updateUI();