/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

// Fix: Removed unused 'Type' import.
import { GoogleGenAI } from '@google/genai';
import { marked } from 'marked';

// --- CONFIGURATION ---
const API_KEY = process.env.API_KEY;
const MODEL_NAME = 'gemini-2.5-flash';

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
// Fix: Cast window to `any` to resolve TypeScript errors for non-standard SpeechRecognition APIs.
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
const ai = new GoogleGenAI({ apiKey: API_KEY });

const SYSTEM_INSTRUCTION = `You are Narrative Sculptor, a supportive, Socratic, and fact-checking agent. Your goal is to help the user refine their unstructured thoughts into a single, cohesive, and factually-backed narrative entry for their journal.

**Your Persona & Rules:**
1.  **Interaction:** Be friendly, non-judgemental, highly focused, and analytical.
2.  **Context:** You will be given the user's previous journal entries. Always analyze the user's new input against this context to identify thematic overlaps, continuity, and progression of ideas.
3.  **Fact-Grounding:** For any claims requiring external validation, you MUST use the provided Google Search tool. Only use information from the search results. Do not invent facts. If a claim cannot be verified, flag it as speculative.
4.  **Questioning:** Only ask one, specific clarifying question at a time. Your questions should aim to define the core topic/thesis, identify the intended audience or purpose, fill logical gaps, or determine the desired tone or structure.
5.  **State Transitions:** During the 'Refining' phase, after you've gathered enough information, you must respond with the exact string "DRAFTING_READY" and nothing else. This will trigger the next step. Do not say this until you are confident you can produce a high-quality draft.`;


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
    speak(speakableText || stripHtml(parsedContent));
  }
  
  updateUI();
}

function resetState() {
    currentState = AppState.INTAKE;
    chatHistory = [];
    rawThought = '';
    finalDraft = null;
    journalContextEl.value = getInitialJournalContext();
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

    const response = await callGemini(prompt, useSearch);

    if (response.trim() === 'DRAFTING_READY') {
      await triggerDrafting();
    } else {
      addMessage('agent', response);
    }
  } catch (error) {
    console.error("Error during conversation:", error);
    addMessage('agent', "Sorry, I encountered an error. Please try again.", "Sorry, I encountered an error.");
    currentState = AppState.INTAKE;
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
        
        Generate a JSON object with the final title, narrative, and sources. The narrative must include citation markers like [1] corresponding to the sources.`;

        const resultJson = await callGemini(draftingPrompt, false, true);
        finalDraft = resultJson;
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
        console.error("Error during drafting:", error);
        addMessage('agent', "Sorry, I failed to generate the draft. Let's start over.", "Draft generation failed. Starting over.");
        resetState();
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
        
        const formattedEntry = `
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
        
        addMessage('agent', `<p>Excellent! The following entry has been formatted and archived to your 'IdeasJournal'.</p>${formattedEntry}<p>You can start a new thought below.</p>`, 'Excellent! The entry has been archived. You can start a new thought.');
        
        // Simulate updating context
        const newEntryText = `Title: ${finalDraft.title}\nTimestamp: ${timestamp}\nNarrative: ${finalDraft.narrative.substring(0, 150)}...\n---\n\n`;
        journalContextEl.value = newEntryText + journalContextEl.value;

        currentState = AppState.INTAKE;
    } else {
        addMessage('agent', "No problem. Let's discard this draft. Please enter a new thought to start over.", "Draft discarded. Please start over.");
        resetState();
    }
    updateUI();
}

// --- GEMINI API CALLER ---
// Fix: Refactored to use the modern `ai.models.generateContent` API, replacing the deprecated `ai.getGenerativeModel`.
// This consolidates model configuration into a single, compliant API call.
async function callGemini(prompt: string, useSearch: boolean, useJson = false) {
  const config: any = {
    systemInstruction: SYSTEM_INSTRUCTION,
  };

  if (useSearch) {
    config.tools = [{ googleSearch: {} }];
  }
  if (useJson) {
    config.responseMimeType = 'application/json';
  }

  const response = await ai.models.generateContent({
    model: MODEL_NAME,
    contents: prompt,
    config: config,
  });

  const text = response.text;

  if (useJson) {
      return JSON.parse(text);
  }

  // If search was used, let's append the sources to the text for context
  const groundingMetadata = response.candidates?.[0]?.groundingMetadata;
  if (useSearch && groundingMetadata?.groundingChunks) {
    const citations = groundingMetadata.groundingChunks
        .map((chunk, i) => `[${i+1}] ${chunk.web.title}: ${chunk.web.uri}`)
        .join('\n');
    if (citations) {
        return `${text}\n\n*Referenced sources for validation.*\n`;
    }
  }

  return text;
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


// --- INITIAL SETUP ---
function getInitialJournalContext() {
    return `Title: The Impact of Remote Work on Urban Planning
Timestamp: 2023-10-26 10:00:00 UTC
Narrative: The shift to remote work, accelerated by the pandemic, is forcing cities to rethink their commercial districts. Empty office buildings could be converted into residential units to address housing shortages...

---

Title: AI in Creative Writing
Timestamp: 2023-10-25 15:30:00 UTC
Narrative: Exploring the use of AI as a brainstorming partner rather than an author. Tools can suggest plot points or character traits, but the core emotional narrative still requires a human touch.`;
}

journalContextEl.value = getInitialJournalContext();
addMessage('agent', 'Welcome to Narrative Sculptor. Please enter an initial thought or idea below to begin the refinement process.', 'Welcome to Narrative Sculptor.');
updateUI();
