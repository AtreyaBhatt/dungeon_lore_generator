# --------- Imports and Setup ---------
import nest_asyncio
nest_asyncio.apply()

import os

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from crewai import Agent, Task, Crew, LLM
from firecrawl import AsyncFirecrawlApp
import asyncio

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# ------------- API KEYS ------------------
GENAI_API_KEY = "YOUR-GEMINI-KEY"
FIRECRAWL_API_KEY = "YOUR-FIRECRAWL-KEY"

# ------------- LLMs and Clients Setup -------
genai.configure(api_key=GENAI_API_KEY)
firecrawl_agent = AsyncFirecrawlApp(api_key=FIRECRAWL_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")
my_llm = LLM(
    model='gemini/gemini-1.5-flash',
    api_key=GENAI_API_KEY
)

# ------------- Rich Console -------------
console = Console()

# ------------- Lore Data Preparation ---------
async def fetch_firecrawl_data(firecrawl_agent):
    response = await firecrawl_agent.scrape_url(
        url="https://en.wikipedia.org/wiki/Quest_%28video_games%29",
        formats=["markdown"],
        only_main_content=True,
        parse_pdf=True
    )
    try:
        markdown_text = response.markdown
    except AttributeError:
        print("No .markdown attribute found in firecrawl response.")
        return []
    chunks = [c.strip() for c in markdown_text.split("\n") if len(c.strip()) > 50]
    return chunks

# Run to get and cache data at process startup
wiki_chunks = asyncio.run(fetch_firecrawl_data(firecrawl_agent))
if not wiki_chunks:
    raise RuntimeError("Failed to fetch lore data from Wikipedia. Check API keys and connectivity.")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
lore_chunks = wiki_chunks[:25]
embeddings = embedder.encode(lore_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve_lore(query: str, k: int = 3) -> str:
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k)
    return "\n".join([lore_chunks[i] for i in I[0]])

# ---------- Lore Generator Crew Agents/Tasks ----------
def lore_task_fn(user_prompt):
    prompt = f"""
The following lore was retrieved based on the prompt: "{user_prompt}"

Expand this into a detailed lore description including:
- History, mythology, geography, magic, factions

Return it as a Markdown section titled '## Expanded Lore'.
"""
    chunks = retrieve_lore(user_prompt)
    prompt += "\n----\n" + chunks
    return gemini.generate_content(prompt).text

def quest_task_fn(expanded_lore, user_prompt):
    prompt = f"""
You are a quest designer. Based on the user prompt and expanded lore, create a full quest including characters, mechanics, and progression.

### User Prompt:
{user_prompt}

### Expanded Lore:
{expanded_lore}

Respond in Markdown with sections:
## Quest Overview
## Characters
## Game Mechanics
## Progression (Acts I–III)
## Rewards
"""
    return gemini.generate_content(prompt).text

def balance_task_fn(quest_markdown):
    prompt = f"""
Balance the following quest for gameplay challenge, pacing, and reward fairness.
Fix issues and return updated Markdown.

{quest_markdown}
"""
    return gemini.generate_content(prompt).text

def qa_task_fn(quest_markdown):
    prompt = f"""
Final QA and polish. Ensure consistency, formatting, and clarity.
Return the final quest scroll in Markdown.

{quest_markdown}
"""
    return gemini.generate_content(prompt).text

def run_game_quest_pipeline(user_prompt: str):
    lore_agent = Agent(
        role="Lore Historian",
        goal="Retrieve and expand ancient game lore",
        backstory="Sage Arin, The Lorekeeper is the last of the Lorekeepers, sworn to preserve forgotten tales.",
        llm=my_llm
    )
    lore_task = Task(
        description="Retrieve and expand lore for: " + user_prompt,
        agent=lore_agent,
        expected_output="## Expanded Lore",
        function=lambda: lore_task_fn(user_prompt)
    )

    quest_agent = Agent(
        role="Quest Designer",
        goal="Craft immersive RPG quests",
        backstory="Lady Lyra, The Bard of the Inkflame is a playwright turned adventurer, weaving epic narratives for heroes.",
        llm = my_llm
    )
    quest_task = Task(
        description="Design quest using expanded lore",
        agent=quest_agent,
        expected_output="## Quest Overview",
        function=lambda: quest_task_fn(lore_task.output, user_prompt)
    )

    balance_agent = Agent(
        role="Game Balancer",
        goal="Ensure challenge and reward are balanced",
        backstory="Grimbweld Sigen, The Tactician once judged the Grand Arena. Now he balances power and pace.",
        llm = my_llm
    )
    balance_task = Task(
        description="Balance the quest",
        agent=balance_agent,
        expected_output="Balanced Markdown Quest",
        function=lambda: balance_task_fn(quest_task.output)
    )

    qa_agent = Agent(
        role="Quest Archivist",
        goal="Polish the final quest for clarity and consistency",
        backstory="Sir Agustus, The Scribe served as a royal scribe and ensures no tale is poorly told.",
        llm = my_llm
    )
    qa_task = Task(
        description="Polish and finalize the quest",
        agent=qa_agent,
        expected_output="Final Markdown Quest",
        function=lambda: qa_task_fn(balance_task.output)
    )

    crew = Crew(
        agents=[lore_agent, quest_agent, balance_agent, qa_agent],
        tasks=[lore_task, quest_task, balance_task, qa_task]
    )

    results = crew.kickoff()
    return results

# --------------- Conversational Crew ---------------
def conversational_agent_fn(message_from_other_agent, role):
    # Displays agent output with rich formatting
    panel = Panel.fit(message_from_other_agent, title=f"[b magenta]{role}[/b magenta]", border_style="magenta")
    console.print(panel)
    return message_from_other_agent

def intent_handler_fn(user_input):
    prompt = f"""
Extract and summarize the user's main intent in a single sentence:

User: "{user_input}"
Summary:
"""
    return gemini.generate_content(prompt).text.strip()

def clarity_agent_fn(user_input, extracted_intent):
    prompt = f"""
Given the following user input and extracted intent, suggest if the user might wish to clarify, expand, or add details.

User Input: "{user_input}"
Extracted Intent: "{extracted_intent}"

Respond with any suggested clarifying questions or additional details. If none are needed, reply "No further details needed."
"""
    return gemini.generate_content(prompt).text.strip()

def structure_agent_fn(final_intent, additional_details):
    prompt = f"""
Given the intent: "{final_intent}"
and extra details: "{additional_details}"

Structure a clear, concise task request for the game lore generation pipeline.
"""
    return gemini.generate_content(prompt).text.strip()

# ------------- Chat Loop -------------
def conversational_chat():
    console.print("[bold cyan]Welcome to the RPG Quest Creator![/bold cyan]\nType 'exit' to quit.\n")
    while True:
        # 1. Get input
        user_input = Prompt.ask("[bold green]You[/bold green]")
        if user_input.strip().lower() == 'exit':
            console.print("[bold red]Goodbye![/bold red]")
            break

        # 2. Intent handler
        intent_summary = intent_handler_fn(user_input)
        conversational_agent_fn(f"Intent extracted: [bold]{intent_summary}[/bold]", "Intent Handler")

        # 3. Clarity agent
        clarity_question = clarity_agent_fn(user_input, intent_summary)
        if "No further details needed" not in clarity_question:
            conversational_agent_fn(clarity_question, "Clarity Agent")
            user_details = Prompt.ask("[bold yellow]Your Additions (or press enter if none)[/bold yellow]")
        else:
            user_details = ""
            conversational_agent_fn("No additional clarification needed.", "Clarity Agent")

        # 4. Structure agent
        structured_request = structure_agent_fn(intent_summary, user_details)
        conversational_agent_fn(structured_request, "Structure Agent")

        # 5. Pass to lore generator crew
        conversational_agent_fn("[italic]Generating your RPG quest...[/italic]", "System")
        final_scroll = run_game_quest_pipeline(structured_request)

        # 6. Final chat output with markdown formatting
        conversational_agent_fn("[b green]Here's your generated quest![/b green]", "Lore Generator")
        console.print(Markdown(final_scroll.raw))
        console.rule()

# ---------- Main ----------
if __name__ == "__main__":
    conversational_chat()
