# Multi-Agent Research & Report System

A multi-agent AI research application built with the OpenAI Agents SDK and Streamlit.

The system takes a user question, breaks it into research tasks, searches for relevant information, analyzes the findings, generates a structured report, and reviews the report before presenting the final result.

## Overview

Instead of using a single AI model for the entire task, this project uses multiple specialized agents.

```text
User Question
      ↓
Manager Agent
      ↓
Research Agent
      ↓
Analysis Agent
      ↓
Writer Agent
      ↓
Reviewer Agent
      ↓
Final Report
```

If the reviewer finds problems, the report is sent back to the Writer for revision.

```text
Writer
  ↓
Reviewer
  ↓
Approved?
 ├── Yes → Final Report
 └── No  → Feedback → Writer → Reviewer
```

## Agents

### Manager Agent

The Manager Agent receives the user's question and creates a structured research plan.

It produces:

- Research topic
- Research objective
- Research tasks

### Research Agent

The Research Agent receives the Manager's research plan and uses web search to collect relevant information.

It produces:

- Research findings
- Summaries
- Source URLs
- Research gaps

### Analysis Agent

The Analysis Agent receives the Research Agent's output and analyzes the collected evidence.

It identifies:

- Key findings
- Comparisons
- Important relationships
- Recommendations
- Missing information

The Analysis Agent does not perform additional web research.

### Writer Agent

The Writer Agent receives the research and analysis and converts them into a structured Markdown report.

The report may contain:

- Introduction
- Key Findings
- Comparison / Analysis
- Recommendation
- Limitations
- Sources

### Reviewer Agent

The Reviewer Agent checks the generated report for:

- Accuracy
- Relevance
- Scope adherence
- Unsupported claims
- Structure
- Source usage

It gives the report a score from 1 to 10.

If the report does not meet the required quality threshold, the reviewer provides feedback and the Writer revises the report.

## Features

- Multi-agent workflow
- Agent-to-agent data passing
- Structured outputs using Pydantic
- OpenAI Agents SDK
- Web search
- Automated research planning
- Research analysis
- Report generation
- Automated report review
- Revision loop
- Streamlit frontend
- Real-time workflow progress
- Markdown report download
- Intermediate agent outputs available in separate tabs

## Project Structure

```text
multi-agent-research-report-system/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── research_agents/
│   ├── __init__.py
│   ├── manager.py
│   ├── researcher.py
│   ├── analyst.py
│   ├── writer.py
│   └── reviewer.py
│
├── models/
│   ├── __init__.py
│   └── schemas.py
│
└── services/
    ├── __init__.py
    └── workflow.py
```

## Tech Stack

- Python
- OpenAI Agents SDK
- OpenAI API
- Streamlit
- Pydantic
- Python Dotenv

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd multi-agent-research-report-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit the `.env` file to GitHub.

Make sure `.gitignore` contains:

```text
.env
__pycache__/
*.pyc
.venv/
venv/
```

## Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Streamlit will provide a local URL that you can open in your browser.

## How It Works

The user enters a question such as:

```text
Compare RAG and fine-tuning for enterprise AI applications.
```

The Manager Agent first creates a research plan.

The Research Agent then searches for evidence based on that plan.

The Analysis Agent interprets the research findings.

The Writer Agent generates a readable report.

The Reviewer Agent evaluates the report.

If necessary, the Writer revises the report using the reviewer's feedback.

The final report is then displayed in the Streamlit application.

## Example Questions

The system is not restricted to one field.

- Compare PostgreSQL and MongoDB for a startup.
- Compare Islam Makhachev and Khabib Nurmagomedov based on fighting skills.
- What are the advantages and disadvantages of nuclear energy?
- Compare RAG and fine-tuning for enterprise AI applications.
- FastAPI vs Flask for building AI APIs.
- What are the major approaches to AI agent memory?

The current workflow is best suited for research, comparison, analysis, and report-generation tasks.

## Streamlit Interface

The frontend provides:

```text
Research Question
        ↓
Run Research Team
        ↓
Agent Progress
        ↓
Final Report
```

Users can also inspect individual stages of the workflow through separate tabs:

- Final Report
- Research Plan
- Research
- Analysis
- Review

The final Markdown report can also be downloaded.

## Agent Orchestration

The high-level workflow is controlled by Python.

```python
plan = await run_manager(topic)

research = await run_research(plan)

analysis = await run_analysis(research)

report = await run_writer(
    research,
    analysis
)

review = await run_reviewer(
    topic,
    research,
    analysis,
    report
)
```

This demonstrates the basic principle of multi-agent systems:

```text
Agent A Output
      ↓
Agent B Input
      ↓
Agent C Input
      ↓
Decision / Action
```

## Revision Loop

The system limits the number of revisions to prevent uncontrolled API usage.

```python
for revision in range(max_revisions):

    review = await run_reviewer(...)

    if review.approved:
        break

    report = await revise_report(...)
```

During development, a low revision limit is recommended to reduce API usage.

## Cost Considerations

Multi-agent applications may generate multiple API calls for a single user request.

A single workflow may involve calls from:

```text
Manager
Researcher
Analyst
Writer
Reviewer
Writer Revision
Reviewer Again
```

Using smaller models during development can significantly reduce costs.

The model can also be configured separately for each agent, allowing stronger models to be used only where additional reasoning quality is required.

## Future Improvements

- PDF report downloads
- Follow-up questions about generated reports
- Persistent research history
- User authentication
- Database storage
- Document/PDF research
- RAG over uploaded files
- Better source verification
- Citation formatting
- Agent tracing and observability
- Model selection from the frontend
- Cost/token monitoring
- Parallel research agents
- Human approval before final publication
- Deployment to Hugging Face Spaces or another cloud platform

## Learning Goals

This project demonstrates practical concepts used in agentic AI systems:

- Agent specialization
- Tool calling
- Structured outputs
- Agent orchestration
- State passing
- Web research
- Prompt design
- Conditional execution
- Autonomous loops
- Output validation
- Human-readable report generation
- Multi-agent application architecture

## Disclaimer

AI-generated research may contain incomplete or incorrect information.

Important claims should be verified against reliable primary sources before being used for professional, academic, financial, medical, legal, or other high-impact decisions.

## License

This project is intended for learning, experimentation, and portfolio use.
