import os
import asyncio
from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool
from langchain_openai import ChatOpenAI
from textwrap import dedent
import chainlit as cl

# Initialize environment variables
GROQ_API_KEY = "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"  # Replace with your actual API key
os.environ['GROQ_API_KEY'] = GROQ_API_KEY  # Set it directly

# Initialize LLM with Groq
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY'],
    model="groq/mixtral-8x7b-32768",  # Added 'groq/' prefix
    temperature=0,
    max_tokens=1000
)

def setup_rag_tool(pdf_path):
    """Initialize RAG tool with the provided PDF"""
    return PDFSearchTool(
        pdf=pdf_path,
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=os.environ['GROQ_API_KEY'],
                    model="groq/mixtral-8x7b-32768"  # Added 'groq/' prefix
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

class LearningSession:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.rag_tool = setup_rag_tool(pdf_path)
        self.agents = self.create_agents()
        self.tasks = self.create_tasks()
        self.crews = self.create_crews()
        
    def create_agents(self):
        """Create and return all agents"""
        return {
            "LearningContentAnalyst": Agent(
                role='LearningContentAnalyst',
                goal='Analyze content from PDF and provide brief summary for context.',
                backstory="You're an expert in analyzing content in the uploaded materials and summarizing them so that they are easily digestible.",
                tools=[self.rag_tool],
                llm=llm
            ),
            "Evaluator": Agent(
                role='Evaluator',
                goal='Evaluate the answers from users and provide constructive feedback',
                backstory="You're an expert in evaluating students' prior knowledge based on the answers in the pre-test and providing suggestions to the QuestionGenerator during the test.",
                tools=[],
                llm=llm
            ),
            "QuestionGenerator": Agent(
                role='QuestionGenerator',
                goal='Generate questions about the uploaded document to assess user understanding',
                backstory="You are an expert in generating questions based on the summary of the learning content for the week.",
                tools=[],
                llm=llm
            ),
            "Facilitator": Agent(
                role='Facilitator',
                goal='Facilitate conversation with user to encourage active thinking',
                backstory="You're an expert in facilitating learning through asking questions related to the summary of the learning content.",
                tools=[],
                llm=llm
            ),
            "SummaryExpert": Agent(
                role='SummaryExpert',
                goal='Summarize user conversation history and learning content',
                backstory="You're an expert in summarizing the conversation history between students and the AI agents.",
                tools=[],
                llm=llm
            )
        }

    def create_tasks(self):
        """Create and return all tasks"""
        return [
            Task(
                description="Analyze the weekly learning materials and generate a summary.",
                expected_output="A summary of the weekly learning materials with key content.",
                agent=self.agents["LearningContentAnalyst"]  # Fixed: using self.agents instead of Agent
            ),
            Task(
                description="Generate pre-test questions based on the summary provided by LearningContentAnalyst.",
                expected_output="A list of 3 pre-test questions.",
                agent=self.agents["QuestionGenerator"]  # Fixed: using self.agents instead of Agent
            ),
            Task(
                description="Ask pre-test questions one by one to the user and record answers.",
                expected_output="A Q&A report including the user's answers.",
                agent=self.agents["QuestionGenerator"]  # Fixed: using self.agents instead of Agent
            ),
            Task(
                description="Evaluate the user's answers from pre-test Q&A and provide feedback for Facilitator.",
                expected_output="An evaluation report (not shown to the user).",
                agent=self.agents["Evaluator"]  # Fixed: using self.agents instead of Agent
            ),
            Task(
                description="Using feedback from Evaluator, guide the user through the key learning points, allowing for user interaction.",
                expected_output="A report of the learning session with key content explanations.",
                agent=self.agents["Facilitator"]  # Fixed: using self.agents instead of Agent
            ),
            Task(
                description="Generate post-test questions based on the learning content covered.",
                expected_output="A list of 3 post-test questions.",
                agent=self.agents["QuestionGenerator"]  # Fixed: using self.agents instead of Agent
            ),
            Task(
                description="Ask post-test questions one by one and record user's answers.",
                expected_output="A Q&A report including user's answers.",
                agent=self.agents["QuestionGenerator"]  # Fixed: using self.agents instead of Agent
            ),
            Task(
                description="Evaluate post-test answers and create an assessment report (not shown to user).",
                expected_output="A post-test evaluation report.",
                agent=self.agents["Evaluator"]  # Fixed: using self.agents instead of Agent
            ),
            Task(
                description="Based on interaction history and post-test evaluation, generate a final learning report and notes for the user.",
                expected_output="A PDF summary of learning content and performance feedback.",
                agent=self.agents["SummaryExpert"]  # Fixed: using self.agents instead of Agent
            )
        ]
    def create_crews(self):
        """Create and return all crews"""
        return {
            "analysis_crew": Crew(
                agents=[self.agents["LearningContentAnalyst"]],
                tasks=[self.tasks[0]],
                process=Process.sequential
            ),
            "pretest_crew": Crew(
                agents=[self.agents["QuestionGenerator"]],
                tasks=[self.tasks[1], self.tasks[2]],
                process=Process.sequential
            ),
            "evaluation_crew": Crew(
                agents=[self.agents["Evaluator"]],
                tasks=[self.tasks[3]],
                process=Process.sequential
            ),
            "facilitation_crew": Crew(
                agents=[self.agents["Facilitator"]],
                tasks=[self.tasks[4]],
                process=Process.sequential
            ),
            "posttest_crew": Crew(
                agents=[self.agents["QuestionGenerator"]],
                tasks=[self.tasks[5], self.tasks[6]],
                process=Process.sequential
            ),
            "summary_crew": Crew(
                agents=[self.agents["SummaryExpert"]],
                tasks=[self.tasks[7]],
                process=Process.sequential
            )
        }

    async def run_phase(self, phase_name):
        """Run a specific phase of the learning session"""
        crew = self.crews.get(phase_name)
        if not crew:
            raise ValueError(f"Unknown phase: {phase_name}")
        # Create a background task for crew.kickoff()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff)
        return result

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    await cl.Message(
        content="# Welcome to PrepMaster! 🚀🤖\nHere, you will be helped by a team of virtual experts to better prepare for the weekly AI Application course."
    ).send()

    # Ask for PDF upload
    files = await cl.AskFileMessage(
        content="Please upload your learning materials",
        accept=["application/pdf"],
        max_size_mb=20,
    ).send()

    # Initialize learning session
    session = LearningSession(files[0].path)
    cl.user_session.set("session", session)
    cl.user_session.set("phase", "analysis")

    # Start content analysis
    await cl.Message(content="Analyzing your materials...").send()
    try:
        # Run analysis phase
        result = await session.run_phase("analysis_crew")
        await cl.Message(content=f"Analysis complete! Here's a summary:\n\n{result}").send()
        
        # Move to pre-test phase
        cl.user_session.set("phase", "pretest")
        pretest_result = await session.run_phase("pretest_crew")
        
        # Store questions and start pre-test
        questions = pretest_result.split('\n')  # Adjust based on actual format
        cl.user_session.set("questions", questions)
        cl.user_session.set("current_question", 0)
        cl.user_session.set("answers", [])
        
        await cl.Message(content="Let's start with some questions to assess your current understanding.").send()
        await cl.Message(content=questions[0]).send()
        
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages"""
    session = cl.user_session.get("session")
    phase = cl.user_session.get("phase")
    
    try:
        if phase == "pretest":
            answers = cl.user_session.get("answers")
            current_q = cl.user_session.get("current_question")
            questions = cl.user_session.get("questions")
            
            # Record answer
            answers.append(message.content)
            cl.user_session.set("answers", answers)
            
            if current_q < len(questions) - 1:
                # More questions to ask
                current_q += 1
                cl.user_session.set("current_question", current_q)
                await cl.Message(content=questions[current_q]).send()
            else:
                # Pre-test complete, move to evaluation
                cl.user_session.set("phase", "evaluation")
                await cl.Message(content="Thank you for completing the pre-test. Let me evaluate your answers...").send()
                
                # Run evaluation asynchronously
                evaluation_result = await session.run_phase("evaluation_crew")
                
                # Move to facilitation
                cl.user_session.set("phase", "facilitation")
                facilitation_result = await session.run_phase("facilitation_crew")
                await cl.Message(content=facilitation_result).send()
                
        elif phase == "facilitation":
            await cl.Message(content="Let's explore that concept further...").send()
            # Add async facilitation logic here
            
        elif phase == "posttest":
            posttest_result = await session.run_phase("posttest_crew")
            await cl.Message(content=posttest_result).send()
            
        elif phase == "summary":
            summary_result = await session.run_phase("summary_crew")
            await cl.Message(content=summary_result).send()
            
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

if __name__ == "__main__":
    cl.run()
