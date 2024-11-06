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
                agent=self.agents["LearningContentAnalyst"]
            ),
            Task(
                description="Generate pre-test questions based on the summary provided by LearningContentAnalyst.",
                expected_output="A list of 3 pre-test questions.",
                agent=self.agents["QuestionGenerator"]
            ),
            Task(
                description="Ask pre-test questions one by one to the user and record answers.",
                expected_output="A Q&A report including the user's answers.",
                agent=self.agents["QuestionGenerator"]
            ),
            Task(
                description="Evaluate the user's answers from pre-test Q&A and provide feedback for Facilitator.",
                expected_output="An evaluation report with recommendations for learning focus areas.",
                agent=self.agents["Evaluator"]
            ),
            Task(
                description="Using feedback from Evaluator, guide the user through the key learning points, allowing for user interaction.",
                expected_output="A detailed learning session report with key concepts explained and user interactions documented.",
                agent=self.agents["Facilitator"]
            ),
            Task(
                description="Generate post-test questions based on the learning content covered.",
                expected_output="A list of 3 post-test questions targeting key learning objectives.",
                agent=self.agents["QuestionGenerator"]
            ),
            Task(
                description="Ask post-test questions one by one and record user's answers.",
                expected_output="A comprehensive Q&A report with user's post-test responses.",
                agent=self.agents["QuestionGenerator"]
            ),
            Task(
                description="Evaluate post-test answers and create an assessment report.",
                expected_output="A detailed assessment report comparing pre and post-test performance.",
                agent=self.agents["Evaluator"]
            ),
            Task(
                description="Based on interaction history and post-test evaluation, generate a final learning report and notes for the user.",
                expected_output="A comprehensive learning summary with performance analysis and recommendations.",
                agent=self.agents["SummaryExpert"]
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
            "evaluation_facilitation_crew": Crew(
            agents=[self.agents["Evaluator"], self.agents["Facilitator"]],
            tasks=[self.tasks[3], self.tasks[4]],
            process=Process.sequential
            ),

            # "facilitation_crew": Crew(
            #     agents=[self.agents["Facilitator"]],
            #     tasks=[self.tasks[4]],
            #     process=Process.sequential
            # ),
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
        
        # Ask for confirmation to start pre-test
        await cl.Message(content="Would you like to start the pre-test questions? (Please reply with 'yes' to begin)").send()
        
        # Set phase to await confirmation
        cl.user_session.set("phase", "await_confirmation")
        
        # Store pretest data for later use
        pretest_result = await session.run_phase("pretest_crew")
        questions = str(pretest_result).split('\n')
        if not questions:
            questions = ["What do you know about this topic?"]
        cl.user_session.set("questions", questions)
        cl.user_session.set("current_question", 0)
        cl.user_session.set("answers", [])
        
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages"""
    session = cl.user_session.get("session")
    phase = cl.user_session.get("phase")
    
    try:
        if phase == "await_confirmation":
            if message.content.lower() in ["yes", "y", "sure", "okay", "ok"]:
                # Change phase to pretest
                cl.user_session.set("phase", "pretest")
                
                # Send pre-test start message
                await cl.Message(content="Let's start with some questions to assess your current understanding.").send()
                
                # Ask first question
                questions = cl.user_session.get("questions")
                await cl.Message(content=questions[0]).send()
            else:
                await cl.Message(content="No problem. Just reply with 'yes' when you're ready to start the pre-test.").send()
                
        elif phase == "pretest":
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
                # Pre-test complete, show Q&A report
                await cl.Message(content="Here is the Q&A report including the user's answers:").send()
                
                # Format and display Q&A report
                qa_report = "Q&A Summary:\n\n"
                for i, (q, a) in enumerate(zip(questions, answers)):
                    qa_report += f"Q{i+1}: {q}\nA: {a}\n\n"
                await cl.Message(content=qa_report).send()
                
                # Move to evaluation and facilitation
                cl.user_session.set("phase", "evaluation_facilitation")
                await cl.Message(content="Let me evaluate your answers and guide you through the learning points...").send()
                
                # Run combined evaluation and facilitation
                result = await session.run_phase("evaluation_facilitation_crew")
                await cl.Message(content=str(result)).send()
                
                # Move to post-test phase
                cl.user_session.set("phase", "posttest")
                posttest_result = await session.run_phase("posttest_crew")
                await cl.Message(content=str(posttest_result)).send()
        
        elif phase == "evaluation_facilitation":
            result = await session.run_phase("evaluation_facilitation_crew")
            await cl.Message(content=str(result)).send()
        
        elif phase == "posttest":
            posttest_result = await session.run_phase("posttest_crew")
            await cl.Message(content=str(posttest_result)).send()
        
        elif phase == "summary":
            summary_result = await session.run_phase("summary_crew")
            await cl.Message(content=str(summary_result)).send()
            
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

if __name__ == "__main__":
    cl.run()
