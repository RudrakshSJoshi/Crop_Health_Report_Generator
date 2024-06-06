from crewai import Agent
import os
from tools.search_tools import SearchTools
from tools.summarise_tools import SummariseTools
from langchain_groq import ChatGroq

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="mixtral-8x7b-32768",
)

class SoilAgents:
    def __init__(self):
        pass

    def expert_data_collector(self):
        return Agent(
            role="Data Collecting Expert",
            goal=f"""
                Gather all the important features from the text

                Your goal is to extract all the relevant information from the information provided to you and write a soil report,
                - must include all information related to the soil, the month of growing the crop, the region where the crop will be grown
                - also, the report must include every technical detail precisely
                - in case of missing data, do not assume any value, just move on to find the other remaining details.
                """,
            backstory="""
                Your role as a Data Collecting Expert is to aid the person in knowing everything about their soil and environment there is to know.
                You extract all the specific details about the soil, and are an expert in providing a report about the soil and know the information related
                to month and region of crop being grown.
                """,
            tools=[
                SummariseTools.summarize_text
                ],
            verbose=True,
            llm=llm,
        )

    def expert_data_analyst(self):
        return Agent(
            role="Data Analysis Expert",
            goal=f"""
                Process and analyze the soil report to assess crop health

                You are assigned the task to assess the soil and weather patterns based on the soil report,
                you must then create an analysis report that will include now include the complete soil report, 
                including soil deficit nutrients and soil excess nutrients based on last crop grown, soil pH,
                weather patterns, including expected rainfalls, average temperatures, 
                and a list of promising crops that can be grown on the soil during the specified month and in the specified region based on soil health and weather patterns.
                """,
            backstory=f"""
                As an Expert Data Analyser you understand soil reports and analyse the soil reports provided to you by detecting soil health,
                including soil pH, soil deficits and excess, and weather patterns to find the best possible crop growth in a specific region
                at a specific month of the year.
                You understand soil and needs for the crop health based on soil health for promising crop growth.
                """,
            tools=[
                SearchTools.search_internet,
                SummariseTools.summarize_text
                ],
            verbose=True,
            llm = llm,
        )
    def expert_report_writer(self):
        return Agent(
            role="Report Writing Expert",
            goal=f"""
                Process the analysis report and write a full fledged report

                You are assigned the task of processing the analysis report and then finally writing a full fledged report
                that will include all the current problems with the soil, entire weather report, including rainfall expectations, weather patterns,
                and then it must include all the ways to rejuvenate and treat the soil to good health for a promising crop growth based on the month and region of growing the crop.
                and then finally, it must include a list of all promising crops that can be grown after the treatment of the soil for a promising yield.
                """,
            backstory=f"""
                As an expert report writer, you write extremely detailed reports based on the information provided
                such that you do not miss any information and your reports are always straightforward which doesn't cause any confusion.
                You always make sure the reports you write do not have any discrepancy or missing information that may cause confusion or trouble in understanding your reports.
                """,
            tools=[
                SummariseTools.summarize_text
                ],
            verbose=True,
            llm = llm,
        )