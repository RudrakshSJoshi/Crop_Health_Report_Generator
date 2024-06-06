from crewai import Crew, Process
from agents import SoilAgents
from tasks import SoilTasks
from dotenv import load_dotenv

load_dotenv()

text_information_list = """
1. Soil pH
2. Last Grown Crop
3. Month
4. Region
5. Soil Nutrient Deficits
6. Soil Nutrient Excess
7. Any additional information
"""

analysis_report_list = """
1. Soil pH
2. Last Grown Crop
3. Rainfall Expectations
4. Weather patterns
5. Soil Deficits
6. Soil Excess
7. Soil Type
8. Soil Health
9. Average Temperature
10. Soil Rejuvenation Methods
10. Any additional information
11. Promising Crops
"""

final_report_template = """
Final Crop Health Report

Crop Health Summary for [Region] in [Month]

Rainfall Expectations:
In the [region], rainfall during the [month] is expected to be [expected rainfall details]. Adequate rainfall is crucial for crop growth, while excessive or insufficient rainfall can be detrimental.

Weather Patterns:
The weather patterns in the [region] for the [month] indicate [weather patterns details]. Typical weather conditions include [average temperatures], which play a critical role in crop growth.

Soil Health Analysis:
The soil in the [region] has been assessed for various health indicators. Currently, the soil nature is [soil pH details]. Soil health also shows [details on soil deficits], indicating a need for specific nutrient supplementation. Conversely, there are [details on soil excess], which could lead to toxicity and should be managed. Overall, the soil health is [overall soil health assessment].

Soil Rejuvenation Methods:
To improve soil health and ensure a conducive environment for crop growth, the following rejuvenation methods are recommended: [detailed recommendations]. These methods will help in balancing soil pH, replenishing deficient nutrients, and mitigating the effects of excessive nutrients.

Additional Information:
Additional observations include [any other relevant information]. This information is critical for making informed decisions about crop management practices.

Recommended Crops for the Upcoming Season:
Based on the analysis of soil and weather conditions, the following crops are recommended for cultivation: [list of promising crops]. These crops are chosen for their suitability to the current soil health, expected rainfall, and typical weather patterns in the region.

This report provides actionable insights into soil and environmental conditions, helping stakeholders make informed decisions for successful crop cultivation in [region] during the [month].
"""

class SoilCrew:
    def __init__(self, text):
        self.text = text
    
    def run(self):
        # 1. Create Agents
        agents = SoilAgents()

        data_collector = agents.expert_data_collector()
        data_analyst = agents.expert_data_analyst()
        report_writer = agents.expert_report_writer()

        # 2. Create Tasks
        tasks = SoilTasks()

        data_collection = tasks.data_collection(
            agent=data_collector,
            text=self.text,
            text_information_list=text_information_list
        )

        data_analysis = tasks.data_analysis(
            agent=data_analyst,
            soil_report=data_collection,
            analysis_report_list=analysis_report_list
        )

        report_writing = tasks.report_writing(
            agent=report_writer,
            detailed_report=data_analysis,
            final_report_template=final_report_template
        )

        # 3. Setup Crew
        crew = Crew(
            agents=[data_collector, data_analyst, report_writer],
            tasks=[data_collection, data_analysis, report_writing],
            max_rpm=10,
            max_iter = 3,
            max_execution_time=30,
            process=Process.sequential,
            verbose=True,
        )

        result=crew.kickoff()
        return result

# 4. Kickoff Crew
if __name__ == "__main__":
    print("## Welcome to Crop & Soil Report Analyzer Crew")
    print('----------------------------------------------')
    info = input("Enter the Report(NOTE: The report must contain soil acidity, region, month, and last grown crops and additional information for best reports)")
    soil_crew = SoilCrew(info)
    result = soil_crew.run()
    print("\n\n########################")
    print("## Here is your Crop & Soil Report Analysis")
    print("########################\n")
    print(result)
