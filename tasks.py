from crewai import Task

class SoilTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def data_collection(self, agent, text, text_information_list):
        return Task(
            description=f"""
                Process the text provided and create a soil report that must include all the important information
                
                IMPORTANT INFORMATION:
                - The report must find all the information from the text provided
                - In case of missing information, do not assume anything, just skip that part
                - The report must include all points provided in text, so that no descrepancy occurs
                - The final report must include all the information that is present in the list provided
                - Do not assume anything, you are only supposed to collect information provided

                Parameters:
                - text: {text}

                The list is provided as a numbered list as follows:
                ```
                {text_information_list}
                ```
            """,
            agent=agent,
            expected_output=f"Soil Data Collection",
            async_execution=False,
        )

    def data_analysis(self, agent, soil_report, analysis_report_list):
        return Task(
            description=f"""
                Analyse the soil report and create an analysis report
                
                IMPORTANT INFORMATION:
                - The analysis report must now include extra information based on soil report provided by searching the internet
                - Additional information must include rainfall expectations, average temperatures, weather patterns, soil health, soil type and a list of promising crops
                - The report must also include all methods to rejuvenate the soil based on soil deficits and soil excess, such as fertilisers required, farming methods, and other techniques
                - The final report must include all the information that is present in the list provided
                - In case of missing information, search the internet for general results specific to that region and month.
                The list is provided as a numbered list as follows:
                ```
                {analysis_report_list}
                ```
            """,
            agent=agent,
            context=[soil_report],
            expected_output=f"Soil Analysis Report",
            async_execution=False,
        )
    def report_writing(self, agent, detailed_report, final_report_template):
        return Task(
            description=f"""
                Analyse the detailed report and write a summarised report

                IMPORTANT INFORMATION:
                - The final report should be summarised, but it should contain all information
                - The report should follow the format of first telling all the information about the soil, 
                then recommendations and methods to treat the soil, 
                and then finally the list of all promising crops to grow.
                - The report should be understandable and easy to read and it should contain as much information as possible.
                - The summarised report must be precise, accurate and properly written.
                - The final report should follow a similar pattern to the template provided, while making sure no information is skipped.

                The template is as follows:
                ```
                {final_report_template}
                ```
            """,
            agent=agent,
            context=[detailed_report],
            expected_output=f"Final Report",
            async_execution=False,
        )