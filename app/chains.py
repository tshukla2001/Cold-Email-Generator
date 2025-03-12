from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from model import Model
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader

llm_model = Model(
    model_name = "llama-3.3-70b-versatile"
)
llm_model = llm_model.call()

json_parser = JsonOutputParser()

class Chains:
    def __init__(self, job_url, resume_file):
        self.job_url = job_url
        self.resume_file = resume_file
    
    def job_extract_json(self, job_url):
        loader = WebBaseLoader(web_path = job_url)
        page_data = loader.load().pop().page_content

        json_prompt = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM GIVEN WEBSITE:
            {data_of_page}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to analyze the data properly in detail and convert it into a JSON format which contains the following keys: `role`, `skills`, `experience`, `description`.
            Only return a valid JSON format.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_for_extract = json_prompt | llm_model
        json_job_page = chain_for_extract.invoke(input={'data_of_page': page_data})

        parsed_job_data = json_parser.parse(json_job_page.content)

        return parsed_job_data
    
    def resume_extract_json(self, resume_file):

        if resume_file:
            temp_file = "../resume_temp.pdf"
            with open(temp_file, "wb") as file:
                file.write(resume_file.getvalue())
                # file_name = resume_file.name

        loader_pdf = PyPDFLoader(file_path = temp_file, mode = "single", pages_delimiter = "")
        resume_details = loader_pdf.load()[0].page_content

        resume_prompt = PromptTemplate.from_template(
            """
            ### RESUME CONTENT:
            {resume_data}
            ### INSTRUCTIONS:
            This is the extracted data from a resume. Convert the data into a valid json format with the following keys: `name`, `mobile no.`, 
            `linkedin URL`, `summary`, `education`, `skills`, experience, `projects`, `certifications` and `leadership roles`.
            Convert into a valid json format
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_for_resume = resume_prompt | llm_model
        extracted_resume = chain_for_resume.invoke(input = {'resume_data': resume_details})

        resume_json = json_parser.parse(extracted_resume.content)
        return resume_json
    
    def job_relevance_with_resume(self, job_json, resume_json):
        job_relevance_prompt = PromptTemplate.from_template(
            """
            ### JOB DETAILS IN JSON:
            {job_details_json}
            ### RESUME DEATILS IN JSON:
            {resume_details_json}
            ### INSTRUCTIONS:
            The above two are json formats of a job detail from a career's page of a website and the 
            resume details of a person respectively. Analyze both the json formats and find out how
            relevant the job is for the person based on the `skills` mentioned in the job details and the
            `skills` mentioned in the resume. Return the sentence `Yayy!! This job is relevant for you 👍` if the
            `skills` are similar in both resume and job details, otherwise return `Sorry, this job is not relevant
            for you 👎`. Don't mention any code or how to calculate relevance. Just return the output
            for the given job and resume details.
            ### (NO PREAMBLE):
            """
        )

        job_relevance_chain = job_relevance_prompt | llm_model
        job_relevance_result = job_relevance_chain.invoke(input = {'job_details_json': job_json, 'resume_details_json': resume_json})
        
        return job_relevance_result.content
    
    def cold_mail_generator(self, job_json_format, resume_json_format):
        cold_email_prompt = PromptTemplate.from_template(
            """
            ### JOB DETAILS IN JSON:
            {job_details_json}
            ### RESUME DEATILS IN JSON:
            {resume_details_json}
            ### INSTRUCTIONS:
            The above two are json formats of a job detail from a career's page of a website and the resume details of a person respectively. Analyze both and
            generate a mail with a subject and a salutation addressed to the hiring manager of the company. The mail should showcase the person's interest
            (whose resume is attached) in the job highlighting the important details from their resume and not the ones mentioned in the job detail. Don't 
            mention anything about the person's `experience` if there is nothing in the resume and only talk about the person's experience if he/she has 
            some `experience` in the resume. If there is no `experience` in the resume then talk about academic experience through projects else talk about 
            the person's `experience` in the resume like what did they do in the company and what skills were used but only when there is some `experience` in
            the resume. Talk about the person's `projects` in brief if no `experience` (in bullet points). Mention a few lines about the `certifications` as well.
            The mail should be formal and a minimum of 2-3 paragraphs long. The mail should be in first person format like you are the person who is applying 
            for the job. Also mention that you have attached the resume in the mail for further reference. Make the mail convincing. At the end after the name,
            add the linkedin url, and the `mobile no.` of the person from their resume. Add a little space after every paragraph.
            ### (NO PREAMBLE):
            """
        )

        cold_email_chain = cold_email_prompt | llm_model
        cold_email_result = cold_email_chain.invoke(input = {'job_details_json': job_json_format, 'resume_details_json': resume_json_format})
        
        return cold_email_result.content

if __name__ == "__main__":
    cold_email = Chains(
        job_url = "https://www.indeed.com/viewjob?jk=1f3f8b3a7c2f3b4f",
        resume_file = "resume.pdf"
    )

    job_json = cold_email.job_extract_json(cold_email.job_url)
    resume_json = cold_email.resume_extract_json(cold_email.resume_file)

    cold_email_result = cold_email.cold_main_generator(job_json, resume_json)
    print(cold_email_result)
