from chains import Chains
import streamlit as st

st.title("✉️ Cold Email Generator")
st.text("This is a simple cold email generator which generates a cold email based on the job link and resume provided.")

job_url = st.text_input("Enter the job link here:")
resume_file = st.file_uploader("Upload your resume here:", type = ['pdf'])

if st.button("Generate mail"):
    cold_email = Chains(job_url = job_url, resume_file = resume_file)
    job_json = cold_email.job_extract_json(job_url)
    resume_json = cold_email.resume_extract_json(resume_file)
    generated_mail = cold_email.cold_mail_generator(job_json_format = job_json, resume_json_format = resume_json)
    job_relevance = cold_email.job_relevance_with_resume(job_json, resume_json)
    st.write(f'<p style="background-color:green; color:white; padding:10px; border-radius:5px;">{job_relevance}</p>', unsafe_allow_html=True)
    st.write(generated_mail)