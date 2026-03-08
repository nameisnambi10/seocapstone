import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import streamlit as st
import pickle
import pandas as pd
import shap
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from transformers import pipeline

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI SEO Intelligence Platform",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI SEO Content Intelligence Platform")

page = st.radio(
    "Navigation",
    ["Home","SEO Analyzer","About"],
    horizontal=True
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

import joblib

@st.cache_resource
def load_model():

    package = joblib.load("seo_model.pkl")

    model = package["model"]
    features = package["features"]

    explainer = shap.TreeExplainer(model)

    return model, features, explainer


model, feature_names, explainer = load_model()


# ------------------------------------------------
# LOAD AI MODEL
# ------------------------------------------------

@st.cache_resource
def load_llm():

    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

    return generator

# ------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------

def extract_features(title,description,keyword):

    title_len=len(title)
    desc_len=len(description)

    title_words=len(title.split())
    desc_words=len(description.split())

    kw_in_title=1 if keyword.lower() in title.lower() else 0
    kw_in_desc=1 if keyword.lower() in description.lower() else 0

    kw_density_title=title.lower().count(keyword.lower())/max(title_words,1)
    kw_density_desc=description.lower().count(keyword.lower())/max(desc_words,1)

    kw_pos_title=title.lower().find(keyword.lower()) if kw_in_title else -1

    title_desc_ratio=title_len/(desc_len+1)

    early_kw_score=kw_pos_title/(title_len+1)

    return [[
        title_len,
        desc_len,
        title_words,
        desc_words,
        kw_in_title,
        kw_in_desc,
        kw_density_title,
        kw_density_desc,
        kw_pos_title,
        title_desc_ratio,
        early_kw_score
    ]]

# ------------------------------------------------
# SCRAPE PAGES (403 FIX)
# ------------------------------------------------

def scrape_pages(urls):

    texts=[]

    headers={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language":"en-US,en;q=0.9"
    }

    for url in urls:

        try:

            r=requests.get(url,headers=headers,timeout=10)

            if r.status_code!=200:
                texts.append("")
                continue

            soup=BeautifulSoup(r.text,"html.parser")

            paragraphs=soup.find_all("p")

            text=" ".join([p.get_text() for p in paragraphs])

            texts.append(text)

        except:
            texts.append("")

    return texts

# ------------------------------------------------
# METRICS
# ------------------------------------------------

def calculate_metrics(title,description,keyword):

    return{
        "Title Length":len(title),
        "Description Words":len(description.split()),
        "Keyword Frequency":title.lower().count(keyword.lower())
    }

def competitor_metrics(texts):

    title_lengths=[]
    desc_words=[]

    for t in texts:

        if t:
            words=t.split()
            title_lengths.append(len(words[:12]))
            desc_words.append(len(words[:30]))

    if len(title_lengths)==0:

        return{
        "Title Length":0,
        "Description Words":0,
        "Keyword Frequency":0
        }

    return{
        "Title Length":sum(title_lengths)/len(title_lengths),
        "Description Words":sum(desc_words)/len(desc_words),
        "Keyword Frequency":1
    }

# ------------------------------------------------
# SEO SCORE GAUGE
# ------------------------------------------------

def seo_score_meter(score):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "SEO Score"},
        gauge={'axis':{'range':[0,100]},
               'bar':{'color':"#00cc96"}}
    ))

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# KEYWORD DIFFICULTY
# ------------------------------------------------

def keyword_difficulty_meter(keyword):

    difficulty=min(len(keyword.split())*25,100)

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=difficulty,
        title={'text':"Keyword Difficulty"},
        gauge={'axis':{'range':[0,100]}}
    ))

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# SERP RANK
# ------------------------------------------------

def serp_rank_meter(score):

    if score>=80:
        position=3
    elif score>=60:
        position=10
    elif score>=40:
        position=20
    else:
        position=50

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=position,
        title={'text':"Predicted Google Rank"},
        gauge={'axis':{'range':[1,50]}}
    ))

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# SEO CHARTS
# ------------------------------------------------

def show_seo_charts(your_metrics,comp_metrics):

    sns.set_style("darkgrid")

    # Title Length
    fig1,ax1=plt.subplots()

    df1=pd.DataFrame({
        "Source":["You","Competitor"],
        "Length":[
            your_metrics["Title Length"],
            comp_metrics["Title Length"]
        ]
    })

    sns.barplot(data=df1,x="Source",y="Length",palette="viridis",ax=ax1)

    st.pyplot(fig1)

    st.write("Your Title Length:",your_metrics["Title Length"])
    st.write("Competitor Avg:",round(comp_metrics["Title Length"],2))

    # Description Words

    fig2,ax2=plt.subplots()

    df2=pd.DataFrame({
        "Source":["You","Competitor"],
        "Words":[
            your_metrics["Description Words"],
            comp_metrics["Description Words"]
        ]
    })

    sns.barplot(data=df2,x="Source",y="Words",palette="magma",ax=ax2)

    st.pyplot(fig2)

    st.write("Your Description Words:",your_metrics["Description Words"])
    st.write("Competitor Avg:",round(comp_metrics["Description Words"],2))

# ------------------------------------------------
# AI GENERATION
# ------------------------------------------------

def generate_content(keyword):

    prompt = f"""
You are an SEO expert.

Generate an SEO optimized title and meta description.

Keyword: {keyword}

Rules:
Title under 60 characters
Description around 150 characters

Output format:

TITLE:
DESCRIPTION:
"""

    result = generator(prompt,max_length=120)

    text=result[0]["generated_text"]

    title=""
    desc=""

    if "TITLE:" in text:
        title=text.split("TITLE:")[1].split("DESCRIPTION:")[0].strip()

    if "DESCRIPTION:" in text:
        desc=text.split("DESCRIPTION:")[1].strip()

    return title,desc

# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------

if page=="Home":

    st.header("AI Powered SEO Content Intelligence")

    st.write("""
This platform combines Machine Learning, Explainable AI,
Competitor Intelligence and Generative AI to optimize content.
""")

# ------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------

elif page=="About":

    st.header("About")

    st.write("""
AI SEO Content Intelligence Platform built with:

• Machine Learning  
• Explainable AI (SHAP)  
• Competitor Intelligence  
• Generative AI  
""")

# ------------------------------------------------
# SEO ANALYZER
# ------------------------------------------------

elif page=="SEO Analyzer":

    st.header("SEO Content Analyzer")

    with st.form("seo_form"):

        title=st.text_input("Title")
        keyword=st.text_input("Keyword")
        description=st.text_area("Description")

        analyze=st.form_submit_button("Analyze SEO")

    if analyze:

        features=extract_features(title,description,keyword)

        input_df=pd.DataFrame(features,columns=feature_names)

        prob=model.predict_proba(input_df)[0][1]

        score=int(prob*100)

        st.session_state["score"]=score
        st.session_state["title"]=title
        st.session_state["keyword"]=keyword
        st.session_state["description"]=description

    if "score" in st.session_state:

        score=st.session_state["score"]

        seo_score_meter(score)

        keyword_difficulty_meter(st.session_state["keyword"])

        serp_rank_meter(score)

        st.subheader("Enter Competitor URLs")

        st.info("If a site blocks scraping results may be limited.")

        with st.form("competitor_form"):

            url1=st.text_input("Competitor URL 1")
            url2=st.text_input("Competitor URL 2")
            url3=st.text_input("Competitor URL 3")

            analyze_comp=st.form_submit_button("Analyze Competitors")

        if analyze_comp:

            urls=[url1,url2,url3]

            texts=scrape_pages(urls)

            your_metrics=calculate_metrics(
                st.session_state["title"],
                st.session_state["description"],
                st.session_state["keyword"]
            )

            comp_metrics=competitor_metrics(texts)

            st.subheader("SEO Metric Comparison")

            show_seo_charts(your_metrics,comp_metrics)

            opt_title,opt_desc=generate_content(st.session_state["keyword"])

            st.subheader("Optimized Title")
            st.success(opt_title)

            st.subheader("Optimized Description")

            st.info(opt_desc)
