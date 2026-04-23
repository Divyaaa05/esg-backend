import os
import json
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("NEWS_API_KEY", "")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

ESG_KNOWLEDGE_BASE = [
    """ESG stands for Environmental, Social, and Governance. It evaluates
    a company sustainability and ethical impact. Environmental factors include carbon emissions,
    energy usage, water consumption, waste management, and biodiversity impact. Social factors
    include labor practices, employee welfare, community relations, diversity and inclusion,
    and human rights in supply chains. Governance factors include board structure, executive
    compensation, shareholder rights, transparency, and anti-corruption policies.""",

    """Scope 3 emissions are indirect greenhouse gas emissions in a company value
    chain, representing 70-90% of total carbon footprint. Upstream includes purchased goods,
    capital goods, transportation and business travel. Downstream includes use of sold products,
    end-of-life treatment, and leased assets.""",

    """Double Materiality from EU CSRD requires companies to report both financial materiality
    how ESG risks affect company finances and impact materiality how the company affects
    people and the environment. More comprehensive than US single materiality frameworks.""",

    """UN SDGs are 17 global goals adopted in 2015. Key corporate ESG SDGs: SDG 7 Clean Energy,
    SDG 8 Decent Work, SDG 9 Industry and Innovation, SDG 12 Responsible Consumption,
    SDG 13 Climate Action, SDG 15 Life on Land, SDG 17 Partnerships.""",

    """Paris Agreement requires limiting global warming to 1.5 degrees Celsius. Companies
    must reduce emissions 45% by 2030 and reach net-zero by 2050. SBTi helps companies
    set science-based emissions reduction targets.""",

    """Biodiversity risk covers company impact on ecosystems. TNFD framework helps assess
    nature-related risks including deforestation, water pollution, land use change.""",

    """BRSR Business Responsibility and Sustainability Report is mandatory for top 1000
    listed Indian companies by SEBI since FY2022-23. Covers ethics, employee wellbeing,
    environment, human rights, and stakeholder engagement.""",

    """Greenwashing means misleading environmental claims. EU Green Claims Directive cracks
    down on unsubstantiated claims. ESG rating agencies penalize greenwashing companies.""",
]

embeddings_model = None
vector_store = None

def initialize_rag():
    global embeddings_model, vector_store
    print("Initializing RAG pipeline...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []
    for i, text in enumerate(ESG_KNOWLEDGE_BASE):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": f"esg_{i}"}))
    vector_store = FAISS.from_documents(documents, embeddings_model)
    print("RAG pipeline ready.")

def retrieve_context(query, k=3):
    global vector_store
    if vector_store is None:
        initialize_rag()
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_esg_analysis(company_name, scraped_news, sentiment_data):
    rag_context = retrieve_context(
        f"ESG analysis {company_name} scope 3 SDG controversy"
    )

    news_context = ""
    if scraped_news:
        news_titles = [a.get("title", "") for a in scraped_news[:5] if a.get("title")]
        news_context = "Recent news:\n" + "\n".join(f"- {t}" for t in news_titles)

    rt_score = sentiment_data.get("realtime_controversy", 50)
    rt_level = sentiment_data.get("controversy_level", "medium")

    prompt = f"""You are an expert ESG analyst. Analyze the ESG profile of "{company_name}".

RAG KNOWLEDGE BASE:
{rag_context}

REAL-TIME NEWS:
{news_context if news_context else "No recent news - use your knowledge."}

SENTIMENT DATA:
- Articles found: {sentiment_data.get("articles_found", 0)}
- Negative signals: {sentiment_data.get("negative_signals", 0)}
- Positive signals: {sentiment_data.get("positive_signals", 0)}
- RT Controversy score: {rt_score}/100
- Level: {rt_level}

Return ONLY valid raw JSON with no markdown and no extra text:
{{"company":"full official name","ticker":"ticker or empty string","sector":"Industry dot Sub-sector","esg_score":72,"sdg_score":68,"controversy_index":75,"financial_materiality":80,"double_materiality":65,"scope3_score":55,"biodiversity_score":60,"realtime_controversy":{rt_score},"controversy_level":"{rt_level}","controversy_summary":"2-3 sentences about controversies","sdg_goals":[7,9,13],"setA_total":74,"setB_total":60,"rag_summary":"4-5 sentence expert ESG analysis","strengths":["strength 1","strength 2","strength 3"],"risks":["risk 1","risk 2","risk 3"],"last_updated":"live"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()

    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON found in AI response")

    return json.loads(text[start:end])