# ============================================================================
# PROJET: Système de recommandation de carrières basé sur les compétences
# ÉTAPE 1: Préparation et nettoyage des données - VERSION CORRIGÉE
# ============================================================================

import pandas as pd 
import re 
import spacy
from spacy.matcher import PhraseMatcher
import wordninja

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

df = pd.read_csv("clean_jobs.csv")

print(f" Nombre initial de jobs: {len(df)}")
print(f" NaN dans 'title' avant nettoyage: {df['title'].isna().sum()}")
print(f" NaN dans 'description' avant nettoyage: {df['description'].isna().sum()}")

# ============================================================================
# 2. SUPPRESSION DES COLONNES INUTILES
# ============================================================================

df1 = df.drop(columns=["link", "source", "date_posted", "work_type", "employment_type"]) 

# ============================================================================
# 3. PREMIÈRE SUPPRESSION DES NaN CRITIQUES (avant transformation)
# ============================================================================
# On supprime d'abord les lignes où title OU description sont NaN

df1 = df1.dropna(subset=['title', 'description'])
print(f" Après suppression NaN initiaux: {len(df1)} jobs restants")

# ============================================================================
# 4. NETTOYAGE TEXTUEL
# ============================================================================

columns = ["title", "company", "location", "description"]

for col in columns:
    # Conversion en string d'abord pour éviter les erreurs
    df1[col] = df1[col].astype(str)
    # Remplacement des 'nan' en string par des chaînes vides
    df1[col] = df1[col].replace('nan', '')
    # Conversion en minuscules
    df1[col] = df1[col].str.lower()
    # Suppression caractères spéciaux
    df1[col] = df1[col].str.replace(r'[^a-z0-9 ]', '', regex=True)
    # Suppression des espaces multiples
    df1[col] = df1[col].str.replace(r'\s+', ' ', regex=True).str.strip()

# ============================================================================
# 5. DEUXIÈME NETTOYAGE : Supprimer les valeurs vides après transformation
# ============================================================================

# Remplacer les chaînes vides par NaN
df1['title'] = df1['title'].replace('', pd.NA)
df1['description'] = df1['description'].replace('', pd.NA)

# Supprimer les NaN
df1 = df1.dropna(subset=['title', 'description'])

print(f" Après nettoyage textuel: {len(df1)} jobs restants")

# ============================================================================
# 6. SUPPRESSION DES DOUBLONS
# ============================================================================

df1 = df1.drop_duplicates()
print(f" Après suppression doublons: {len(df1)} jobs restants")

# ============================================================================
# 7. SUPPRESSION DES STOPWORDS
# ============================================================================

stop_words = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 
    'between', 'both', 'but', 'by', 'could', 'did', 'do', 'does', 'doing', 'down', 
    'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 
    'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 
    'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 
    'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 
    'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 
    'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 
    'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 
    'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 
    'yours', 'yourself', 'yourselves'
}

df1["description"] = df1["description"].apply(
    lambda x: " ".join([w for w in str(x).split() if w not in stop_words])
)
df1["title"] = df1["title"].apply(
    lambda x: " ".join([w for w in str(x).split() if w not in stop_words])
)

# ============================================================================
# 8. EXTRACTION AUTOMATIQUE DES COMPÉTENCES (NLP)
# ============================================================================

skills = [
    # Analyse statistique et data science
    "exploratory data analysis", "statistical modeling",
    "linear regression and logistic regression", "linear regression", "logistic regression",
    "correlation and variance analysis", "anova", 
    "hypothesis testing and statistical inference", "time series",
    "cohort analysis and segmentation", "testing and experimentation",
    "data storytelling and insight interpretation",
    
    # Langages de programmation
    "python", "tensorflow", "numpy", "pandas", "seaborn", "matplotlib", "pyspark", "pytorch", "scikit-learn",
    "r", "ggplot2", "dplyr", "caret",
    "sql", "postgresql", "mysql", "bigquery", "snowflake", "redshift",
    "sas", "spss", "matlab", "scala", "java", "c++", "bash", "shell scripting",
    
    # Outils de visualisation et BI
    "power bi", "tableau", "looker", "looker studio", "qlikview", "qlik sense", "google data studio",
    "advanced excel", "power query", "vba macros",
    "metabase", "superset", "plotly", "dash", "streamlit",
    
    # Cloud et infrastructure
    "aws s3", "aws redshift", "aws glue", "aws sagemaker", "aws athena", "aws",
    "gcp bigquery", "gcp dataflow", "gcp vertex ai", "gcp",
    "azure synapse", "azure data factory", "azure ml studio", "azure",
    "databricks", "snowflake", "hadoop", "hive",
    
    # Big data et data engineering
    "spark pyspark", "spark scala", "spark",
    "kafka", "airflow", "prefect", "dagster",
    "etl pipeline design", "data warehousing", "data lakes",
    
    # Machine learning et IA
    "data cleaning and preparation", "feature engineering", "model selection and evaluation",
    "supervised and unsupervised learning",
    "nlp", "natural language processing", "computer vision", "recommendation systems",
    "mlops", "mlflow", "docker", "cicd",
    "deep learning", "machine learning", "dl", "ml",
    "automl", "llms", "hugging face", "langchain", "llamaindex",
    "deep learning cnn rnn transformers",
    "llms and generative ai", "generative ai", "openai", "llama",
    "finetuning", "prompt engineering", "reinforcement learning",
    "computer vision yolo resnet", "yolo", "resnet",
    "speech analysis", "audio analysis", "ai ethics", "explainable ai xai",
    
    # DevOps et développement
    "git", "github", "gitlab", "cicd", "jenkins", "docker", "kubernetes",
    "apis rest", "graphql", "api integration rest and graphql", "api rest", "api graphql",
    "unit testing", "integration testing",
    "documentation swagger postman", "swagger", "postman",
    "dependency management", "code versioning",
    
    # Compétences métier et analytics
    "schema management and data quality",
    "real-time data processing kafka flink", "flink",
    "sql optimization and indexing",
    "infrastructure as code terraform cloudformation", "terraform", "cloud", "cloudformation",
    "product analytics", "marketing analytics", "financial analytics", "pricing analytics",
    "healthcare analytics", "fraud detection", "risk modeling", "customer insights",
    "crm analytics", "operations analytics", "supply chain analytics",
    "esg data", "sustainability data", "people analytics", "hr analytics",
    
    # Soft skills et méthodologies
    "understanding business kpis", "agile project management", "scrum",
    "cross-team collaboration", "stakeholder communication",
    "storytelling", "data visualization", "clear documentation",
    "powerpoint presentations", "dax", "regex",
    "text parsing", "web scraping", "beautifulsoup", "selenium",
    "excel macros", "automation",
    "statistical process control", "spc",
    "gdpr compliance", "data governance", "data security and anonymization",
    "consulting", "digital consulting", "big data"
]

nlp = spacy.load("en_core_web_sm")

matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(skill) for skill in skills]
matcher.add("SKILLS", patterns)

skills_found = []

print("🔍 Extraction des compétences en cours...")

for desc in df1["description"]:
    doc = nlp(str(desc))
    matches = matcher(doc)
    
    seen = set()
    skills_in_desc = []
    
    for match_id, start, end in matches:
        skill = doc[start:end].text
        if skill not in seen:
            skills_in_desc.append(skill)
            seen.add(skill)
    
    skills_found.append(skills_in_desc)

df1["skills"] = skills_found
df1["skills_text"] = df1["skills"].apply(lambda x: " ".join(x))

# ============================================================================
# 9. NETTOYAGE DU TITLE AVEC WORDNINJA
# ============================================================================

def clean_title(row):
    try:
        title_words = wordninja.split(str(row["title"]))
        exclusions = set(str(row["location"]).split() + 
                        str(row["company"]).split() + 
                        str(row["skills_text"]).split())
        cleaned = " ".join(word for word in title_words if word not in exclusions)
        # Si le titre devient vide, on garde le titre original
        return cleaned if cleaned.strip() else str(row["title"])
    except:
        return str(row["title"])

df1["title"] = df1.apply(clean_title, axis=1)

# ============================================================================
# 10. NETTOYAGE FINAL : Supprimer les titles vides après wordninja
# ============================================================================

df1['title'] = df1['title'].str.strip()
df1['title'] = df1['title'].replace('', pd.NA)
df1 = df1.dropna(subset=['title'])

print(f" Après clean_title: {len(df1)} jobs restants")

# ============================================================================
# 11. CRÉATION DU PROFIL GLOBAL
# ============================================================================

df1['profile_text'] = df1['title'] + ' ' + df1['skills_text'] + ' ' + df1['description']

# ============================================================================
# 12. VÉRIFICATION FINALE
# ============================================================================

print("\n" + "="*70)
print(" RÉSUMÉ DU NETTOYAGE")
print("="*70)
print(f" Nombre final de jobs: {len(df1)}")
print(f"NaN dans 'title': {df1['title'].isna().sum()}")
print(f"NaN dans 'description': {df1['description'].isna().sum()}")
print(f" Valeurs vides dans 'title': {(df1['title'] == '').sum()}")
print(f" Valeurs vides dans 'description': {(df1['description'] == '').sum()}")
print("\n Exemple de job nettoyé:")
print(df1.iloc[0])

# ============================================================================
# 13. SAUVEGARDE
# ============================================================================

df1.to_csv("BD_nettoyée.csv", index=False)

print("\n Dataset nettoyé et sauvegardé avec succès!")







