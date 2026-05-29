import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import re
import os
from collections import Counter, defaultdict

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_NUM = {m: i+1 for i, m in enumerate(MONTHS)}

# Resolve all data paths relative to this script so the app works regardless of
# which directory `streamlit run` is invoked from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="OGC Legal AI Automation Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# LegalBench Task Categories adapted for Rimon's matter-based classification
LEGALBENCH_TASKS = {
    # CONTRACT ANALYSIS & REVIEW (High automation 85-95%)
    'Contract-Clause-Identification': {
        'description': 'Identifying and extracting specific contract clauses',
        'automation_potential': 0.92,
        'keywords': ['contract', 'agreement', 'lease', 'license', 'licensing', 'amendment',
                    'addendum', 'nda', 'msa', 'sla', 'consulting agreement', 'service agreement',
                    'vendor', 'supplier', 'procurement', 'terms', 'clause review'],
        'examples': ['Contract review', 'Lease agreements', 'Service agreements']
    },
    
    # M&A AND CORPORATE (Medium-High automation 75-90%)
    'MA-Deal-Terms': {
        'description': 'M&A deal terms analysis and acquisition matters',
        'automation_potential': 0.82,
        'keywords': ['acquisition', 'merger', 'buyout', 'purchase', 'sale', 'transaction',
                    'm&a', 'due diligence', 'closing', 'earnout', 'escrow', 'indemnit',
                    'representation and warrant', 'definitive agreement'],
        'examples': ['M&A transactions', 'Acquisitions', 'Corporate sales']
    },
    'Corporate-Governance': {
        'description': 'Corporate governance, formation, and general corporate matters',
        'automation_potential': 0.80,
        'keywords': ['general corporate', 'corporate matters', 'corporate governance', 
                    'formation', 'incorporation', 'llc', 'corporation', 'bylaws', 
                    'operating agreement', 'shareholder', 'board', 'director', 'officer',
                    'corporate advice', 'general representation', 'retainer',
                    'corporate counsel', 'spv', 'entity'],
        'examples': ['Corporate formation', 'General corporate counsel', 'Entity structuring']
    },
    'Securities-Compliance': {
        'description': 'Securities matters and compliance',
        'automation_potential': 0.78,
        'keywords': ['securities', 'sec', 'offering', 'private placement', 'regulation d',
                    'stock', 'equity', 'financing', 'investment', 'fund', 'investor'],
        'examples': ['Securities filings', 'Private placements', 'Fund matters']
    },
    
    # LITIGATION & PROCEDURE (Medium automation 70-85%)
    'Litigation-Matters': {
        'description': 'Litigation, disputes, and court proceedings',
        'automation_potential': 0.75,
        'keywords': [' v ', ' v. ', ' vs ', ' vs. ', ' versus ', 'litigation', 'lawsuit',
                    'complaint', 'dispute', 'arbitration', 'mediation', 'trial', 'hearing',
                    'motion', 'discovery', 'deposition', 'settlement', 'judgment',
                    'appeal', 'court', 'plaintiff', 'defendant'],
        'examples': ['Civil litigation', 'Commercial disputes', 'Court proceedings']
    },
    'Bankruptcy-Receivership': {
        'description': 'Bankruptcy and receivership matters',
        'automation_potential': 0.73,
        'keywords': ['bankruptcy', 'chapter 11', 'chapter 7', 'receiver', 'receivership',
                    'creditor', 'debtor', 'insolvency', 'reorganization', 'liquidation'],
        'examples': ['Bankruptcy proceedings', 'Receivership matters', 'Creditor rights']
    },
    
    # EMPLOYMENT & HR (Medium-High automation 75-85%)
    'Employment-Law': {
        'description': 'Employment contracts, disputes, and HR matters',
        'automation_potential': 0.80,
        'keywords': ['employment', 'employee', 'hr ', 'human resources', 'labor',
                    'termination', 'severance', 'discrimination', 'harassment',
                    'wage', 'compensation', 'benefits', 'non-compete', 'restrictive covenant',
                    'wrongful termination', 'employment agreement'],
        'examples': ['Employment agreements', 'HR compliance', 'Wrongful termination']
    },
    
    # REAL ESTATE (Medium automation 70-80%)
    'Real-Estate': {
        'description': 'Real estate transactions and property matters',
        'automation_potential': 0.75,
        'keywords': ['real estate', 'property', 'lease', 'landlord', 'tenant', 'rental',
                    'commercial lease', 'retail lease', 'office lease', 'title',
                    'escrow', 'deed', 'mortgage', 'foreclosure', 'easement', 'zoning'],
        'examples': ['Lease negotiations', 'Property acquisitions', 'Real estate closings']
    },
    
    # INTELLECTUAL PROPERTY (High automation 85-90%)
    'Intellectual-Property': {
        'description': 'IP matters including patents, trademarks, and copyrights',
        'automation_potential': 0.85,
        'keywords': ['patent', 'trademark', 'copyright', 'intellectual property', ' ip ',
                    'infringement', 'licensing', 'royalty', 'trade secret', 'confidential',
                    'epo', 'uspto', 'office action', 'prosecution', 'portfolio'],
        'examples': ['Patent prosecution', 'Trademark filing', 'IP licensing']
    },
    
    # ESTATE PLANNING & TRUSTS (High automation 80-90%)
    'Estate-Planning': {
        'description': 'Estate planning, wills, trusts, and probate',
        'automation_potential': 0.88,
        'keywords': ['estate planning', 'estate', 'trust', 'will', 'probate',
                    'trustee', 'beneficiary', 'inheritance', 'succession', 'gift',
                    'estate tax', 'generation skipping', 'living trust', 'testamentary',
                    'administration', 'executor', 'fiduciary'],
        'examples': ['Estate planning', 'Trust administration', 'Will preparation']
    },
    
    # FAMILY LAW (Medium automation 65-75%)
    'Family-Law': {
        'description': 'Divorce, custody, and family law matters',
        'automation_potential': 0.68,
        'keywords': ['divorce', 'dissolution', 'marriage', 'custody', 'child support',
                    'alimony', 'spousal support', 'marital', 'family law', 'prenup',
                    'postnup', 'separation', 'domestic', 'parenting', 'visitation'],
        'examples': ['Divorce proceedings', 'Custody matters', 'Support calculations']
    },
    
    # TAX (Medium automation 70-80%)
    'Tax-Law': {
        'description': 'Tax planning, compliance, and disputes',
        'automation_potential': 0.75,
        'keywords': ['tax', 'irs', 'taxation', 'tax planning', 'tax compliance',
                    'tax return', 'audit', 'tax dispute', 'tax opinion', 'tax structure'],
        'examples': ['Tax planning', 'IRS disputes', 'Tax compliance']
    },
    
    # IMMIGRATION (Medium automation 70-80%)
    'Immigration-Law': {
        'description': 'Immigration and visa matters',
        'automation_potential': 0.78,
        'keywords': ['immigration', 'visa', 'h-1b', 'green card', 'citizenship',
                    'naturalization', 'deportation', 'asylum', 'refugee', 'work permit',
                    'uscis', 'ice', 'border', 'immigrant'],
        'examples': ['Visa applications', 'Immigration compliance', 'Citizenship matters']
    },
    
    # REGULATORY & COMPLIANCE (High automation 80-90%)
    'Regulatory-Compliance': {
        'description': 'Regulatory compliance and government affairs',
        'automation_potential': 0.83,
        'keywords': ['regulatory', 'compliance', 'regulation', 'permit', 'license',
                    'government', 'agency', 'fda', 'epa', 'osha', 'ftc', 'fcc',
                    'administrative', 'rulemaking', 'enforcement', 'investigation'],
        'examples': ['Regulatory compliance', 'Government permits', 'Agency matters']
    },
    
    # CANNABIS (Medium automation 70-80%)
    'Cannabis-Law': {
        'description': 'Cannabis industry legal matters',
        'automation_potential': 0.72,
        'keywords': ['cannabis', 'marijuana', 'dispensary', 'cultivation', 'cbd',
                    'thc', 'hemp', 'marijuana license', 'cannabis license'],
        'examples': ['Cannabis licensing', 'Dispensary operations', 'Cannabis compliance']
    },
    
    # HEALTHCARE (Medium automation 70-80%)
    'Healthcare-Law': {
        'description': 'Healthcare and medical law matters',
        'automation_potential': 0.74,
        'keywords': ['healthcare', 'medical', 'hospital', 'physician', 'hipaa',
                    'health insurance', 'medicare', 'medicaid', 'pharmaceutical'],
        'examples': ['Healthcare compliance', 'Medical practice matters', 'HIPAA compliance']
    },
    
    # ADMINISTRATIVE & ROUTINE (Very High automation 90-95%)
    'General-Matters': {
        'description': 'General advice, consultation, and miscellaneous matters',
        'automation_potential': 0.65,
        'keywords': ['general matters', 'general legal advice', 'general corporate',
                    'miscellaneous', 'various', 'other matters'],
        'examples': ['General legal advice', 'Miscellaneous matters']
    },
    
    # INTERNAL TIME (0% automation - not client work)
    'Internal-Time': {
        'description': 'Internal firm time - administrative and non-billable',
        'automation_potential': 0.00,
        'keywords': ['internal time', 'internal', 'admin', 'administrative', 'training',
                    'business development', 'marketing', 'firm', 'vacation', 'pto',
                    'sick', 'holiday'],
        'examples': ['Internal meetings', 'Training', 'Administrative tasks']
    }
}

# Rimon-specific OLI Benchmark
RIMON_OLI_BENCHMARK = {
    '100% AI Replaceable - Routine Corporate & Documents': {
        'automation_potential': 1.00,
        'keywords': [
            'general corporate', 'corporate matters', 'general representation',
            'advice and counsel', 'corporate advice', 'general corporate advice',
            'general legal advice', 'legal advice',
            'nda review', 'contract review', 'agreement review', 'contract reviews',
            'agreement reviews', 'document review', 'template review',
            'standard agreement', 'routine agreement', 'form review'
        ],
        'description': 'Routine corporate counsel and document review',
        'examples': ['General corporate matters', 'NDA Review', 'Contract review']
    },
    '100% AI Replaceable - Estate Planning': {
        'automation_potential': 1.00,
        'keywords': [
            'estate planning', 'estate', 'trust', 'will', 'probate',
            'trust administration'
        ],
        'description': 'Estate planning documents and trust administration - highly templated',
        'examples': ['Estate planning', 'Trust documents', 'Will preparation']
    },
    '70% AI Replaceable - Transactional Work': {
        'automation_potential': 0.70,
        'keywords': [
            'acquisition', 'merger', 'purchase', 'sale', 'transaction',
            'financing', 'investment', 'fund', 'securities',
            'lease', 'real estate', 'property',
            'nda', 'commercial', 'vendor', 'software', 'licensing agreement',
            'service agreement', 'supply', 'procurement', 'corporate transaction',
            'asset purchase', 'asset management', 'equity', 'joint venture',
            'partnership agreement', 'monetization', 'distribution agreement'
        ],
        'description': 'M&A, financing, and complex transactional work',
        'examples': ['Acquisitions', 'Financings', 'Real estate transactions', 'NDA drafting']
    },
    '70% AI Replaceable - IP & Regulatory': {
        'automation_potential': 0.70,
        'keywords': [
            'patent', 'trademark', 'copyright', 'intellectual property', 'intellectual prop',
            'ip counsel', 'ip advice', 'ip matters', 'trade secret',
            'regulatory', 'compliance', 'permit', 'license',
            'immigration', 'visa',
            'pharma', 'pharmaceutical', 'clinical', 'biotech', 'drug',
            'data protection', 'privacy', 'dpo', 'gdpr', 'ccpa',
            'trademark search', 'tm search', 'patent search', 'brand'
        ],
        'description': 'IP prosecution, regulatory compliance, and data/pharma matters',
        'examples': ['Patent filings', 'Trademark prosecution', 'Pharma regulatory', 'Privacy compliance']
    },
    '30% AI Replaceable - Litigation & Complex Matters': {
        'automation_potential': 0.30,
        'keywords': [
            ' v ', ' v. ', ' vs ', ' vs. ', 'litigation', 'lawsuit', 'dispute',
            'arbitration', 'trial', 'court', 'motion', 'discovery',
            'bankruptcy', 'receiver', 'settlement'
        ],
        'description': 'Litigation and complex disputes requiring significant judgment',
        'examples': ['Civil litigation', 'Arbitrations', 'Court proceedings']
    },
    '30% AI Replaceable - Employment & Family Law': {
        'automation_potential': 0.30,
        'keywords': [
            'employment', 'hr ', 'labor', 'termination',
            'divorce', 'dissolution', 'custody', 'family law',
            'tax'
        ],
        'description': 'Matters requiring nuanced human judgment and counseling',
        'examples': ['Employment disputes', 'Divorce matters', 'Tax planning']
    },
    '0% AI Replaceable - Internal & Strategic': {
        'automation_potential': 0.00,
        'keywords': [
            'internal time', 'vacation', 'pto', 'holiday', 'training',
            'business development', 'marketing', 'admin',
            'g&a', 'general and administrative', 'overhead'
        ],
        'description': 'Internal time and strategic work',
        'examples': ['Internal meetings', 'Business development', 'Training', 'G&A']
    }
}


# ============================================================================
# TASK-LEVEL CLASSIFICATION (for detailed description CSV)
# ============================================================================
TASK_LEVEL_AUTOMATION = {
    'Document-Review-Standard': {'description': 'Review of standard documents', 'automation_potential': 0.95, 'keywords': ['review agreement', 'review contract', 'review and revise', 'review draft', 'review standard', 'review form', 'review template', 'review nda', 'review msa', 'review amendment', 'review lease']},
    'Email-Status-Updates': {'description': 'Status update emails', 'automation_potential': 0.92, 'keywords': ['email regarding status', 'email correspondence regarding', 'status update', 'follow up email', 'exchange emails', 'email to client regarding status']},
    'Document-Drafting-Standard': {'description': 'Drafting standard forms', 'automation_potential': 0.93, 'keywords': ['draft amendment', 'draft addendum', 'draft standard', 'draft form', 'draft certificate', 'draft notice', 'draft letter agreement', 'prepare draft amendment']},
    'Research-Straightforward': {'description': 'Straightforward legal research', 'automation_potential': 0.88, 'keywords': ['research case law', 'research statute', 'research regulation', 'research precedent', 'legal research regarding']},
    'Form-Completion': {'description': 'Completing forms', 'automation_potential': 0.96, 'keywords': ['complete form', 'fill out', 'prepare filing', 'file notice', 'file certificate', 'submit form']},
    'Document-Analysis': {'description': 'Analyzing documents', 'automation_potential': 0.85, 'keywords': ['review and analyze', 'analyze agreement', 'analyze contract', 'analyze terms', 'analyze provision', 'review for compliance', 'analyze draft']},
    'Due-Diligence-Review': {'description': 'Due diligence review', 'automation_potential': 0.82, 'keywords': ['due diligence', 'dd review', 'review due diligence', 'diligence materials', 'data room review']},
    'Discovery-Review': {'description': 'Document discovery review', 'automation_potential': 0.87, 'keywords': ['review discovery', 'review production', 'review interrogator', 'review request for production', 'discovery response', 'respond to discovery']},
    'Clause-Extraction': {'description': 'Extracting specific clauses', 'automation_potential': 0.90, 'keywords': ['extract provision', 'identify clause', 'locate language', 'find provision', 'pull clause', 'summarize terms']},
    'Drafting-Complex': {'description': 'Drafting complex agreements', 'automation_potential': 0.65, 'keywords': ['draft purchase agreement', 'draft psa', 'draft merger agreement', 'draft complex', 'draft financing', 'draft loan', 'draft settlement']},
    'Negotiation-Support': {'description': 'Supporting negotiations', 'automation_potential': 0.60, 'keywords': ['revise per comments', 'address comments', 'incorporate revisions', 'revise based on', 'respond to comments', 'counter proposal']},
    'Legal-Memos': {'description': 'Legal memoranda', 'automation_potential': 0.55, 'keywords': ['draft memo', 'memorandum', 'legal opinion', 'prepare memo', 'draft analysis', 'memo regarding']},
    'Client-Calls': {'description': 'Client communications', 'automation_potential': 0.45, 'keywords': ['call with client', 'telephone conference with', 'conference call', 'client meeting', 'discuss with client', 'call regarding', 'phone call']},
    'Court-Appearances': {'description': 'Court appearances', 'automation_potential': 0.30, 'keywords': ['court appearance', 'appear in court', 'attend hearing', 'oral argument', 'trial', 'deposition', 'attend conference']},
    'Strategic-Advice': {'description': 'Strategic legal counseling', 'automation_potential': 0.35, 'keywords': ['advise regarding', 'counsel regarding', 'discuss strategy', 'strategic advice', 'recommendation regarding', 'consult on']},
    'Negotiations': {'description': 'Negotiation sessions', 'automation_potential': 0.25, 'keywords': ['negotiate', 'negotiation', 'negotiating', 'negotiate terms', 'negotiate with', 'settlement discussion']},
    'Internal-Admin': {'description': 'Internal administrative tasks', 'automation_potential': 0.10, 'keywords': ['internal', 'administrative', 'firm meeting', 'training', 'business development', 'marketing', 'time entry', 'billing']},
    'General-Communication': {'description': 'General correspondence', 'automation_potential': 0.50, 'keywords': ['email', 'correspondence', 'communicate', 'discuss', 'exchange', 'speak with', 'follow up']}
}

def classify_task_description(description):
    """Classify based on detailed task description"""
    if pd.isna(description):
        return 'General-Communication', 0.50
    desc_lower = description.lower()
    scores = {}
    for category, info in TASK_LEVEL_AUTOMATION.items():
        score = sum(1 for keyword in info['keywords'] if keyword in desc_lower)
        if score > 0:
            scores[category] = score
    if scores:
        best_category = max(scores, key=scores.get)
        return best_category, TASK_LEVEL_AUTOMATION[best_category]['automation_potential']
    return 'General-Communication', 0.50

# ── Direct Practice-Group → classification maps ───────────────────────────────
# Used for 2025/2026 synthesized data where Matter Name is the attorney's PG label.
# Keyword matching on PG names produces false positives (e.g. "Commercial Real Estate"
# matches the "estate" keyword → incorrectly classified as 100% Estate Planning).
# These explicit maps give accurate, defensible tier assignments.
PG_LEGALBENCH_DIRECT = {
    'commercial & tech transactions': ('Contract-Clause-Identification', 0.92),
    'corporate & finance':            ('Corporate-Governance', 0.80),
    'life sciences and healthcare':   ('Healthcare-Law', 0.74),
    'intellectual property':          ('Intellectual-Property', 0.85),
    'employment, labor & immigration':('Employment-Law', 0.80),
    'cyber security & privacy':       ('Regulatory-Compliance', 0.83),
    'retail, marketing & media':      ('Contract-Clause-Identification', 0.92),
    'federal government contracts & procurement': ('Regulatory-Compliance', 0.83),
    'exempt organizations & higher education':    ('Corporate-Governance', 0.80),
    'commercial real estate':         ('Real-Estate', 0.75),
    'dispute resolution':             ('Litigation-Matters', 0.75),
    'tax':                            ('Tax-Law', 0.75),
    'family law':                     ('Family-Law', 0.68),
    'estate planning':                ('Estate-Planning', 0.88),
    'securities':                     ('Securities-Compliance', 0.78),
    'immigration':                    ('Immigration-Law', 0.78),
}

PG_OLI_DIRECT = {
    'commercial & tech transactions': ('70% AI Replaceable - Transactional Work', 0.70),
    'corporate & finance':            ('70% AI Replaceable - Transactional Work', 0.70),
    'life sciences and healthcare':   ('70% AI Replaceable - IP & Regulatory', 0.70),
    'intellectual property':          ('70% AI Replaceable - IP & Regulatory', 0.70),
    'employment, labor & immigration':('30% AI Replaceable - Employment & Family Law', 0.30),
    'cyber security & privacy':       ('70% AI Replaceable - IP & Regulatory', 0.70),
    'retail, marketing & media':      ('70% AI Replaceable - Transactional Work', 0.70),
    'federal government contracts & procurement': ('70% AI Replaceable - IP & Regulatory', 0.70),
    'exempt organizations & higher education':    ('70% AI Replaceable - Transactional Work', 0.70),
    'commercial real estate':         ('70% AI Replaceable - Transactional Work', 0.70),
    'dispute resolution':             ('30% AI Replaceable - Litigation & Complex Matters', 0.30),
    'tax':                            ('30% AI Replaceable - Employment & Family Law', 0.30),
    'family law':                     ('30% AI Replaceable - Employment & Family Law', 0.30),
    'estate planning':                ('100% AI Replaceable - Estate Planning', 1.00),
    'securities':                     ('70% AI Replaceable - Transactional Work', 0.70),
    'immigration':                    ('70% AI Replaceable - IP & Regulatory', 0.70),
}
# ─────────────────────────────────────────────────────────────────────────────


def classify_matter_legalbench(matter_name, pg_fallback: str = ''):
    """
    Classify a matter using LegalBench framework.
    If the matter name does not match any keyword, fall back to the attorney's
    Practice Group label (pg_fallback) which is mapped via PG_LEGALBENCH_DIRECT.
    """
    if pd.isna(matter_name):
        # Try PG fallback before giving up
        if pg_fallback:
            pg_lower = str(pg_fallback).lower().strip()
            if pg_lower in PG_LEGALBENCH_DIRECT:
                return PG_LEGALBENCH_DIRECT[pg_lower]
        return 'General-Matters', 0.65

    matter_lower = str(matter_name).lower().strip()

    # Direct PG mapping takes priority (avoids false keyword matches on PG labels)
    if matter_lower in PG_LEGALBENCH_DIRECT:
        return PG_LEGALBENCH_DIRECT[matter_lower]

    # Score each category by keyword hits
    scores = {}
    for category, info in LEGALBENCH_TASKS.items():
        score = sum(1 for keyword in info['keywords'] if keyword in matter_lower)
        if score > 0:
            scores[category] = score

    if scores:
        best_category = max(scores, key=scores.get)
        automation_potential = LEGALBENCH_TASKS[best_category]['automation_potential']
        return best_category, automation_potential

    # No keyword match — fall back to attorney's Practice Group
    if pg_fallback:
        pg_lower = str(pg_fallback).lower().strip()
        if pg_lower in PG_LEGALBENCH_DIRECT:
            return PG_LEGALBENCH_DIRECT[pg_lower]

    return 'General-Matters', 0.65


def classify_matter_oli(matter_name, pg_fallback: str = ''):
    """
    Classify a matter using OLI Benchmark.
    If the matter name does not match any keyword, fall back to the attorney's
    Practice Group label (pg_fallback) which is mapped via PG_OLI_DIRECT.
    """
    if pd.isna(matter_name):
        if pg_fallback:
            pg_lower = str(pg_fallback).lower().strip()
            if pg_lower in PG_OLI_DIRECT:
                return PG_OLI_DIRECT[pg_lower]
        return 'Unclassified / General Retainer', 0.40

    matter_lower = str(matter_name).lower().strip()

    # Direct PG mapping takes priority (avoids false keyword matches on PG labels)
    if matter_lower in PG_OLI_DIRECT:
        return PG_OLI_DIRECT[matter_lower]

    # Check for internal time first (0% automation)
    if 'internal time' in matter_lower or 'vacation' in matter_lower:
        return '0% AI Replaceable - Internal & Strategic', 0.0

    # Score each category
    scores = {}
    for category, info in RIMON_OLI_BENCHMARK.items():
        if category == '0% AI Replaceable - Internal & Strategic':
            continue
        score = sum(1 for keyword in info['keywords'] if keyword in matter_lower)
        if score > 0:
            scores[category] = score

    if scores:
        best_category = max(scores, key=scores.get)
        automation_potential = RIMON_OLI_BENCHMARK[best_category]['automation_potential']
        return best_category, automation_potential

    # No keyword match — fall back to attorney's Practice Group
    if pg_fallback:
        pg_lower = str(pg_fallback).lower().strip()
        if pg_lower in PG_OLI_DIRECT:
            return PG_OLI_DIRECT[pg_lower]

    # Generic retainer entries with no PG: conservative 40% baseline
    return 'Unclassified / General Retainer', 0.40

@st.cache_data
def load_raw_year(csv_path: str) -> pd.DataFrame:
    """Load and preprocess a raw SIX_FULL_MOS style CSV (2024 format)."""
    df = pd.read_csv(csv_path, low_memory=False)
    df['Service Date'] = pd.to_datetime(df['Service Date'], format='%m/%d/%y', errors='coerce')
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce').fillna(0)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Year'] = df['Service Date'].dt.year
    df['Month'] = df['Service Date'].dt.month
    df['Month_Name'] = df['Service Date'].dt.strftime('%B')
    df['Quarter'] = df['Service Date'].dt.quarter
    df['Matter Name'] = df['Matter Name'].fillna('Unknown')
    df['Description'] = df['Description'].fillna('Unknown')
    df['Original_Hours'] = df['Hours'].copy()
    df.loc[df['Activity Type'] == 'Fixed Fee', 'Hours'] = 1.0
    df['Date of Work'] = df['Service Date']
    df['Billable Hours'] = df['Hours']
    df['Billable Amount'] = df['Amount']
    df['User Name'] = df['Associated Attorney']
    # Carry the Practice Group label forward as a consistent column name
    df['Attorney_PG'] = df['PG'].fillna('') if 'PG' in df.columns else ''
    df['Data_Source'] = 'raw'
    return df


@st.cache_data
def load_pivot_year(year: int) -> pd.DataFrame:
    """
    Load PIVOT_SOURCE_1 + ATTORNEY_CLIENTS for 2025 or 2026 and synthesize
    time-entry records matching the schema of the raw 2024 data.
    Hours are taken from PIVOT_SOURCE_1; billing amounts from ATTORNEY_CLIENTS
    (allocated proportionally to each client from the attorney's monthly total).
    The PG (Practice Group) field is used as the Matter Name for AI classification.
    """
    pivot_path = os.path.join(BASE_DIR, f'{year}/PIVOT_SOURCE_1_{year}.csv')
    clients_path = os.path.join(BASE_DIR, f'{year}/ATTORNEY_CLIENTS_{year}.csv')

    if not os.path.exists(pivot_path):
        return pd.DataFrame()

    # ── Load PIVOT_SOURCE_1 ──────────────────────────────────────────────────
    try:
        raw_check = pd.read_csv(pivot_path, nrows=25, header=None)
        header_row = None
        for idx, row in raw_check.iterrows():
            if 'Associated Attorney' in str(row.iloc[0]):
                header_row = idx
                break
        if header_row is None:
            return pd.DataFrame()
        pivot_df = pd.read_csv(pivot_path, skiprows=header_row, low_memory=False)
        pivot_df = pivot_df.dropna(subset=['Associated Attorney'])
        pivot_df = pivot_df[pivot_df['Associated Attorney'].astype(str).str.strip() != 'Associated Attorney']
    except Exception:
        return pd.DataFrame()

    # Build attorney-month → hours lookup
    atty_hours: dict = {}
    atty_pg: dict = {}
    for _, row in pivot_df.iterrows():
        attorney = str(row.get('Associated Attorney', '')).strip()
        if not attorney or attorney == 'nan':
            continue
        pg = str(row.get('PG', 'General Matters')).strip()
        atty_pg[attorney] = pg if pg not in ('nan', '') else 'General Matters'
        for m in MONTHS:
            if m in pivot_df.columns:
                h = pd.to_numeric(row.get(m, 0), errors='coerce')
                if pd.notna(h) and h > 0:
                    atty_hours[(attorney, MONTH_NUM[m])] = float(h)

    # ── Load ATTORNEY_CLIENTS ────────────────────────────────────────────────
    month_col_map: dict = {}
    client_records: list = []

    if os.path.exists(clients_path):
        try:
            raw_clients = pd.read_csv(clients_path, nrows=25, header=None)
            data_start = None
            for idx, row in raw_clients.iterrows():
                row_vals = [str(v) for v in row.values]
                if any(m in row_vals for m in MONTHS):
                    for m in MONTHS:
                        if m in row_vals:
                            month_col_map[m] = row_vals.index(m)
                if 'Client Name' in row_vals:
                    data_start = idx + 1
                    break

            if data_start and month_col_map:
                clients_full = pd.read_csv(
                    clients_path, skiprows=data_start, header=None, low_memory=False
                )
                for _, row in clients_full.iterrows():
                    client = str(row.iloc[0]).strip() if len(row) > 0 else ''
                    attorney = str(row.iloc[1]).strip() if len(row) > 1 else ''
                    if not client or client == 'nan' or not attorney or attorney == 'nan':
                        continue
                    for m, col_idx in month_col_map.items():
                        if col_idx < len(row):
                            amount = pd.to_numeric(row.iloc[col_idx], errors='coerce')
                            if pd.notna(amount) and amount > 0:
                                client_records.append({
                                    'client': client,
                                    'attorney': attorney,
                                    'month_num': MONTH_NUM[m],
                                    'amount': float(amount),
                                })
        except Exception:
            pass

    # ── Build synthetic records ──────────────────────────────────────────────
    records: list = []

    if client_records:
        # Proportionally allocate attorney hours across clients by billing share
        atty_month_total: dict = defaultdict(float)
        for r in client_records:
            atty_month_total[(r['attorney'], r['month_num'])] += r['amount']

        for r in client_records:
            atty = r['attorney']
            m_num = r['month_num']
            total_hours = atty_hours.get((atty, m_num), 0.0)
            total_amount = atty_month_total.get((atty, m_num), 1.0)
            hours = total_hours * (r['amount'] / total_amount) if total_amount > 0 else 0.0
            pg = atty_pg.get(atty, 'General Matters')
            service_date = pd.Timestamp(year=year, month=m_num, day=1)
            records.append({
                'Service Date': service_date,
                'Date of Work': service_date,
                'Client Name': r['client'],
                'Associated Attorney': atty,
                'User Name': atty,
                'Matter Name': pg,
                'Description': '',
                'Hours': hours,
                'Amount': r['amount'],
                'Activity Type': 'Time',
                'Year': year,
                'Month': m_num,
                'Month_Name': service_date.strftime('%B'),
                'Quarter': (m_num - 1) // 3 + 1,
                'Billable Hours': hours,
                'Billable Amount': r['amount'],
                'Original_Hours': hours,
                'Data_Source': 'pivot',
            })
    else:
        # Fallback: attorney-level records only (no client detail)
        for (attorney, month_num), hours in atty_hours.items():
            pg = atty_pg.get(attorney, 'General Matters')
            service_date = pd.Timestamp(year=year, month=month_num, day=1)
            records.append({
                'Service Date': service_date,
                'Date of Work': service_date,
                'Client Name': 'Multiple Clients',
                'Associated Attorney': attorney,
                'User Name': attorney,
                'Matter Name': pg,
                'Description': '',
                'Hours': hours,
                'Amount': 0.0,
                'Activity Type': 'Time',
                'Year': year,
                'Month': month_num,
                'Month_Name': service_date.strftime('%B'),
                'Quarter': (month_num - 1) // 3 + 1,
                'Billable Hours': hours,
                'Billable Amount': 0.0,
                'Original_Hours': hours,
                'Data_Source': 'pivot',
            })

    return pd.DataFrame(records) if records else pd.DataFrame()

@st.cache_data
def load_clio_year(xlsx_path: str) -> pd.DataFrame:
    """
    Load a Clio Activities XLSX export and return individual time-entry records
    matching the schema of the 2024 SIX_FULL_MOS data.

    Matter names are extracted from the 'Matter Number' column, which Clio formats as
    '{ID}-{Client Name}-{Matter Description}'.  The last segment (after the final '-')
    is used as the Matter Name for AI classification — e.g. 'NDA Review 2025',
    'Patent and Trademark', 'General Corporate'.

    Hours are back-calculated from billing amounts using 2024 median rates per attorney
    (same methodology as transform_clio_to_2025.py).
    """
    DEFAULT_RATE = 425.0
    ATTORNEY_NAME_MAP = {"Eddie Litton": "W. Edwin Litton"}

    if not os.path.exists(xlsx_path):
        return pd.DataFrame()

    try:
        df = pd.read_excel(xlsx_path, sheet_name='Report')
    except Exception:
        return pd.DataFrame()

    try:
        # Keep hourly and flat-rate time entries only
        time_types = {'Hourly time entry', 'Flat rate time entry'}
        df = df[df['Type'].isin(time_types)].copy()

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)

        # Normalise attorney names
        df['User'] = df['User'].astype(str).str.strip().replace(ATTORNEY_NAME_MAP)

        # Extract matter name: last '-'-delimited segment of Matter Number
        df['Matter_Name'] = (
            df['Matter Number']
            .astype(str)
            .str.rsplit('-', n=1)
            .str[-1]
            .str.strip()
        )

        # Build per-attorney rate table AND practice-group lookup from 2024 raw data
        rate_table: dict = {}
        pg_table: dict = {}
        raw_2024_path = os.path.join(BASE_DIR, '2024/SIX_FULL_MOS_2024.csv')
        if os.path.exists(raw_2024_path):
            try:
                df24 = pd.read_csv(raw_2024_path, low_memory=False)
                df24['Rate'] = pd.to_numeric(df24['Rate'], errors='coerce')
                time_rows = df24[(df24['Activity Type'] == 'Time') & (df24['Rate'] > 0)]
                rate_table = time_rows.groupby('Associated Attorney')['Rate'].median().to_dict()
                # Most common PG per attorney (mode)
                if 'PG' in df24.columns:
                    pg_rows = df24.dropna(subset=['PG'])
                    pg_table = (
                        pg_rows.groupby('Associated Attorney')['PG']
                        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '')
                        .to_dict()
                    )
            except Exception:
                pass

        rates = df['User'].map(rate_table).fillna(DEFAULT_RATE)
        rates = rates.where(rates > 0, DEFAULT_RATE)

        # Back-calculate hours
        hourly_mask = df['Type'] == 'Hourly time entry'
        flat_mask   = df['Type'] == 'Flat rate time entry'
        hours = pd.Series(0.0, index=df.index)
        hours[hourly_mask] = (df.loc[hourly_mask, 'Total'] / rates[hourly_mask]).round(4)
        hours[flat_mask]   = 1.0

        activity_type = df['Type'].map({
            'Hourly time entry':   'Time',
            'Flat rate time entry': 'Fixed Fee',
        }).fillna('Time')

        pg_series = df['User'].map(pg_table).fillna('')

        client_names = df['Client'].astype(str).str.strip()

        out = pd.DataFrame({
            'Service Date':        df['Date'].values,
            'Date of Work':        df['Date'].values,
            'Client Name':         client_names.values,
            'Associated Attorney': df['User'].values,
            'User Name':           df['User'].values,
            # Matter Name = client name for display (matches 2024 convention)
            'Matter Name':         client_names.values,
            # Matter_Type = extracted Clio matter description, used for AI classification only
            'Matter_Type':         df['Matter_Name'].values,
            'Description':         '',
            'Hours':               hours.values,
            'Amount':              df['Total'].values,
            'Activity Type':       activity_type.values,
            'Year':                df['Date'].dt.year.values,
            'Month':               df['Date'].dt.month.values,
            'Month_Name':          df['Date'].dt.strftime('%B').values,
            'Quarter':             df['Date'].dt.quarter.values,
            'Billable Hours':      hours.values,
            'Billable Amount':     df['Total'].values,
            'Original_Hours':      hours.values,
            'Attorney_PG':         pg_series.values,
            'Data_Source':         'clio',
        })

        return out

    except Exception:
        return pd.DataFrame()


def extract_keywords(matter_names):
    """Extract common keywords from matter names"""
    all_words = []
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 're', 'from', 'by', 'as', 'is', 'was', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                  'llc', 'inc', 'corp', 'ltd', 'vs', 'v'}
    
    for matter in matter_names:
        if pd.notna(matter):
            words = re.findall(r'\b[a-z]{4,}\b', matter.lower())
            all_words.extend([w for w in words if w not in stop_words])
    
    return Counter(all_words).most_common(30)

def check_password():
    """Returns `True` if the user has entered the correct password."""

    if st.session_state.get("password_correct", False):
        return True

    st.markdown('<h1 class="main-header">⚖️ OGC Legal AI Automation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### 🔐 Secure Access Required")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Use a form so the submission is isolated and cannot be triggered
        # by other widget interactions (e.g. the year dropdown).
        with st.form("login_form"):
            pwd = st.text_input(
                "Enter Password",
                type="password",
                help="Contact your administrator for access"
            )
            submitted = st.form_submit_button("🔓 Login")

        if submitted:
            if pwd == "AIOGC2026":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")

        st.info("💡 This dashboard contains confidential firm data and automation analysis.")

    return False

def main():
    # Check password first
    if not check_password():
        return
    
    st.markdown('<h1 class="main-header">⚖️ OGC Legal AI Automation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Outside GC - AI-Powered Efficiency Analysis (2024 – 2026 YTD)")

    st.warning(
        "**📋 Data Source Change — Please Read Before Comparing Years**\n\n"
        "The **2024** data comes from **LeanLaw**, where every time entry was recorded against a specific "
        "client matter *and* included a free-text task description (e.g. *'Draft NDA – review redlines'*). "
        "Hours and billing amounts are recorded directly. The 'Task-Level Deep Dive' tab is only available "
        "for this year.\n\n"
        "The **2025 and 2026** data comes from **Clio**. Individual time entries are loaded directly from "
        "Clio exports; the matter type (e.g. *'NDA Review 2025'*, *'Patent and Trademark'*) is extracted "
        "from each entry's matter identifier and used for AI classification. Hours are **back-calculated** "
        "from billed amounts using each attorney's 2024 median hourly rate, since Clio's export does not "
        "include recorded hours. Free-text task descriptions are not available for these years.\n\n"
        "**AI classification** uses the same two-step method for all years: (1) keyword matching on the "
        "matter name/type; (2) if no keyword match, the attorney's practice group is used as a fallback. "
        "Entries logged under a generic *'General'* matter type — where no specific work type is recorded "
        "in Clio — are assigned a conservative **40% automation baseline**. Adding descriptive matter "
        "names in Clio would improve the accuracy of this analysis."
    )

    # Sidebar with logout option
    st.sidebar.title("📊 Dashboard Controls")
    
    if st.sidebar.button("🚪 Logout", help="Log out of the dashboard"):
        st.session_state["password_correct"] = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Load all available years
    try:
        all_dfs = []

        # 2024 — raw time-entry data (filter to 2024 only to avoid overlap with pivot data)
        raw_2024_path = os.path.join(BASE_DIR, '2024/SIX_FULL_MOS_2024.csv')
        df_2024 = load_raw_year(raw_2024_path)
        df_2024 = df_2024[df_2024['Year'] == 2024].copy()
        all_dfs.append(df_2024)
        st.sidebar.success(f"✅ 2024: {len(df_2024):,} time entries loaded")
        fixed_fee_count = (df_2024['Activity Type'] == 'Fixed Fee').sum()
        if fixed_fee_count > 0:
            st.sidebar.info(f"ℹ️ {fixed_fee_count:,} fixed fee entries (2024) counted as 1 hr")

        # 2025 — Clio individual time entries (with matter names extracted from Matter Number)
        clio_2025_path = os.path.join(BASE_DIR, 'RAW_DATA_AND_TRANSFORMATIONS/OGC Clio Activities 2025.xlsx')
        if os.path.exists(clio_2025_path):
            df_2025 = load_clio_year(clio_2025_path)
            if not df_2025.empty and 'Year' in df_2025.columns:
                df_2025 = df_2025[df_2025['Year'] == 2025].copy()
            source_label_2025 = "Clio time entries"
        else:
            df_2025 = load_pivot_year(2025)
            source_label_2025 = "synthesized entries"
        if not df_2025.empty:
            all_dfs.append(df_2025)
            st.sidebar.success(f"✅ 2025: {len(df_2025):,} {source_label_2025} loaded")

        # 2026 — Clio individual time entries (YTD)
        clio_2026_path = os.path.join(BASE_DIR, 'RAW_DATA_AND_TRANSFORMATIONS/OGC Clio Activities 1.1 to 5.29.26.xlsx')
        if os.path.exists(clio_2026_path):
            df_2026 = load_clio_year(clio_2026_path)
            if not df_2026.empty and 'Year' in df_2026.columns:
                df_2026 = df_2026[df_2026['Year'] == 2026].copy()
            source_label_2026 = "Clio time entries"
        else:
            df_2026 = load_pivot_year(2026)
            source_label_2026 = "synthesized entries"
        if not df_2026.empty:
            all_dfs.append(df_2026)
            st.sidebar.success(f"✅ 2026: {len(df_2026):,} {source_label_2026} (YTD) loaded")

        df = pd.concat(all_dfs, ignore_index=True)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    # Year filter — dropdown instead of multiselect
    available_years = sorted(df['Year'].dropna().unique().astype(int))
    year_options = ["All Years"] + [str(y) for y in available_years]
    selected_year_label = st.sidebar.selectbox("📅 Select Year", year_options, index=0)

    if selected_year_label == "All Years":
        year_filtered_df = df.copy()
    else:
        selected_year_int = int(selected_year_label)
        year_filtered_df = df[df['Year'] == selected_year_int].copy()
    
    # User filter
    users = sorted(year_filtered_df['User Name'].dropna().unique())
    selected_users = st.sidebar.multiselect("Select Users", users, default=[])
    
    # Apply filters
    filtered_df = year_filtered_df.copy()
    if selected_users:
        filtered_df = filtered_df[filtered_df['User Name'].isin(selected_users)]
    
    # Detect data sources present in the current view
    sources_present = set(filtered_df.get('Data_Source', pd.Series(dtype=str)).unique())
    has_raw_data    = 'raw'   in sources_present   # 2024 LeanLaw entries (with Descriptions)
    has_clio_data   = 'clio'  in sources_present   # 2025/2026 Clio entries (with Matter Names)
    is_pivot_only   = not (has_raw_data or has_clio_data)

    # Task descriptions exist only in 2024 raw data
    has_detailed_data = has_raw_data

    # Build classification inputs:
    #   - Clio (2025/2026) entries: use Matter_Type (the extracted matter description)
    #     as the primary classification key; fall back to Attorney_PG.
    #   - Raw 2024 entries: use Matter Name (client company name); fall back to Attorney_PG.
    _pg_col = filtered_df['Attorney_PG'].fillna('') if 'Attorney_PG' in filtered_df.columns else pd.Series('', index=filtered_df.index)
    if 'Matter_Type' in filtered_df.columns:
        _classif_col = filtered_df['Matter_Type'].where(
            filtered_df.get('Data_Source', '') == 'clio',
            filtered_df['Matter Name']
        ).fillna(filtered_df['Matter Name'])
    else:
        _classif_col = filtered_df['Matter Name']

    with st.spinner("🤖 Analyzing matters for AI automation potential..."):
        _lb_results = [
            classify_matter_legalbench(mn, pg)
            for mn, pg in zip(_classif_col, _pg_col)
        ]
        filtered_df['Task_Category']      = [r[0] for r in _lb_results]
        filtered_df['Automation_Potential'] = [r[1] for r in _lb_results]
    
    # Calculate automation hours
    filtered_df['Automatable_Hours'] = filtered_df['Billable Hours'] * filtered_df['Automation_Potential']
    filtered_df['Manual_Hours'] = filtered_df['Billable Hours'] - filtered_df['Automatable_Hours']
    
    # Main tabs
    if has_detailed_data:
        st.sidebar.success("✨ Task descriptions available! Check the 'Task-Level Deep Dive' tab.")
    else:
        st.sidebar.info("ℹ️ Task-level descriptions only available for 2024 data.")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview (LegalBench)", 
        "🎯 OLI Benchmark",
        "💰 Cost Savings", 
        "🔮 Predictions",
        "📚 Category Definitions",
        "🔬 Task-Level Deep Dive"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("Overview Dashboard")
        
        st.markdown("""
        ### 🎯 Understanding AI Automation Potential
        This analysis is based on the **LegalBench framework** adapted for OGC's matter-based time entries. 
        Each matter type has been assigned an automation potential based on current AI capabilities.
        
        **Note:** *Fixed fee entries are counted as 1 hour for analysis purposes. 2025/2026 entries are synthesized from monthly attorney-level pivot data with billing amounts allocated proportionally by client.*
        """)
        
        with st.expander("📊 **How We Calculate Your Automation Potential**", expanded=False):
            st.markdown("""
            #### Calculation Methodology
            
            **Step 1: Matter Classification**
            - We analyze each time entry's "Matter Name"
            - Match matters to one of 16 legal practice area categories
            - Examples: Estate Planning, M&A, Litigation, Corporate Governance, etc.
            
            **Step 2: Apply Automation Potential**
            - Each category has researched automation potential (0%-92%)
            - Based on LegalBench research and current AI capabilities
            - Higher % = more suitable for AI assistance
            
            **Step 3: Calculate Automatable Hours**
            ```
            Automatable Hours = Total Hours × Automation Potential %
            
            Example:
            • Matter Type: Estate Planning (88% automation potential)
            • Time Spent: 100 hours
            • Automatable: 100 × 0.88 = 88 hours
            • Manual Oversight: 12 hours
            ```
            
            #### 🤖 What Makes Hours "Automatable"?
            
            **High Automation (80-92%):**
            - ✅ Estate planning documents (88%)
            - ✅ Intellectual property prosecution (85%)
            - ✅ Regulatory compliance (83%)
            - ✅ Contract review and drafting (92%)
            - ✅ Corporate governance documents (80%)
            
            **Medium Automation (65-80%):**
            - ⚠️ M&A transactions (82%)
            - ⚠️ Employment law (80%)
            - ⚠️ Immigration matters (78%)
            - ⚠️ Tax planning (75%)
            - ⚠️ Real estate (75%)
            
            **Lower Automation (0-70%):**
            - ⛔ Litigation and disputes (75%)
            - ⛔ Family law (68%)
            - ⛔ General advice (65%)
            - ⛔ Internal time (0%)
            """)
        
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_hours = filtered_df['Billable Hours'].sum()
        automatable_hours = filtered_df['Automatable_Hours'].sum()
        automation_rate = (automatable_hours / total_hours * 100) if total_hours > 0 else 0
        
        with col1:
            st.metric(
                label="Total Billable Hours",
                value=f"{total_hours:,.0f}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="AI-Automatable Hours",
                value=f"{automatable_hours:,.0f}",
                delta=f"{automation_rate:.1f}% of total",
                help="Hours that could be accelerated with AI assistance"
            )
        
        with col3:
            total_billable = filtered_df['Billable Amount'].sum()
            st.metric(
                label="Total Billable Amount",
                value=f"${total_billable:,.0f}"
            )
        
        with col4:
            unique_clients = filtered_df['Client Name'].nunique()
            st.metric(
                label="Unique Clients",
                value=f"{unique_clients:,}"
            )
        
        st.markdown("---")
        
        # Monthly trend visualization
        st.subheader("💰 Monthly Work Distribution: AI-Automatable vs. Human-Required")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            monthly_data = filtered_df.groupby(['Year', 'Month', 'Month_Name']).agg({
                'Billable Hours': 'sum',
                'Automatable_Hours': 'sum',
                'Manual_Hours': 'sum'
            }).reset_index()
            monthly_data = monthly_data.sort_values(['Year', 'Month'])
            monthly_data['Period'] = monthly_data['Month_Name'] + ' ' + monthly_data['Year'].astype(str)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=monthly_data['Period'],
                y=monthly_data['Automatable_Hours'],
                name='AI-Automatable',
                mode='lines',
                line=dict(width=0.5, color='rgb(34, 139, 34)'),
                stackgroup='one',
                fillcolor='rgba(34, 139, 34, 0.6)',
                hovertemplate='%{y:.0f} automatable hours<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_data['Period'],
                y=monthly_data['Manual_Hours'],
                name='Human-Required',
                mode='lines',
                line=dict(width=0.5, color='rgb(255, 140, 0)'),
                stackgroup='one',
                fillcolor='rgba(255, 140, 0, 0.6)',
                hovertemplate='%{y:.0f} manual hours<extra></extra>'
            ))
            
            fig.update_layout(
                title='Monthly Hours: AI-Automatable vs. Human-Required',
                xaxis_title='Month',
                yaxis_title='Hours',
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=['AI-Automatable Hours', 'Human-Required Hours'],
                values=[automatable_hours, total_hours - automatable_hours],
                hole=0.5,
                marker_colors=['#228B22', '#FF8C00'],
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title='Overall Work Distribution',
                height=400,
                showlegend=False,
                annotations=[dict(
                    text=f'{automation_rate:.1f}%<br>Automatable',
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top automation opportunities
        st.subheader("📊 Top Automation Opportunities by Practice Area")
        
        category_data = filtered_df[filtered_df['Task_Category'] != 'Internal-Time'].groupby('Task_Category').agg({
            'Billable Hours': 'sum',
            'Automatable_Hours': 'sum',
            'Automation_Potential': 'first'
        }).reset_index()
        category_data = category_data.sort_values('Automatable_Hours', ascending=False).head(12)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                category_data,
                x='Automatable_Hours',
                y='Task_Category',
                orientation='h',
                title='Top 12 Practice Areas by AI-Automatable Hours',
                labels={'Automatable_Hours': 'AI-Automatable Hours', 'Task_Category': 'Practice Area'},
                color='Automation_Potential',
                color_continuous_scale='Greens',
                text='Automatable_Hours'
            )
            
            fig.update_traces(
                texttemplate='%{text:.0f}h',
                textposition='outside'
            )
            
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='Hours',
                coloraxis_colorbar=dict(title="Automation<br>Potential")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            category_data['Potential_Savings_Pct'] = category_data['Automation_Potential'] * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=category_data['Task_Category'],
                x=category_data['Billable Hours'],
                name='Total Hours',
                orientation='h',
                marker_color='lightblue',
                text=category_data['Billable Hours'].round(0),
                textposition='inside'
            ))
            
            fig.update_layout(
                title='Total Hours by Practice Area<br><sub>Darker green = higher automation potential</sub>',
                xaxis_title='Hours',
                yaxis_title='',
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            
            colors = category_data['Automation_Potential'].apply(
                lambda x: f'rgba(34, 139, 34, {x})' if x > 0.8 else 
                         f'rgba(255, 165, 0, {x})' if x > 0.7 else 
                         f'rgba(255, 99, 71, {x})'
            )
            fig.data[0].marker.color = colors
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # User analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Top 15 Users by Hours")
            user_hours = filtered_df.groupby('User Name').agg({
                'Billable Hours': 'sum',
                'Automatable_Hours': 'sum'
            }).reset_index().sort_values('Billable Hours', ascending=False).head(15)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=user_hours['User Name'],
                x=user_hours['Billable Hours'],
                name='Total Hours',
                orientation='h',
                marker_color='lightcoral'
            ))
            fig.add_trace(go.Bar(
                y=user_hours['User Name'],
                x=user_hours['Automatable_Hours'],
                name='AI-Automatable',
                orientation='h',
                marker_color='darkred'
            ))
            fig.update_layout(
                barmode='overlay',
                height=500,
                hovermode='y unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Top 15 Matters by Hours")
            matter_hours = filtered_df.groupby('Matter Name').agg({
                'Billable Hours': 'sum',
                'Automatable_Hours': 'sum'
            }).reset_index().sort_values('Billable Hours', ascending=False).head(15)
            
            fig = px.bar(
                matter_hours,
                x='Billable Hours',
                y='Matter Name',
                orientation='h',
                title='',
                color='Automatable_Hours',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **🤖 AI Could Assist With:**
            - {automatable_hours:,.0f} hours ({automation_rate:.1f}%)
            - Equivalent to {automatable_hours/40:.0f} work weeks
            - Or {automatable_hours/2080:.1f} full-time employees
            """)
        
        with col2:
            avg_automation = filtered_df['Automation_Potential'].mean() * 100
            st.success(f"""
            **📈 Average Matter Automation:**
            - {avg_automation:.1f}% automation potential
            - Based on {len(filtered_df):,} time entries
            - Across {unique_clients:,} clients
            """)
        
        with col3:
            if len(category_data) > 0:
                top_category = category_data.iloc[0]
                st.warning(f"""
                **🎯 Top Opportunity:**
                - **{top_category['Task_Category']}**
                - {top_category['Automatable_Hours']:.0f} automatable hours
                - {top_category['Automation_Potential']*100:.0f}% automation potential
                """)
    
    # TAB 2: Rimon Benchmark
    with tab2:
        st.header("🎯 OGC Benchmark - Custom Practice Area Analysis")
        
        st.markdown("""
        ### 📊 OGC AI Automation Assessment
        This tab uses **OGC Benchmark** - a custom assessment tailored to OGC's specific 
        practice areas and matter types.

        **Note:** *Fixed fee entries are counted as 1 hour for analysis purposes.*
        """)

        if has_clio_data and not has_raw_data:
            # Explain the "General" retainer limitation for Clio-only views
            general_pct = (
                filtered_df['Matter Name'].str.lower().str.strip() == 'general'
            ).mean() * 100
            if general_pct > 20:
                st.info(
                    f"📌 **Data Granularity Note:** {general_pct:.0f}% of entries in this view have a "
                    f"matter name of **'General'** — Clio's label for ongoing retainer work where no "
                    f"specific matter type was recorded. These entries are classified as "
                    f"**'Unclassified / General Retainer'** at a conservative 40% automation baseline. "
                    f"Entries with descriptive matter names (NDA Review, Patent, Contract Review, etc.) "
                    f"are classified at their specific tier. "
                    f"Adding matter-level descriptions in Clio would significantly improve this analysis."
                )

        if is_pivot_only:
            st.info(
                "📌 **Data Note:** Aggregated practice-group data is being used as the matter name source. "
                "OLI tiers are assigned via a direct PG → tier mapping. "
                "Year-over-year comparisons should be interpreted with caution."
            )
        
        # Classify using OGC Benchmark — same Matter_Type / PG logic as LegalBench
        with st.spinner("🤖 Analyzing using OGC Benchmark..."):
            _oli_results = [
                classify_matter_oli(mn, pg)
                for mn, pg in zip(_classif_col, _pg_col)
            ]
            filtered_df['OLI_Category']            = [r[0] for r in _oli_results]
            filtered_df['OLI_Automation_Potential'] = [r[1] for r in _oli_results]
        
        filtered_df['OLI_Automatable_Hours'] = filtered_df['Billable Hours'] * filtered_df['OLI_Automation_Potential']
        filtered_df['OLI_Manual_Hours'] = filtered_df['Billable Hours'] - filtered_df['OLI_Automatable_Hours']
        
        with st.expander("📋 **OGC Benchmark Categories**", expanded=False):
            st.markdown("""
            ### OGC Benchmark Classification
            
            #### 🟢 100% Automatable (Two Categories)
            
            **1. Routine Corporate & Documents:**
            - General Corporate Matters
            - General Representation
            - Retainer Services
            - Corporate Advice
            
            **2. Estate Planning:**
            - Estate Planning
            - Trust Administration
            - Wills & Probate
            
            #### 🟡 70% Automatable (Two Categories)
            
            **1. Transactional Work:**
            - M&A & Acquisitions
            - Financings & Investments
            - Real Estate Transactions
            - Securities Matters
            
            **2. IP & Regulatory:**
            - Patent & Trademark Prosecution
            - Regulatory Compliance
            - Immigration Matters
            
            #### 🟠 30% Automatable (Two Categories)
            
            **1. Litigation & Complex:**
            - Civil Litigation
            - Arbitration & Mediation
            - Bankruptcy
            
            **2. Employment & Family:**
            - Employment Disputes
            - Family Law Matters
            - Tax Planning
            
            #### ⚫ 0% Automatable
            **Internal & Strategic:**
            - Internal Time
            - Business Development
            - Training
            """)
        
        st.markdown("---")
        
        # Rimon metrics
        st.subheader("📈 OGC Benchmark Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        oli_total = filtered_df['Billable Hours'].sum()
        oli_automatable = filtered_df['OLI_Automatable_Hours'].sum()
        oli_rate = (oli_automatable / oli_total * 100) if oli_total > 0 else 0
        oli_manual = filtered_df['OLI_Manual_Hours'].sum()
        
        with col1:
            st.metric(
                label="Total Hours",
                value=f"{oli_total:,.0f}"
            )
        
        with col2:
            st.metric(
                label="OLI AI-Automatable",
                value=f"{oli_automatable:,.0f}",
                delta=f"{oli_rate:.1f}% of total"
            )
        
        with col3:
            st.metric(
                label="Human-Required",
                value=f"{oli_manual:,.0f}",
                delta=f"{(oli_manual/oli_total*100):.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            potential_savings = oli_automatable * 0.60 * 500
            st.metric(
                label="Potential Savings (Est.)",
                value=f"${potential_savings:,.0f}",
                help="At 60% efficiency, $500/hour"
            )
        
        st.markdown("---")
        
        # Rimon visualization
        st.subheader("💰 OGC Benchmark Distribution")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            oli_monthly = filtered_df.groupby(['Year', 'Month', 'Month_Name']).agg({
                'Billable Hours': 'sum',
                'OLI_Automatable_Hours': 'sum',
                'OLI_Manual_Hours': 'sum'
            }).reset_index()
            oli_monthly = oli_monthly.sort_values(['Year', 'Month'])
            oli_monthly['Period'] = oli_monthly['Month_Name'] + ' ' + oli_monthly['Year'].astype(str)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=oli_monthly['Period'],
                y=oli_monthly['OLI_Automatable_Hours'],
                name='AI-Automatable',
                mode='lines',
                line=dict(width=0.5, color='rgb(0, 128, 0)'),
                stackgroup='one',
                fillcolor='rgba(0, 128, 0, 0.7)'
            ))
            
            fig.add_trace(go.Scatter(
                x=oli_monthly['Period'],
                y=oli_monthly['OLI_Manual_Hours'],
                name='Human-Required',
                mode='lines',
                line=dict(width=0.5, color='rgb(220, 20, 60)'),
                stackgroup='one',
                fillcolor='rgba(220, 20, 60, 0.7)'
            ))
            
            fig.update_layout(
                title='OGC Benchmark: Monthly Distribution',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=['AI-Automatable', 'Human-Required'],
                values=[oli_automatable, oli_manual],
                hole=0.5,
                marker_colors=['#008000', '#DC143C']
            )])
            
            fig.update_layout(
                title='Overall Distribution',
                height=400,
                annotations=[dict(
                    text=f'{oli_rate:.1f}%<br>Automatable',
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Category breakdown
        st.subheader("📊 Hours by OGC Automation Tier")
        
        oli_categories = filtered_df[filtered_df['OLI_Category'] != 'Unclassified'].groupby('OLI_Category').agg({
            'Billable Hours': 'sum',
            'OLI_Automatable_Hours': 'sum',
            'OLI_Automation_Potential': 'first'
        }).reset_index()
        oli_categories = oli_categories.sort_values('OLI_Automation_Potential', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=oli_categories['OLI_Category'],
                x=oli_categories['Billable Hours'],
                name='Total',
                orientation='h',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                y=oli_categories['OLI_Category'],
                x=oli_categories['OLI_Automatable_Hours'],
                name='Automatable',
                orientation='h',
                marker_color='darkgreen'
            ))
            
            fig.update_layout(
                title='Hours by OGC Category',
                height=450,
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                oli_categories,
                y='OLI_Category',
                x='OLI_Automatable_Hours',
                orientation='h',
                title='Automatable Hours by Tier',
                color='OLI_Automation_Potential',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=450)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Top matters
        st.markdown("---")
        st.subheader("🎯 Top Matters for AI Implementation")
        
        matter_analysis = filtered_df[
            filtered_df['OLI_Category'] != '0% AI Replaceable - Internal & Strategic'
        ].groupby('Matter Name').agg({
            'Billable Hours': 'sum',
            'OLI_Automatable_Hours': 'sum'
        }).reset_index()
        matter_analysis['Automation_Rate'] = (
            matter_analysis['OLI_Automatable_Hours'] / matter_analysis['Billable Hours'] * 100
        )
        matter_analysis = matter_analysis.sort_values('OLI_Automatable_Hours', ascending=False).head(20)
        
        st.dataframe(
            matter_analysis.style.format({
                'Billable Hours': '{:.1f}',
                'OLI_Automatable_Hours': '{:.1f}',
                'Automation_Rate': '{:.1f}%'
            }).background_gradient(subset=['OLI_Automatable_Hours'], cmap='Greens'),
            use_container_width=True,
            height=500
        )
    
    # TAB 3: Cost Savings
    with tab3:
        st.header("💰 Potential Cost Savings with AI")
        
        st.subheader("⚙️ Assumptions & Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_hourly_rate = st.number_input(
                "Average Hourly Rate ($)",
                min_value=100,
                max_value=1000,
                value=500,
                step=50
            )
        
        with col2:
            ai_efficiency_gain = st.slider(
                "AI Efficiency Gain (%)",
                min_value=10,
                max_value=90,
                value=60,
                help="Percentage of time saved on automatable tasks"
            ) / 100
        
        with col3:
            ai_cost_per_hour = st.number_input(
                "AI Cost per Hour ($)",
                min_value=1,
                max_value=100,
                value=10,
                step=5
            )
        
        st.markdown("---")
        
        # Calculate savings
        hours_saved = automatable_hours * ai_efficiency_gain
        labor_saved = hours_saved * avg_hourly_rate
        ai_cost = automatable_hours * ai_cost_per_hour
        net_savings = labor_saved - ai_cost
        roi = (net_savings / ai_cost * 100) if ai_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Hours Potentially Saved",
                value=f"{hours_saved:,.0f}",
                delta=f"{(hours_saved/total_hours*100):.1f}% of total"
            )
        
        with col2:
            st.metric(
                label="Labor Cost Savings",
                value=f"${labor_saved:,.0f}"
            )
        
        with col3:
            st.metric(
                label="AI Implementation Cost",
                value=f"${ai_cost:,.0f}"
            )
        
        with col4:
            st.metric(
                label="Net Savings",
                value=f"${net_savings:,.0f}",
                delta=f"ROI: {roi:.0f}%"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💵 Savings by Practice Area")
            
            category_savings = filtered_df.groupby('Task_Category').agg({
                'Automatable_Hours': 'sum'
            }).reset_index()
            
            category_savings['Hours_Saved'] = category_savings['Automatable_Hours'] * ai_efficiency_gain
            category_savings['Cost_Savings'] = category_savings['Hours_Saved'] * avg_hourly_rate
            category_savings = category_savings.sort_values('Cost_Savings', ascending=False).head(12)
            
            fig = px.bar(
                category_savings,
                x='Task_Category',
                y='Cost_Savings',
                title='Potential Savings by Category',
                color='Cost_Savings',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Cumulative Savings")
            
            monthly_savings = filtered_df.groupby(['Year', 'Month']).agg({
                'Automatable_Hours': 'sum'
            }).reset_index()
            monthly_savings = monthly_savings.sort_values(['Year', 'Month'])
            monthly_savings['Hours_Saved'] = monthly_savings['Automatable_Hours'] * ai_efficiency_gain
            monthly_savings['Monthly_Savings'] = monthly_savings['Hours_Saved'] * avg_hourly_rate
            monthly_savings['Cumulative_Savings'] = monthly_savings['Monthly_Savings'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_savings.index,
                y=monthly_savings['Cumulative_Savings'],
                mode='lines+markers',
                name='Cumulative',
                fill='tozeroy',
                line=dict(color='green', width=3)
            ))
            fig.update_layout(
                    title='Cumulative Cost Savings Over Time',
                    height=400
                )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Predictions
    with tab4:
        # Use the most recent year with data for projections
        latest_year = int(filtered_df['Year'].max()) if len(filtered_df) > 0 else 2024
        st.header(f"🔮 {latest_year} Projections")
        
        current_data = filtered_df[filtered_df['Year'] == latest_year]
        
        if len(current_data) > 0:
            latest_month = current_data['Month'].max()
            
            monthly_avg = current_data.groupby('Month').agg({
                'Billable Hours': 'sum',
                'Automatable_Hours': 'sum'
            }).mean()
            
            months_elapsed = latest_month
            months_remaining = 12 - months_elapsed
            
            projected_total = (current_data['Billable Hours'].sum() + 
                              monthly_avg['Billable Hours'] * months_remaining)
            projected_automatable = (current_data['Automatable_Hours'].sum() + 
                                    monthly_avg['Automatable_Hours'] * months_remaining)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Projected Total Hours (2025)",
                    value=f"{projected_total:,.0f}",
                    delta=f"+{months_remaining} months projected"
                )
            
            with col2:
                st.metric(
                    label="Projected Automatable Hours",
                    value=f"{projected_automatable:,.0f}",
                    delta=f"{(projected_automatable/projected_total*100):.1f}%"
                )
            
            with col3:
                projected_savings = projected_automatable * ai_efficiency_gain * avg_hourly_rate
                st.metric(
                    label="Projected Annual Savings",
                    value=f"${projected_savings:,.0f}"
                )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Monthly Projection")
                
                actual_monthly = current_data.groupby('Month').agg({
                    'Billable Hours': 'sum',
                    'Automatable_Hours': 'sum'
                }).reset_index()
                
                all_months = pd.DataFrame({'Month': range(1, 13)})
                projection_df = all_months.merge(actual_monthly, on='Month', how='left')
                
                projection_df['Billable Hours'] = projection_df['Billable Hours'].fillna(monthly_avg['Billable Hours'])
                projection_df['Automatable_Hours'] = projection_df['Automatable_Hours'].fillna(
                    monthly_avg['Automatable_Hours']
                )
                projection_df['Type'] = projection_df['Month'].apply(
                    lambda x: 'Actual' if x <= months_elapsed else 'Projected'
                )
                
                fig = go.Figure()
                
                actual = projection_df[projection_df['Type'] == 'Actual']
                fig.add_trace(go.Bar(
                    x=actual['Month'],
                    y=actual['Billable Hours'],
                    name='Actual Total',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=actual['Month'],
                    y=actual['Automatable_Hours'],
                    name='Actual Automatable',
                    marker_color='darkblue'
                ))
                
                projected = projection_df[projection_df['Type'] == 'Projected']
                fig.add_trace(go.Bar(
                    x=projected['Month'],
                    y=projected['Billable Hours'],
                    name='Projected Total',
                    marker_color='lightcoral',
                    opacity=0.6
                ))
                fig.add_trace(go.Bar(
                    x=projected['Month'],
                    y=projected['Automatable_Hours'],
                    name='Projected Automatable',
                    marker_color='darkred',
                    opacity=0.6
                ))
                
                fig.update_layout(
                    title=f'{latest_year} Monthly Hours Projection',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💰 Cumulative Savings Projection")
                
                projection_df['Monthly_Savings'] = (
                    projection_df['Automatable_Hours'] * ai_efficiency_gain * avg_hourly_rate
                )
                projection_df['Cumulative_Savings'] = projection_df['Monthly_Savings'].cumsum()
                
                fig = go.Figure()
                
                actual_cum = projection_df[projection_df['Type'] == 'Actual']
                fig.add_trace(go.Scatter(
                    x=actual_cum['Month'],
                    y=actual_cum['Cumulative_Savings'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='green', width=3),
                    fill='tozeroy'
                ))
                
                fig.add_trace(go.Scatter(
                    x=projection_df['Month'],
                    y=projection_df['Cumulative_Savings'],
                    mode='lines+markers',
                    name='Projected',
                    line=dict(color='lightgreen', width=3, dash='dash'),
                    fill='tozeroy',
                    opacity=0.5
                ))
                
                fig.update_layout(
                    title='Cumulative Savings Projection',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Scenario analysis
            st.markdown("---")
            st.subheader("🎲 Scenario Analysis")
            
            scenarios = {
                'Conservative (40% efficiency)': 0.40,
                'Moderate (60% efficiency)': 0.60,
                'Optimistic (80% efficiency)': 0.80
            }
            
            scenario_results = []
            for name, efficiency in scenarios.items():
                h_saved = projected_automatable * efficiency
                cost_saved = h_saved * avg_hourly_rate
                ai_cost_total = projected_automatable * ai_cost_per_hour
                net = cost_saved - ai_cost_total
                
                scenario_results.append({
                    'Scenario': name,
                    'Hours Saved': h_saved,
                    'Cost Saved': cost_saved,
                    'AI Cost': ai_cost_total,
                    'Net Savings': net,
                    'ROI (%)': (net / ai_cost_total * 100) if ai_cost_total > 0 else 0
                })
            
            scenario_df = pd.DataFrame(scenario_results)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=scenario_df['Scenario'],
                    y=scenario_df['Net Savings'],
                    marker_color='green',
                    text=scenario_df['Net Savings'].apply(lambda x: f'${x:,.0f}'),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Net Savings by Scenario',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    scenario_df.style.format({
                        'Hours Saved': '{:,.0f}',
                        'Cost Saved': '${:,.0f}',
                        'AI Cost': '${:,.0f}',
                        'Net Savings': '${:,.0f}',
                        'ROI (%)': '{:.0f}%'
                    }),
                    use_container_width=True,
                    height=400
                )
        else:
            st.warning(f"No {latest_year} data available for projections.")
    
    # TAB 5: Definitions
    with tab5:
        st.header("📚 LegalBench Practice Area Definitions")
        
        st.markdown("""
        Based on the **LegalBench framework** adapted for OGC time tracking.
        """)
        
        for category, info in LEGALBENCH_TASKS.items():
            if category == 'Internal-Time':
                continue
            
            with st.expander(f"**{category}** - Automation: {info['automation_potential']*100:.0f}%"):
                st.markdown(f"**Description:** {info['description']}")
                
                st.markdown("**Keywords:**")
                st.write(", ".join(info['keywords']))
                
                st.markdown("**Examples:**")
                for example in info['examples']:
                    st.write(f"• {example}")
                
                matching = filtered_df[
                    filtered_df['Task_Category'] == category
                ]['Matter Name'].value_counts().head(5)
                
                if len(matching) > 0:
                    st.markdown("**Top 5 Matters in Your Data:**")
                    for matter, count in matching.items():
                        st.write(f"• {matter} ({count} entries)")
        
        st.markdown("---")
        st.info("""
        **Note:** Automation potentials are estimates based on current AI capabilities 
        and the LegalBench framework. Actual results depend on implementation and oversight.
        """)


    
    # ========================================================================
    # TAB 6: TASK-LEVEL DEEP DIVE (conditional)
    # ========================================================================
    if has_detailed_data:
        with tab6:
            st.header("🔬 Task-Level Deep Dive: 2024 Raw Data")
            
            st.markdown("""
            ### 🎯 Ultra-Precise Automation Analysis (2024 Raw Data)
            
            This tab uses **actual task descriptions** from the 2024 raw time-entry data for much
            more accurate automation scoring than the matter-name approach used in other tabs.
            
            **Examples from your data:**
            - "Email regarding status" → 92% automatable
            - "Review and analyze agreement" → 85% automatable
            - "Telephone conference with client" → 45% automatable
            - "Negotiate settlement" → 25% automatable
            
            **Note:** Task-level detail is only available for 2024. Select 2024 or All Years to use this tab.
            """)
            
            st.info("""
            💡 **What This Tab Shows:**

            This analyzes 2024 time entries using detailed task descriptions from the Description column.

            This tab demonstrates:
            - How much MORE PRECISE we can be with detailed task descriptions
            - What automation looks like at the task level vs matter level
            - Ultra-granular analysis of your actual work
            """)
            
            # Use the main filtered dataframe (it already has Description column)
            with st.spinner("🔬 Analyzing task descriptions..."):
                detailed_df = filtered_df.copy()
                
                # Classify tasks if not already done
                if 'Task_Type' not in detailed_df.columns:
                    detailed_df[['Task_Type', 'Task_Automation']] = detailed_df['Description'].apply(
                        lambda x: pd.Series(classify_task_description(x))
                    )
                    detailed_df['Task_Automatable_Hours'] = detailed_df['Billable Hours'] * detailed_df['Task_Automation']
                    detailed_df['Task_Manual_Hours'] = detailed_df['Billable Hours'] - detailed_df['Task_Automatable_Hours']
            
            # Classify tasks
            st.info(f"📊 Analyzing {len(detailed_df):,} detailed task entries...")
            
            # Calculate automatable hours (if not already done)
            if 'Task_Automatable_Hours' not in detailed_df.columns:
                detailed_df['Task_Automatable_Hours'] = detailed_df['Billable Hours'] * detailed_df['Task_Automation']
                detailed_df['Task_Manual_Hours'] = detailed_df['Billable Hours'] - detailed_df['Task_Automatable_Hours']
            
            # Key metrics
            st.markdown("---")
            st.subheader("📊 Task-Level Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            task_total = detailed_df['Billable Hours'].sum()
            task_auto = detailed_df['Task_Automatable_Hours'].sum()
            task_rate = (task_auto / task_total * 100) if task_total > 0 else 0
            
            with col1:
                st.metric(
                    label="Total Hours Analyzed",
                    value=f"{task_total:,.0f}",
                    help="From detailed task descriptions"
                )
            
            with col2:
                st.metric(
                    label="AI-Automatable (Task-Level)",
                    value=f"{task_auto:,.0f}",
                    delta=f"{task_rate:.1f}%",
                    help="Based on specific task descriptions"
                )
            
            with col3:
                avg_task_auto = detailed_df['Task_Automation'].mean() * 100
                st.metric(
                    label="Avg Task Automation",
                    value=f"{avg_task_auto:.1f}%",
                    help="Average automation potential per task"
                )
            
            with col4:
                unique_tasks = detailed_df['Task_Type'].nunique()
                st.metric(
                    label="Unique Task Types",
                    value=f"{unique_tasks}"
                )
            
            st.markdown("---")
            
            # Task type breakdown
            st.subheader("🎯 Automation by Task Type")
            
            task_breakdown = detailed_df.groupby('Task_Type').agg({
                'Billable Hours': 'sum',
                'Task_Automatable_Hours': 'sum',
                'Task_Automation': 'first'
            }).reset_index()
            task_breakdown = task_breakdown.sort_values('Task_Automatable_Hours', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    task_breakdown.head(12),
                    x='Task_Automatable_Hours',
                    y='Task_Type',
                    orientation='h',
                    title='Top 12 Task Types by Automatable Hours',
                    color='Task_Automation',
                    color_continuous_scale='RdYlGn',
                    text='Task_Automatable_Hours'
                )
                fig.update_traces(texttemplate='%{text:.0f}h', textposition='outside')
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=task_breakdown.head(12)['Task_Type'],
                    x=task_breakdown.head(12)['Billable Hours'],
                    name='Total Hours',
                    orientation='h',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    y=task_breakdown.head(12)['Task_Type'],
                    x=task_breakdown.head(12)['Task_Automatable_Hours'],
                    name='Automatable',
                    orientation='h',
                    marker_color='darkgreen'
                ))
                fig.update_layout(
                    title='Total vs Automatable Hours',
                    height=500,
                    barmode='overlay',
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Sample tasks
            st.subheader("📝 Sample Task Classifications")
            
            sample_tasks = []
            for task_type in task_breakdown.head(10)['Task_Type']:
                sample = detailed_df[detailed_df['Task_Type'] == task_type].head(2)
                sample_tasks.append(sample)
            
            if sample_tasks:
                sample_df = pd.concat(sample_tasks)
                display_cols = ['Description', 'Task_Type', 'Task_Automation', 'Billable Hours']
                
                st.dataframe(
                    sample_df[display_cols].style.format({
                        'Task_Automation': '{:.0%}',
                        'Billable Hours': '{:.2f}'
                    }),
                    use_container_width=True,
                    height=400
                )
            
            st.markdown("---")
            
            # Comparison
            st.subheader("📊 Task-Level vs Matter-Level Comparison")
            
            st.warning("""
            **⚠️ Important:** Both analyses use the same 2024 raw data:
            - Task-Level (this tab): Analyzed by detailed task descriptions
            - Matter-Level (main tabs): Analyzed by matter names only
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🔬 Task-Level (2024 Raw Data):**
                - Dataset: 2024 raw time entries
                - Entries: {len(detailed_df):,}
                - Automatable: {task_auto:,.0f} hours
                - Rate: {task_rate:.1f}%
                - Precision: HIGH (task descriptions)
                """)
            
            with col2:
                # Matter-level for same 2024 raw data
                matter_auto = filtered_df.loc[
                    filtered_df['Data_Source'] == 'raw', 'Automatable_Hours'
                ].sum() if 'Automatable_Hours' in filtered_df.columns else 0
                matter_total = filtered_df.loc[
                    filtered_df['Data_Source'] == 'raw', 'Billable Hours'
                ].sum()
                matter_rate = (matter_auto / matter_total * 100) if matter_total > 0 else 0
                raw_count = (filtered_df['Data_Source'] == 'raw').sum()

                st.info(f"""
                **📊 Matter-Level (2024 Raw Data):**
                - Dataset: 2024 raw time entries
                - Entries: {raw_count:,}
                - Automatable: {matter_auto:,.0f} hours
                - Rate: {matter_rate:.1f}%
                - Precision: MEDIUM (matter names)
                """)
            
            difference = abs(task_rate - automation_rate)
            if task_rate > automation_rate:
                st.warning(f"⚡ Task-level analysis shows {difference:.1f} percentage points MORE automation potential!")
            else:
                st.info(f"📉 Task-level analysis shows {difference:.1f} percentage points LESS automation potential (more conservative).")
            
            # Top opportunities
            st.markdown("---")
            st.subheader("🎯 Top Automation Opportunities (Task-Level)")
            
            high_auto_tasks = detailed_df[detailed_df['Task_Automation'] >= 0.85].groupby('Description').agg({
                'Billable Hours': 'sum',
                'Task_Automation': 'first'
            }).reset_index().sort_values('Billable Hours', ascending=False).head(20)
            
            if len(high_auto_tasks) > 0:
                st.write(f"**{len(high_auto_tasks)} high-automation tasks (≥85%) with most hours:**")
                st.dataframe(
                    high_auto_tasks.style.format({
                        'Task_Automation': '{:.0%}',
                        'Billable Hours': '{:.1f}'
                    }),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No high-automation tasks found in this dataset.")


    else:
        with tab6:
            st.header("🔬 Task-Level Deep Dive")
            st.info(
                "Task-level analysis requires the 2024 raw time-entry data (which includes "
                "individual task descriptions). The currently selected year filter returns "
                "only 2025/2026 aggregate data.\n\n"
                "Select **All Years** or **2024** in the sidebar to enable this tab."
            )


if __name__ == "__main__":
    main()
