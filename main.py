import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import re
from collections import Counter

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
    </style>
""", unsafe_allow_html=True)

# TASK-LEVEL CLASSIFICATION
TASK_LEVEL_AUTOMATION = {
    'Document-Review-Standard': {'description': 'Review of standard documents', 'automation_potential': 0.95, 'keywords': ['review agreement', 'review contract', 'review and revise', 'review draft', 'review standard', 'review form', 'review template', 'review nda', 'review msa', 'review amendment', 'review lease', 'review psa']},
    'Email-Status-Updates': {'description': 'Status update emails', 'automation_potential': 0.92, 'keywords': ['email regarding', 'email correspondence', 'status update', 'follow up email', 'exchange emails', 'send email', 'attn to corresp', 'correspondence with']},
    'Document-Drafting-Standard': {'description': 'Drafting standard forms', 'automation_potential': 0.93, 'keywords': ['draft amendment', 'draft addendum', 'draft standard', 'draft form', 'draft certificate', 'draft notice', 'draft letter', 'prepare draft', 'drafting agreement']},
    'Research-Straightforward': {'description': 'Straightforward legal research', 'automation_potential': 0.88, 'keywords': ['research case law', 'research statute', 'research regulation', 'research precedent', 'legal research']},
    'Form-Completion': {'description': 'Completing forms', 'automation_potential': 0.96, 'keywords': ['complete form', 'fill out', 'prepare filing', 'file notice', 'file certificate', 'submit form']},
    'Document-Analysis': {'description': 'Analyzing documents', 'automation_potential': 0.85, 'keywords': ['review and analyze', 'analyze agreement', 'analyze contract', 'analyze terms', 'analyze provision', 'review for compliance', 'analyze draft', 'analysis of']},
    'Due-Diligence-Review': {'description': 'Due diligence review', 'automation_potential': 0.82, 'keywords': ['due diligence', 'dd review', 'review due diligence', 'diligence materials', 'data room review']},
    'Discovery-Review': {'description': 'Document discovery review', 'automation_potential': 0.87, 'keywords': ['review discovery', 'review production', 'review interrogator', 'review request for production', 'discovery response', 'respond to discovery']},
    'Clause-Extraction': {'description': 'Extracting specific clauses', 'automation_potential': 0.90, 'keywords': ['extract provision', 'identify clause', 'locate language', 'find provision', 'pull clause', 'summarize terms']},
    'Drafting-Complex': {'description': 'Drafting complex agreements', 'automation_potential': 0.65, 'keywords': ['draft purchase agreement', 'draft psa', 'draft merger agreement', 'drafting apa', 'draft complex', 'draft financing', 'draft loan', 'draft settlement']},
    'Negotiation-Support': {'description': 'Supporting negotiations', 'automation_potential': 0.60, 'keywords': ['revise per comments', 'address comments', 'incorporate revisions', 'revise based on', 'respond to comments', 'counter proposal']},
    'Legal-Memos': {'description': 'Legal memoranda', 'automation_potential': 0.55, 'keywords': ['draft memo', 'memorandum', 'legal opinion', 'prepare memo', 'draft analysis', 'memo regarding']},
    'Client-Calls': {'description': 'Client communications', 'automation_potential': 0.45, 'keywords': ['call with', 'telephone conference', 'conference call', 'client meeting', 'discuss with', 'conf w/', 'confer with']},
    'Court-Appearances': {'description': 'Court appearances', 'automation_potential': 0.30, 'keywords': ['court appearance', 'appear in court', 'attend hearing', 'oral argument', 'trial', 'deposition', 'attend conference']},
    'Strategic-Advice': {'description': 'Strategic legal counseling', 'automation_potential': 0.35, 'keywords': ['advise regarding', 'counsel regarding', 'discuss strategy', 'strategic advice', 'recommendation regarding', 'consult on', 'advise accordingly']},
    'Negotiations': {'description': 'Negotiation sessions', 'automation_potential': 0.25, 'keywords': ['negotiate', 'negotiation', 'negotiating', 'negotiate terms', 'negotiate with', 'settlement discussion']},
    'Internal-Admin': {'description': 'Internal administrative tasks', 'automation_potential': 0.10, 'keywords': ['internal', 'administrative', 'firm meeting', 'training', 'business development', 'marketing', 'time entry', 'billing']},
    'General-Communication': {'description': 'General correspondence', 'automation_potential': 0.50, 'keywords': ['email', 'correspondence', 'communicate', 'discuss', 'exchange', 'speak with', 'follow up', 'attention to']}
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

@st.cache_data
def load_data(csv_path):
    """Load and preprocess the OGC CSV data"""
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Convert date
    df['Service Date'] = pd.to_datetime(df['Service Date'], format='%m/%d/%y', errors='coerce')
    
    # Convert hours and amount to numeric
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce').fillna(0)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    # Extract year, month, quarter
    df['Year'] = df['Service Date'].dt.year
    df['Month'] = df['Service Date'].dt.month
    df['Month_Name'] = df['Service Date'].dt.strftime('%B')
    df['Quarter'] = df['Service Date'].dt.quarter
    
    # Clean description
    df['Description'] = df['Description'].fillna('Unknown')
    
    # Handle Fixed Fee entries - count as 1 hour for analysis
    df['Original_Hours'] = df['Hours'].copy()
    df.loc[df['Activity Type'] == 'Fixed Fee', 'Hours'] = 1.0
    
    return df

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "AIOGC2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown('<h1 class="main-header">⚖️ OGC Legal AI Automation Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### 🔐 Secure Access Required")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
            st.info("💡 This dashboard contains confidential firm data.")
        return False
    
    elif not st.session_state["password_correct"]:
        st.markdown('<h1 class="main-header">⚖️ OGC Legal AI Automation Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### 🔐 Secure Access Required")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
            st.error("❌ Incorrect password. Please try again.")
        return False
    
    else:
        return True

def main():
    # Check password first
    if not check_password():
        return
    
    st.markdown('<h1 class="main-header">⚖️ OGC Legal AI Automation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Outside GC - AI-Powered Efficiency Analysis (Jan-Sep 2024)")
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state["password_correct"] = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Load data
    try:
        csv_path = './data/SIX_FULL_MOS.csv'
        import os
        if not os.path.exists(csv_path):
            csv_path = '/mnt/user-data/uploads/SIX_FULL_MOS.csv'
        
        df = load_data(csv_path)
        st.sidebar.success(f"✅ Loaded {len(df):,} time entries")
        
        flat_fee_count = (df['Activity Type'] == 'Fixed Fee').sum()
        if flat_fee_count > 0:
            st.sidebar.info(f"ℹ️ {flat_fee_count:,} fixed fee entries counted as 1 hour each")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    years = sorted(df['Year'].dropna().unique())
    selected_years = st.sidebar.multiselect("Select Years", years, default=years)
    
    attorneys = sorted(df['Associated Attorney'].dropna().unique())
    selected_attorneys = st.sidebar.multiselect("Select Attorneys", attorneys, default=[])
    
    # Apply filters
    filtered_df = df[df['Year'].isin(selected_years)]
    if selected_attorneys:
        filtered_df = filtered_df[filtered_df['Associated Attorney'].isin(selected_attorneys)]
    
    # Classify tasks
    with st.spinner("🤖 Analyzing tasks for AI automation potential..."):
        filtered_df[['Task_Type', 'Task_Automation']] = filtered_df['Description'].apply(
            lambda x: pd.Series(classify_task_description(x))
        )
    
    # Calculate automatable hours
    filtered_df['Automatable_Hours'] = filtered_df['Hours'] * filtered_df['Task_Automation']
    filtered_df['Manual_Hours'] = filtered_df['Hours'] - filtered_df['Automatable_Hours']
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview & Analysis",
        "💰 Cost Savings",
        "👥 Attorney Performance",
        "📚 Task Definitions"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("Overview & Task-Level Analysis")
        
        st.markdown("""
        ### 🎯 AI Automation Potential Analysis
        This dashboard analyzes **actual task descriptions** to calculate precise automation potential.
        Each task is classified into 18 categories based on AI capabilities.
        
        **Note:** *Fixed fee entries are counted as 1 hour for analysis purposes.*
        """)
        
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_hours = filtered_df['Hours'].sum()
        automatable_hours = filtered_df['Automatable_Hours'].sum()
        automation_rate = (automatable_hours / total_hours * 100) if total_hours > 0 else 0
        
        with col1:
            st.metric("Total Hours", f"{total_hours:,.0f}")
        
        with col2:
            st.metric("AI-Automatable Hours", f"{automatable_hours:,.0f}", delta=f"{automation_rate:.1f}%")
        
        with col3:
            total_amount = filtered_df['Amount'].sum()
            st.metric("Total Billed", f"${total_amount:,.0f}")
        
        with col4:
            unique_clients = filtered_df['Client Name'].nunique()
            st.metric("Unique Clients", f"{unique_clients:,}")
        
        st.markdown("---")
        
        # Monthly trends
        st.subheader("📅 Monthly Automation Analysis")
        
        monthly_data = filtered_df.groupby(['Year', 'Month', 'Month_Name']).agg({
            'Hours': 'sum',
            'Automatable_Hours': 'sum',
            'Manual_Hours': 'sum'
        }).reset_index().sort_values(['Year', 'Month'])
        monthly_data['Period'] = monthly_data['Month_Name'] + ' ' + monthly_data['Year'].astype(str)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_data['Period'], y=monthly_data['Automatable_Hours'], name='AI-Automatable', mode='lines', line=dict(color='rgb(34, 139, 34)'), stackgroup='one', fillcolor='rgba(34, 139, 34, 0.6)'))
        fig.add_trace(go.Scatter(x=monthly_data['Period'], y=monthly_data['Manual_Hours'], name='Human-Required', mode='lines', line=dict(color='rgb(255, 140, 0)'), stackgroup='one', fillcolor='rgba(255, 140, 0, 0.6)'))
        fig.update_layout(title='Monthly Hours: AI-Automatable vs Human-Required', height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Task type breakdown
        st.subheader("🎯 Automation by Task Type")
        
        task_breakdown = filtered_df.groupby('Task_Type').agg({
            'Hours': 'sum',
            'Automatable_Hours': 'sum',
            'Task_Automation': 'first'
        }).reset_index().sort_values('Automatable_Hours', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(task_breakdown.head(12), x='Automatable_Hours', y='Task_Type', orientation='h',
                        title='Top 12 Task Types by Automatable Hours', color='Task_Automation',
                        color_continuous_scale='RdYlGn', text='Automatable_Hours')
            fig.update_traces(texttemplate='%{text:.0f}h', textposition='outside')
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(y=task_breakdown.head(12)['Task_Type'], x=task_breakdown.head(12)['Hours'],
                                name='Total', orientation='h', marker_color='lightblue'))
            fig.add_trace(go.Bar(y=task_breakdown.head(12)['Task_Type'], x=task_breakdown.head(12)['Automatable_Hours'],
                                name='Automatable', orientation='h', marker_color='darkgreen'))
            fig.update_layout(title='Total vs Automatable Hours', height=500, barmode='overlay',
                            yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Sample tasks
        st.subheader("📝 Sample Task Classifications")
        
        sample_tasks = []
        for task_type in task_breakdown.head(8)['Task_Type']:
            sample = filtered_df[filtered_df['Task_Type'] == task_type].head(2)
            sample_tasks.append(sample)
        
        if sample_tasks:
            sample_df = pd.concat(sample_tasks)
            display_cols = ['Description', 'Task_Type', 'Task_Automation', 'Hours', 'Associated Attorney']
            st.dataframe(sample_df[display_cols].style.format({'Task_Automation': '{:.0%}', 'Hours': '{:.2f}'}),
                        use_container_width=True, height=400)
        
        # Practice Group Analysis
        st.markdown("---")
        st.subheader("⚖️ Practice Group Analysis")
        
        if 'PG' in filtered_df.columns:
            pg_data = filtered_df[filtered_df['PG'].notna()].groupby('PG').agg({
                'Hours': 'sum',
                'Automatable_Hours': 'sum'
            }).reset_index().sort_values('Automatable_Hours', ascending=False).head(10)
            
            fig = px.bar(pg_data, x='PG', y='Automatable_Hours', title='Top 10 Practice Groups by Automatable Hours',
                        color='Automatable_Hours', color_continuous_scale='Greens')
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Cost Savings
    with tab2:
        st.header("💰 Potential Cost Savings with AI")
        
        st.subheader("⚙️ Assumptions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_rate = st.number_input("Average Hourly Rate ($)", min_value=100, max_value=1000, value=450, step=50)
        
        with col2:
            ai_efficiency = st.slider("AI Efficiency Gain (%)", min_value=10, max_value=90, value=60) / 100
        
        with col3:
            ai_cost = st.number_input("AI Cost per Hour ($)", min_value=1, max_value=100, value=10, step=5)
        
        st.markdown("---")
        
        hours_saved = automatable_hours * ai_efficiency
        labor_saved = hours_saved * avg_rate
        ai_total_cost = automatable_hours * ai_cost
        net_savings = labor_saved - ai_total_cost
        roi = (net_savings / ai_total_cost * 100) if ai_total_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Hours Potentially Saved", f"{hours_saved:,.0f}", delta=f"{(hours_saved/total_hours*100):.1f}%")
        
        with col2:
            st.metric("Labor Cost Savings", f"${labor_saved:,.0f}")
        
        with col3:
            st.metric("AI Implementation Cost", f"${ai_total_cost:,.0f}")
        
        with col4:
            st.metric("Net Savings", f"${net_savings:,.0f}", delta=f"ROI: {roi:.0f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💵 Savings by Task Type")
            
            task_breakdown['Hours_Saved'] = task_breakdown['Automatable_Hours'] * ai_efficiency
            task_breakdown['Cost_Savings'] = task_breakdown['Hours_Saved'] * avg_rate
            
            fig = px.bar(task_breakdown.head(10), x='Task_Type', y='Cost_Savings',
                        title='Potential Savings by Task Type', color='Cost_Savings',
                        color_continuous_scale='Greens')
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Top Automation Opportunities")
            
            high_auto = filtered_df[filtered_df['Task_Automation'] >= 0.85].groupby('Description').agg({
                'Hours': 'sum',
                'Task_Automation': 'first'
            }).reset_index().sort_values('Hours', ascending=False).head(15)
            
            if len(high_auto) > 0:
                st.dataframe(high_auto.style.format({'Task_Automation': '{:.0%}', 'Hours': '{:.1f}'}),
                            use_container_width=True, height=400)
    
    # TAB 3: Attorney Performance
    with tab3:
        st.header("👥 Attorney Performance Analysis")
        
        attorney_data = filtered_df.groupby('Associated Attorney').agg({
            'Hours': 'sum',
            'Automatable_Hours': 'sum',
            'Amount': 'sum'
        }).reset_index()
        attorney_data['Automation_Rate'] = (attorney_data['Automatable_Hours'] / attorney_data['Hours'] * 100)
        attorney_data = attorney_data.sort_values('Hours', ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 20 Attorneys by Hours")
            fig = px.bar(attorney_data, x='Associated Attorney', y='Hours', title='Total Hours by Attorney',
                        color='Automation_Rate', color_continuous_scale='RdYlGn')
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Automation Potential by Attorney")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=attorney_data['Associated Attorney'], y=attorney_data['Hours'],
                                name='Total Hours', marker_color='lightblue'))
            fig.add_trace(go.Bar(x=attorney_data['Associated Attorney'], y=attorney_data['Automatable_Hours'],
                                name='Automatable Hours', marker_color='darkgreen'))
            fig.update_layout(height=500, barmode='group', xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Attorney Performance Table")
        
        st.dataframe(attorney_data.style.format({
            'Hours': '{:.0f}',
            'Automatable_Hours': '{:.0f}',
            'Amount': '${:,.0f}',
            'Automation_Rate': '{:.1f}%'
        }), use_container_width=True, height=500)
    
    # TAB 4: Definitions
    with tab4:
        st.header("📚 Task Classification Definitions")
        
        st.markdown("""
        ### 🎯 18 Task Categories by Automation Potential
        Each task is classified based on keywords in the description.
        """)
        
        for category, info in sorted(TASK_LEVEL_AUTOMATION.items(), key=lambda x: x[1]['automation_potential'], reverse=True):
            with st.expander(f"**{category}** - {info['automation_potential']*100:.0f}% Automation Potential"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Keywords:** {', '.join(info['keywords'][:10])}...")
                
                matching = filtered_df[filtered_df['Task_Type'] == category]
                if len(matching) > 0:
                    st.markdown(f"**Found in your data:** {len(matching):,} tasks, {matching['Hours'].sum():.0f} hours")
                    st.write("Sample tasks:")
                    for desc in matching['Description'].head(3):
                        st.write(f"• {desc[:100]}...")

if __name__ == "__main__":
    main()
