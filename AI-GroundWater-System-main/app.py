from flask import Flask, render_template, request, redirect, url_for, session, flash
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import joblib as jb
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
app.secret_key = "secret123"

users = {}

def make_responsive(fig):
    fig.update_layout(
        template='plotly_white',
        title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return pio.to_html(
        fig,
        full_html=False,
        config={
            'responsive': True,
            'scrollZoom': True,
            'displaylogo': False,
            'displayModeBar': False
        }
    )
# DATA
if os.path.exists('groundwater_ml_dataset_cleaned.csv'):
    df = pd.read_csv('groundwater_ml_dataset_cleaned.csv')
else:
    df = pd.DataFrame({'category': ['Safe', 'Critical', 'Overexploited']})

def apply_filters(data):
    state = request.args.get('state')
    category = request.args.get('category')
    
    filtered = data.copy()
    if state and 'state' in filtered.columns:
        filtered = filtered[filtered['state'] == state]
    if category and 'category' in filtered.columns:
        filtered = filtered[filtered['category'] == category]
        
    if filtered.empty:
        return data.copy()
    return filtered

@app.context_processor
def inject_filters():
    # Provide global variables to all templates
    states = sorted(df['state'].unique().tolist()) if not df.empty and 'state' in df else []
    categories = sorted(df['category'].unique().tolist()) if not df.empty and 'category' in df else []
    
    # Calculate global alerts for the Notification Bell
    alerts = []
    if not df.empty and 'category' in df and 'risk_score' in df:
        # Find Over-Exploited districts, sort by risk score
        critical_df = df[df['category'] == 'Over-Exploited'].sort_values('risk_score', ascending=False)
        for _, row in critical_df.head(5).iterrows():
            alerts.append({
                "district": row['district'],
                "state": row['state'],
                "risk": round(row['risk_score'], 2)
            })
            
    return dict(states=states, categories=categories, alerts=alerts)


#============ MODEL LOADED ===================
try:
    model = jb.load('random_forest_model.pkl')
    print("Model Loaded Successfully")
    
    # --- Preprocessing Setup (As per Notebook) ---
    cat_cols = ['state', 'district']
    state_enc = OneHotEncoder(drop='first')
    dummy_cols = state_enc.fit_transform(df[cat_cols]).toarray()
    dummy_df = pd.DataFrame(dummy_cols, columns=state_enc.get_feature_names_out(cat_cols))

    cat_enc = OneHotEncoder(drop='first')
    cat_dummy = cat_enc.fit_transform(df[['category']]).toarray()
    cat_dummy_df = pd.DataFrame(cat_dummy, columns=cat_enc.get_feature_names_out(['category']))

    clean_df = pd.concat([df.reset_index(drop=True), 
                          dummy_df.reset_index(drop=True), 
                          cat_dummy_df.reset_index(drop=True)], axis=1).drop(columns=cat_cols + ['category'])
    
    # Feature names used for training (must match exactly)
    training_features = clean_df.drop(['risk_score'], axis=1).columns.tolist()
    
except Exception as e:
    model = None
    training_features = []
    print(f"Model error: {e}")
# ====================== HOME =======================
@app.route('/')
def home():
    return render_template('index.html')

# AUTH
@app.route('/auth', methods=['GET', 'POST'])
def auth():
    mode = request.args.get('mode', 'register')

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # ================= REGISTER =================
        if mode == 'register':

            if email in users:
                flash("⚠ You are already registered! Please login.", "error")
                return redirect(url_for('auth', mode='login'))

            users[email] = password

            flash("✅ Registered successfully! Please login.", "success")
            return redirect(url_for('auth', mode='login'))

        # ================= LOGIN =================
        if mode == 'login':

            if email not in users:
                flash("❌ User not found! Please register first.", "error")
                return redirect(url_for('auth', mode='register'))

            if users[email] == password:

                session['user_email'] = email
                session['user_name'] = email.split('@')[0]

                flash("🎉 Login successful!", "success")
                return redirect(url_for('home'))
            else:
                flash("❌ Wrong password!", "error")
                return redirect(url_for('auth', mode='login'))

    return render_template('auth.html', mode=mode)

# PROTECTION
def login_required():
    return 'user_email' not in session

@app.route('/about')
def about():
    if login_required():
        return redirect(url_for('auth', mode='login'))
    return render_template('about.html')

# ================= UPLOAD DATA =================
@app.route('/upload', methods=['POST'])
def upload_data():
    if login_required():
        return redirect(url_for('auth', mode='login'))
        
    if 'dataset' not in request.files:
        flash("No file part", "error")
        return redirect(request.referrer or url_for('home'))
        
    file = request.files['dataset']
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(request.referrer or url_for('home'))
        
    if file and file.filename.endswith('.csv'):
        # Save the file and replace the existing dataset
        filepath = os.path.join(os.getcwd(), 'groundwater_ml_dataset_cleaned.csv')
        file.save(filepath)
        
        # Reload global dataframe
        global df
        df = pd.read_csv(filepath)
        
        flash("✅ Dataset imported and updated successfully!", "success")
        return redirect(request.referrer or url_for('overview'))
    else:
        flash("❌ Invalid file format. Please upload a CSV.", "error")
        return redirect(request.referrer or url_for('home'))


# Overview Route ------------------
@app.route('/overview')
def overview():
    if login_required():
        return redirect(url_for('auth', mode='login'))

    df_copy = apply_filters(df)
    df_copy['water_balance'] = df_copy['annual_recharge'] - df_copy['annual_extraction']


    # ================= GRAPH 1 =================
    fig1 = px.pie(df_copy, names='category',
                title='Category Distribution',
                color_discrete_sequence=px.colors.sequential.Blugrn_r)
    graph1 = make_responsive(fig1)

    # ================= GRAPH 2 =================
    district_count = df_copy.groupby('state')['district'].nunique().reset_index()
    district_count.columns = ['state', 'district_count']

    fig2 = px.treemap(district_count,
                    path=['state'],
                    values='district_count',
                    title='District Count per State',
                    color='district_count',
                    color_continuous_scale=px.colors.sequential.Tealgrn)
    graph2 = make_responsive(fig2)

    # ================= GRAPH 3 =================
    avg_values = df_copy.groupby('category')[['annual_recharge', 'annual_extraction']].mean().reset_index()

    fig3 = px.bar(avg_values,
                x='category',
                y=['annual_recharge', 'annual_extraction'],
                barmode='group',
                title='Avg Recharge vs Extraction',
                color_discrete_sequence=px.colors.sequential.Blugrn_r)
    graph3 = make_responsive(fig3)

    # ================= GRAPH 4 =================
    avg_risk = df_copy.groupby('category')['risk_score'].mean().reset_index()

    fig4 = px.bar(avg_risk,
                x='category',
                y='risk_score',
                color='risk_score',
                color_continuous_scale=px.colors.sequential.Blugrn,
                title='Average Risk Score')
    graph4 = make_responsive(fig4)


    # ================= GRAPH 5 (FIXED) =================

# 🔥 convert to numeric (fix black graph issue)
    df_copy['stress_level'] = pd.to_numeric(df_copy['stress_level'], errors='coerce')

    fig5 = px.histogram(
            df_copy.dropna(subset=['stress_level']),
            x='stress_level',
            nbins=25,
            color_discrete_sequence=px.colors.sequential.Blugrn_r
)

    graph5 = make_responsive(fig5)

    # ================= GRAPH 6 =================
    avg_util = df_copy.groupby('category')['utilization_rate'].mean().reset_index()

    fig6 = px.pie(avg_util,
                names='category',
                values='utilization_rate',
                hole=0.5,
                title='Utilization Rate',
                color_discrete_sequence=px.colors.sequential.Blugrn_r)
    graph6 = make_responsive(fig6)

    # ================= GRAPH 7 =================
    fig7 = px.box(df_copy,
                x='category',
                y='annual_extraction',
                color='category',
                title='Extraction Distribution',
                color_discrete_sequence=px.colors.sequential.Tealgrn)
    graph7 = make_responsive(fig7)

    # ================= GRAPH 8 =================
    fig8 = px.histogram(df_copy,
                        x='water_balance',
                        nbins=25,
                        title='Water Balance Distribution',
                        color_discrete_sequence=px.colors.sequential.Blugrn_r)
    graph8 = make_responsive(fig8)

    # ================= GRAPH 9 =================
    heat_df = df_copy.groupby(['state', 'category'])['annual_extraction'].mean().reset_index()

    fig9 = px.density_heatmap(heat_df,
                            x='state',
                            y='category',
                            z='annual_extraction',
                            color_continuous_scale='Tealgrn',
                            title='State-wise Category Heatmap')
    graph9 = make_responsive(fig9)

    # ================= GRAPH 10 =================
    trend_df = df_copy.groupby(['year', 'category'])['annual_extraction'].mean().reset_index()

    fig10 = px.bar(trend_df,
                x='year',
                y='annual_extraction',
                color='category',
                barmode='group',
                title='Yearly Extraction Trend',
                color_discrete_sequence=px.colors.sequential.Blugrn_r)
    graph10 = make_responsive(fig10)

    # ================= GRAPH 11 =================
    top_districts = df_copy.loc[df_copy.groupby('category')['annual_extraction'].idxmax()]

    fig11 = px.bar(top_districts,
                x='category',
                y='annual_extraction',
                color='district',
                title='Top District per Category',
                color_discrete_sequence=px.colors.sequential.Blugrn_r)
    graph11 = make_responsive(fig11)

    # ================= GRAPH 12 (TABLE) =================
    df_clean = df_copy.dropna(subset=['state', 'district'])

    fig12 = go.Figure(data=[go.Table(
        header=dict(
            values=['State','District','Recharge','Extraction','Risk','Stress'],
            fill_color='#023047',
            font=dict(color='white'),
            align='center'
        ),
        cells=dict(
            values=[
                df_clean['state'],
                df_clean['district'],
                df_clean['annual_recharge'].round(2),
                df_clean['annual_extraction'].round(2),
                df_clean['risk_score'].round(2),
                df_clean['stress_level']
            ],
            fill_color='#E0FBFC',
            align='center'
        )
    )])

    graph12 = make_responsive(fig12)

    # ================= RETURN =================
    kpis = [
        {"title": "Total Districts Monitored", "value": f"{df_copy['district'].nunique()}"},
        {"title": "Avg Utilization Rate", "value": f"{df_copy['utilization_rate'].mean():.1f}%" if 'utilization_rate' in df_copy else "N/A"},
        {"title": "Avg Water Balance", "value": f"{df_copy['water_balance'].mean():,.2f}" if 'water_balance' in df_copy else "N/A"},
        {"title": "Safe Districts", "value": f"{len(df_copy[df_copy['category'] == 'Safe'])}"}
    ]
    return render_template(
        'overview.html',
        kpis=kpis,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3,
        graph4=graph4,
        graph5=graph5,
        graph6=graph6,
        graph7=graph7,
        graph8=graph8,
        graph9=graph9,
        graph10=graph10,
        graph11=graph11,
        graph12=graph12
    )
# <---------------------- Risk Intelligence page -------------------->
@app.route('/risk-intelligence')
def risk_intelligence():
    if login_required():
        return redirect(url_for('auth', mode='login'))

    df_copy = apply_filters(df)

    # ================= 1 TREEMAP: District Risk =================
    district_tree = df_copy.groupby(['state', 'district'])['risk_score'].mean().reset_index()

    fig1 = px.treemap(
        district_tree,
        path=['state', 'district'],
        values='risk_score',
        color='risk_score',
        color_continuous_scale='Tealgrn',
        title='District-wise Risk Structure'
    )
    graph1 = make_responsive(fig1)

    # ================= 2 DONUT: Risk Band =================
    df_copy['risk_band'] = pd.cut(df_copy['risk_score'], bins=[0, 0.33, 0.66, 1],
                                labels=['Low', 'Medium', 'High'])

    donut = df_copy['risk_band'].value_counts().reset_index()
    donut.columns = ['risk_band', 'count']

    fig2 = px.pie(
        donut,
        names='risk_band',
        values='count',
        hole=0.6,
        title='Risk Level Distribution',
        color_discrete_sequence=px.colors.sequential.Blugrn
    )
    graph2 = make_responsive(fig2)

    # ================= 3 BAR: Category Risk =================
    avg_df = df_copy.groupby('category')['risk_score'].mean().reset_index()

    fig3 = px.bar(
        avg_df,
        x='category',
        y='risk_score',
        color='category',
        title='Average Risk Score by Category',
        color_discrete_sequence=px.colors.sequential.Blugrn_r
    )
    graph3 = make_responsive(fig3)

    # ================= 4 BOX: Extraction Ratio =================
    fig4 = px.box(
    df,
    x='category',
    y='extraction_ratio',
    color='category',
    color_discrete_sequence=px.colors.sequential.Blugrn_r,
    title='Extraction Ratio Distribution by Category'
)
    graph4 = make_responsive(fig4)
    # ================= 5 SCATTER: Stress vs Risk =================
    fig5 = px.scatter(
    df,
    x='stress_level',
    y='risk_score',
    color='risk_score',
    color_continuous_scale='Blugrn',
    size='risk_score',
    title='Risk Score vs Stress Level'
)
    graph5 = make_responsive(fig5)
    # ================= 6 HISTOGRAM: Risk Score =================
    fig6 = px.histogram(
        df_copy,
        x='risk_score',
        nbins=25,
        title='Risk Score Distribution',
        color_discrete_sequence=px.colors.sequential.Blugrn_r
    )
    graph6 = make_responsive(fig6)

    # ================= 7 TOP DISTRICTS =================
    top_districts = df_copy.sort_values('risk_score', ascending=False).head(10)

    fig7 = px.treemap(
    top_districts,
    path=['district'],
    values='risk_score',
    color='risk_score',
    color_continuous_scale='Tealgrn',
    title='Top 10 High Risk Districts'
)
    graph7 = make_responsive(fig7)
    # ================= 8 HIGH vs LOW RISK =================
    df_copy['risk_level'] = df_copy['risk_score'].apply(
        lambda x: 'High Risk' if x > df_copy['risk_score'].mean() else 'Low Risk'
    )

    risk_count = df_copy['risk_level'].value_counts().reset_index()
    risk_count.columns = ['risk_level', 'count']

    fig8 = px.pie(
        risk_count,
        names='risk_level',
        values='count',
        title='High vs Low Risk Areas',
        color_discrete_sequence=px.colors.sequential.Blugrn_r
    )
    graph8 = make_responsive(fig8)
    # ================= 9 CATEGORY RISK =================
    avg_risk = df_copy.groupby('category')['risk_score'].mean().reset_index()

    fig9 = px.bar(
        avg_risk,
        x='category',
        y='risk_score',
        color='risk_score',
        title='Risk Score by Category',
        color_continuous_scale='Blugrn'
    )
    graph9 = make_responsive(fig9)

    kpis = [
        {"title": "Highest Risk Score", "value": f"{df_copy['risk_score'].max():.2f}" if 'risk_score' in df_copy else "N/A"},
        {"title": "Average Risk Score", "value": f"{df_copy['risk_score'].mean():.2f}" if 'risk_score' in df_copy else "N/A"},
        {"title": "High Risk Districts", "value": f"{len(df_copy[df_copy['risk_score'] > 0.66])}" if 'risk_score' in df_copy else "N/A"},
        {"title": "Avg Stress Level", "value": f"{df_copy['stress_level'].mean():.2f}" if 'stress_level' in df_copy else "N/A"}
    ]
    return render_template(
        'risk_intelligence.html',
        kpis=kpis,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3,
        graph4=graph4,
        graph5=graph5,
        graph6=graph6,
        graph7=graph7,
        graph8=graph8,
        graph9=graph9
    )


#==================== Water-Balance Page ====================>
@app.route('/water-balance')
def water_balance():
    if login_required():
        return redirect(url_for('auth', mode='login'))

    df_copy = apply_filters(df)

    # ================= GRAPH 1 =================
    df_copy['water_balance'] = df_copy['annual_recharge'] - df_copy['annual_extraction']
    df_copy['balance_type'] = df_copy['water_balance'].apply(lambda x: 'Surplus' if x > 0 else 'Deficit')
    df_copy['balance_magnitude'] = df_copy['water_balance'].abs()

    tree_df = df_copy.groupby(['state', 'district', 'balance_type'])['balance_magnitude'].mean().reset_index()

    fig1 = px.treemap(
        tree_df,
        path=['state', 'district', 'balance_type'],
        values='balance_magnitude',
        color='balance_magnitude',
        color_continuous_scale='Tealgrn',
        title='Water Balance Treemap (State → District)'
    )
    graph1 = make_responsive(fig1)

    # ================= GRAPH 2 =================
    df_copy['stress_category'] = pd.cut(
        df_copy['stress_level'],
        bins=[0, 0.3, 0.6, 1],
        labels=['Low Stress', 'Moderate Stress', 'High Stress']
    )

    overall = df_copy['stress_category'].value_counts().reindex(
        ['Low Stress','Moderate Stress','High Stress']
    ).fillna(0)

    fig2 = go.Figure()
    fig2.add_trace(go.Pie(
        labels=overall.index,
        values=overall.values,
        hole=0.55,
        marker=dict(colors=['#2CA58D', '#1F7A8C', '#0B3C5D'])
    ))
    graph2 = make_responsive(fig2)

    # ================= GRAPH 3 =================
    corr = df_copy.corr(numeric_only=True)

    fig3 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale=px.colors.sequential.Tealgrn,
        title='Correlation Heatmap'
    )
    graph3 = make_responsive(fig3)

    # ================= GRAPH 4 =================
    fig4 = px.box(
        df_copy,
        x='category',
        y='annual_recharge',
        color='category',
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        title='Recharge per Category'
    )
    graph4 = make_responsive(fig4)

    # ================= GRAPH 5 =================
    df_copy['sustainability'] = df_copy['extraction_ratio'].apply(
        lambda x: 'Unsustainable' if x > 1 else 'Sustainable'
    )

    fig5 = px.pie(
        df_copy,
        names='sustainability',
        color_discrete_sequence=px.colors.sequential.Blugrn_r,
        title='Sustainable vs Unsustainable Zones'
    )
    graph5 = make_responsive(fig5)

    # ================= GRAPH 6 =================
    balance_count = df_copy['balance_type'].value_counts().reset_index()
    balance_count.columns = ['balance_type', 'count']

    fig6 = px.bar(
        balance_count,
        x='balance_type',
        y='count',
        color='balance_type',
        color_discrete_sequence=['#2CA58D', '#0B3C5D'],
        title='Water Surplus vs Deficit'
    )
    graph6 = make_responsive(fig6)

    # ================= GRAPH 7 =================
    df_copy['overuse_intensity'] = df_copy['annual_extraction'] - df_copy['annual_recharge']

    top_overuse = df_copy.sort_values('overuse_intensity', ascending=False).head(10)

    fig7 = px.treemap(
        top_overuse,
        path=['state', 'district'],
        values='overuse_intensity',
        color='overuse_intensity',
        color_continuous_scale='Tealgrn',
        title='Top 10 Overuse Districts'
    )
    graph7 = make_responsive(fig7)

    # ================= GRAPH 8 =================
    fig8 = px.bar(
        df_copy.groupby('balance_type').size().reset_index(name='count'),
        x='balance_type',
        y='count',
        color='balance_type',
        color_discrete_sequence=['#2CA58D', '#0B3C5D'],
        title='District-wise Water Balance'
    )
    graph8 = make_responsive(fig8)

    # ================= GRAPH 9 =================
    util_avg = df_copy.groupby('category')['utilization_rate'].mean().reset_index()

    fig9 = px.bar(
        util_avg,
        x='category',
        y='utilization_rate',
        color='category',
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        title='Utilization Rate by Category'
    )
    graph9 = make_responsive(fig9)

    # ================= GRAPH 10 =================
    fig10 = px.box(
        df_copy,
        x='category',
        y='extraction_ratio',
        color='category',
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        title='Extraction Ratio Distribution'
    )
    graph10 = make_responsive(fig10)

    # ================= GRAPH 11 =================
    fig11 = px.scatter(
        df_copy,
        x='annual_recharge',
        y='annual_extraction',
        color='risk_score',
        color_continuous_scale=px.colors.sequential.Blugrn,
        title='Recharge vs Extraction'
    )
    graph11 = make_responsive(fig11)

    kpis = [
        {"title": "Total Surplus (MCM)", "value": f"{df_copy[df_copy['water_balance'] > 0]['water_balance'].sum():,.2f}" if 'water_balance' in df_copy else "N/A"},
        {"title": "Total Deficit (MCM)", "value": f"{abs(df_copy[df_copy['water_balance'] < 0]['water_balance'].sum()):,.2f}" if 'water_balance' in df_copy else "N/A"},
        {"title": "Districts in Deficit", "value": f"{len(df_copy[df_copy['water_balance'] < 0])}" if 'water_balance' in df_copy else "N/A"},
        {"title": "Avg Extraction Ratio", "value": f"{df_copy['extraction_ratio'].mean():.2f}" if 'extraction_ratio' in df_copy else "N/A"}
    ]
    return render_template(
        'water_Balance.html',
        kpis=kpis,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3,
        graph4=graph4,
        graph5=graph5,
        graph6=graph6,
        graph7=graph7,
        graph8=graph8,
        graph9=graph9,
        graph10=graph10,
        graph11=graph11
    )


# <========================Geo spatial pgae 4 ====================>
@app.route('/geo-spatial')
def geo_spatial():
    if login_required():
        return redirect(url_for('auth', mode='login'))

    df_copy = apply_filters(df)

    # ================= GRAPH 1 =================
    top_rec = df_copy.groupby('state')['annual_recharge'].sum().reset_index()
    top_rec = top_rec.sort_values('annual_recharge', ascending=False).head(10)

    fig1 = px.treemap(
        top_rec,
        path=['state'],
        values='annual_recharge',
        color='annual_recharge',
        color_continuous_scale=px.colors.sequential.Blugrn_r,
        title='Top 10 States by Recharge'
    )
    graph1 = make_responsive(fig1)

    # ================= GRAPH 2 =================
    district_risk = df_copy.groupby(['district'])['risk_score'].mean().reset_index().nlargest(30, 'risk_score')

    fig2 = px.bar(
        district_risk,
        x='district',
        y='risk_score',
        color='risk_score',
        color_continuous_scale='Tealgrn',
        title='District-wise Risk Score'
    )

    graph2 = make_responsive(fig2)

    # ================= GRAPH 3 =================
    fig3 = px.scatter(
        df_copy,
        x='annual_recharge',
        y='annual_extraction',
        color='risk_score',
        size='risk_score',
        hover_name='district',
        color_continuous_scale='Blugrn',
        title='Recharge vs Extraction (District Level)'
    )
    graph3 = make_responsive(fig3)

    # ================= GRAPH 4 =================
    avg_risk_state = df_copy.groupby('state')['risk_score'].mean().reset_index()

    fig4 = px.bar(
        avg_risk_state,
        x='state',
        y='risk_score',
        color='risk_score',
        color_continuous_scale='Tealgrn',
        title='State-wise Avg Risk Score'
    )
    graph4 = make_responsive(fig4)

    # ================= GRAPH 5 =================
    top_districts = df_copy.sort_values('risk_score', ascending=False).head(10)

    fig5 = px.treemap(
        top_districts,
        path=['district'],
        values='risk_score',
        color='risk_score',
        color_continuous_scale='Blugrn',
        title='Top 10 High Risk Districts'
    )
    graph5 = make_responsive(fig5)



    # ================= GRAPH 6 =================
    fig6 = px.scatter(
        df_copy,
        x='annual_extraction',
        y='risk_score',
        size='utilization_rate',
        color='risk_score',
        hover_name='district',
        color_continuous_scale='Blugrn_r',
        title='Extraction vs Risk (Bubble Chart)'
    )
    graph6 = make_responsive(fig6)

    # ================= GRAPH 7 =================
    fig7 = px.scatter(
        df_copy,
        x='extraction_ratio',
        y='risk_score',
        color='risk_score',
        hover_name='district',
        color_continuous_scale='Blugrn_r',
        title='Extraction Ratio vs Risk'
    )
    graph7 = make_responsive(fig7)

    # ================= GRAPH 8 =================
    sun_df = df_copy.groupby(['state','district','category'])['risk_score'].mean().reset_index()

    fig8 = px.sunburst(
        sun_df,
        path=['state','district','category'],
        values='risk_score',
        color='risk_score',
        color_continuous_scale='Tealgrn',
        title='Risk Distribution Sunburst'
    )
    graph8 = make_responsive(fig8)

    # ================= GRAPH 9 =================
    top_ext = df_copy.groupby('state')['annual_extraction'].sum().reset_index()
    top_ext = top_ext.sort_values('annual_extraction', ascending=False).head(10)

    fig9 = px.bar(
        top_ext,
        x='state',
        y='annual_extraction',
        color='annual_extraction',
        color_continuous_scale='Blugrn_r',
        title='Top 10 States by Extraction'
    )
    graph9 = make_responsive(fig9)

    top_extract_state = df_copy.groupby('state')['annual_extraction'].sum().idxmax() if not df_copy.empty else "N/A"
    kpis = [
        {"title": "Top State (Extraction)", "value": f"{top_extract_state}"},
        {"title": "Total Recharge (MCM)", "value": f"{df_copy['annual_recharge'].sum():,.2f}" if 'annual_recharge' in df_copy else "N/A"},
        {"title": "Monitored States", "value": f"{df_copy['state'].nunique()}"},
        {"title": "Monitored Districts", "value": f"{df_copy['district'].nunique()}"}
    ]
    return render_template(
        'geo_spatial.html',
        kpis=kpis,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3,
        graph4=graph4,
        graph5=graph5,
        graph6=graph6,
        graph7=graph7,
        graph8=graph8,
        graph9=graph9,
        
    )

# ========================Geo-Spatial Page 5 =======================>
@app.route('/crisis-detection')
def crisis_detection():
    if login_required():
        return redirect(url_for('auth', mode='login'))

    df_copy = apply_filters(df)

    # ================= GRAPH 1 =================
    district_risk = df_copy.groupby(['district'])['risk_score'].mean().reset_index().nlargest(10, 'risk_score')

    fig1 = px.bar(
        district_risk,
        x='district',
        y='risk_score',
        color='risk_score',
        color_continuous_scale='Tealgrn',
        title='Top 10 Crisis Districts'
    )

    graph1 = make_responsive(fig1)

    # ================= GRAPH 2 =================
    tree_df = df_copy.groupby(['state','district']).agg({
        'extraction_ratio': 'mean',
        'risk_score': 'mean'
    }).reset_index()

    fig2 = px.treemap(
        tree_df,
        path=['state', 'district'],
        values='extraction_ratio',
        color='risk_score',
        color_continuous_scale='Tealgrn',
        title='State → District Risk Treemap'
    )
    graph2 = make_responsive(fig2)

    # ================= GRAPH 3 =================
    top_extract = df_copy.groupby('district')[['extraction_ratio', 'risk_score']].mean().reset_index().nlargest(30, 'extraction_ratio')

    fig3 = px.bar(
        top_extract,
        x='district',
        y='extraction_ratio',
        color='risk_score',
        color_continuous_scale='Tealgrn',
        title='Extraction Ratio & Risk'
    )

    graph3 = make_responsive(fig3)

    # ================= GRAPH 4 =================
    fig4 = px.scatter(
        df_copy,
        x='extraction_ratio',
        y='risk_score',
        color='risk_score',
        color_continuous_scale='Blugrn_r',
        hover_name='district',
        title='Extraction Ratio vs Risk'
    )
    graph4 =  make_responsive(fig4)

    # ================= GRAPH 5 =================
    threshold = df_copy['risk_score'].mean()
    df_copy['alert'] = df_copy['risk_score'].apply(lambda x: 'Critical' if x > threshold else 'Safe')

    alert_count = df_copy['alert'].value_counts().reset_index()
    alert_count.columns = ['alert', 'count']

    fig5 = px.pie(
        alert_count,
        names='alert',
        values='count',
        hole=0.5,
        color_discrete_sequence=['#2CA58D', '#0B3C5D'],
        title='High Risk Alert Zones'
    )
    graph5 = make_responsive(fig5)

    # ================= GRAPH 6 =================
    df_copy['over_exploited'] = df_copy['extraction_ratio'].apply(
        lambda x: 'Over-Exploited' if x > 1 else 'Normal'
    )

    count_exploit = df_copy['over_exploited'].value_counts().reset_index()
    count_exploit.columns = ['status', 'count']

    fig6 = px.bar(
        count_exploit,
        x='status',
        y='count',
        color='status',
        color_discrete_sequence=['#2CA58D', '#0B3C5D'],
        title='Over-Exploited District Count'
    )
    graph6 = make_responsive(fig6)

    # ================= GRAPH 7 =================
    fig7 = px.scatter(
        df_copy,
        x='stress_level',
        y='risk_score',
        color='risk_score',
        size='risk_score',
        hover_name='district',
        color_continuous_scale='Blugrn',
        title='Stress Level vs Risk'
    )
    graph7 = make_responsive(fig7)

    # ================= GRAPH 8 =================
    top_crisis = df_copy.sort_values('risk_score', ascending=False).head(10)

    fig8 = px.bar(
        top_crisis,
        x='district',
        y='risk_score',
        color='risk_score',
        color_continuous_scale='Blugrn',
        title='Top Crisis Districts'
    )
    graph8 = make_responsive(fig8)

    # ================= GRAPH 9 =================
    fig9 = px.pie(
        df_copy,
        names='alert',
        color_discrete_sequence=px.colors.sequential.Blugrn_r,
        title='Alert Distribution'
    )
    graph9 = make_responsive(fig9)

    # ================= GRAPH 10 =================
    df_copy['risk_level'] = pd.cut(df_copy['risk_score'], bins=3, labels=['Low','Medium','High'])

    risk_seg = df_copy['risk_level'].value_counts().reset_index()
    risk_seg.columns = ['risk_level', 'count']

    fig10 = px.bar(
        risk_seg,
        x='risk_level',
        y='count',
        color='risk_level',
        color_discrete_sequence=['#2CA58D','#1F7A8C','#0B3C5D'],
        title='Risk Level Segmentation'
    )
    graph10 = make_responsive(fig10)

    # ================= GRAPH 11 =================
    threshold = df_copy['risk_score'].mean()
    df_copy['alert_flag'] = df_copy['risk_score'] > threshold

    alert_df = df_copy.groupby('alert_flag').size().reset_index(name='count')

    fig11 = px.bar(
        alert_df,
        x='alert_flag',
        y='count',
        color='count',
        color_continuous_scale='Tealgrn',
        title='Risk Threshold Breach'
    )
    graph11 = make_responsive(fig11)

    # ================= GRAPH 12 =================
    fig12 = px.scatter(
        df_copy,
        x='stress_level',
        y='risk_score',
        color='risk_score',
        color_continuous_scale='Tealgrn',
        hover_name='district',
        title='Stress vs Risk Clustering'
    )
    graph12 = make_responsive(fig12)

    # ================= GRAPH 13 (GAUGE: National Avg Risk) =================
    avg_risk = df_copy['risk_score'].mean() if not df_copy.empty and 'risk_score' in df_copy else 0
    fig13 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_risk,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Average Risk Score"},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.33], 'color': "#2CA58D"},
                {'range': [0.33, 0.66], 'color': "#FFB300"},
                {'range': [0.66, 1.0], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': avg_risk
            }
        }
    ))
    graph13 = make_responsive(fig13)

    mean_risk = df_copy['risk_score'].mean() if 'risk_score' in df_copy else 0
    most_critical = df_copy.loc[df_copy['risk_score'].idxmax()]['district'] if not df_copy.empty and 'risk_score' in df_copy else "N/A"
    
    kpis = [
        {"title": "Over-Exploited Districts", "value": f"{len(df_copy[df_copy['extraction_ratio'] > 1])}" if 'extraction_ratio' in df_copy else "N/A"},
        {"title": "Crisis Alerts (Risk > Mean)", "value": f"{len(df_copy[df_copy['risk_score'] > mean_risk])}" if 'risk_score' in df_copy else "N/A"},
        {"title": "Max Stress Level", "value": f"{df_copy['stress_level'].max():.2f}" if 'stress_level' in df_copy else "N/A"},
        {"title": "Most Critical District", "value": f"{most_critical}"}
    ]
    return render_template(
        'crisis_detection.html',
        kpis=kpis,
        graph1=graph1,
        graph2=graph2,
        graph3=graph3,
        graph4=graph4,
        graph5=graph5,
        graph6=graph6,
        graph7=graph7,
        graph8=graph8,
        graph9=graph9,
        graph10=graph10,
        graph11=graph11,
        graph12=graph12,
        graph13=graph13
    )



# Prediction route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if login_required():
        return redirect(url_for('auth', mode='login'))

    result = None
    gauge_chart = None
    feature_chart = None
    history_chart = None
    ai_suggestion = ""
    
    # Get unique values for dropdowns mapping state to districts
    state_district_map = df.groupby('state')['district'].unique().apply(list).to_dict() if 'state' in df.columns else {}

    if request.method == 'POST':
        try:
            # 1. Collect Primary Inputs
            state = request.form.get('state')
            district = request.form.get('district')
            annual_recharge = float(request.form.get('annual_recharge'))
            annual_extraction = float(request.form.get('annual_extraction'))
            category = request.form.get('category')
            year = 2026 # Default current year

            # 2. Smart Calculation / Data Imputation
            district_data = df[df['district'] == district]
            
            if not district_data.empty:
                extractable_resource = district_data['extractable_resource'].mean()
                stage_of_development = district_data['stage_of_development'].mean()
                stress_level = district_data['stress_level'].astype(float).mean()
            else:
                extractable_resource = annual_recharge * 0.8
                stage_of_development = df['stage_of_development'].mean()
                stress_level = df['stress_level'].astype(float).mean()

            # Dynamic calculations based on user input
            extraction_ratio = annual_extraction / annual_recharge if annual_recharge > 0 else 0
            utilization_rate = annual_extraction / extractable_resource if extractable_resource > 0 else 0

            # 3. Create DataFrame for processing
            input_dict = {
                'annual_recharge': annual_recharge,
                'extractable_resource': extractable_resource,
                'annual_extraction': annual_extraction,
                'stage_of_development': stage_of_development,
                'extraction_ratio': extraction_ratio,
                'utilization_rate': utilization_rate,
                'stress_level': stress_level,
                'year': year
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # 4. Same Preprocessing as Training
            cat_input = pd.DataFrame([[state, district]], columns=['state', 'district'])
            dummy_cols = state_enc.transform(cat_input).toarray()
            dummy_df = pd.DataFrame(dummy_cols, columns=state_enc.get_feature_names_out(['state', 'district']))
            
            cat_val_input = pd.DataFrame([[category]], columns=['category'])
            cat_dummy_val = cat_enc.transform(cat_val_input).toarray()
            cat_dummy_df_val = pd.DataFrame(cat_dummy_val, columns=cat_enc.get_feature_names_out(['category']))
            
            final_input = pd.concat([input_df, dummy_df, cat_dummy_df_val], axis=1)
            final_input = final_input.reindex(columns=training_features, fill_value=0)
            
            # 5. Predict
            if model:
                prediction_val = model.predict(final_input)[0]
                result = round(prediction_val, 4)
                
                # --- ENHANCEMENTS ---
                
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result,
                    title = {'text': "Predicted Risk Score", 'font': {'color': '#0B3C5D', 'size': 24, 'weight': 'bold'}},
                    number = {'font': {'color': '#0B3C5D', 'size': 40, 'weight': 'bold'}},
                    gauge = {
                        'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#0B3C5D", 'thickness': 0.25},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 0.33], 'color': "rgba(16, 185, 129, 0.6)"}, # Light Green
                            {'range': [0.33, 0.66], 'color': "rgba(245, 158, 11, 0.6)"}, # Light Orange
                            {'range': [0.66, 1.0], 'color': "rgba(239, 68, 68, 0.6)"} # Light Red
                        ],
                        'threshold': {
                            'line': {'color': "#0B3C5D", 'width': 8},
                            'thickness': 0.85,
                            'value': result
                        }
                    }
                ))
                gauge_chart = make_responsive(fig_gauge)
                
                # Feature Chart (Mock Impact based on inputs)
                impact_df = pd.DataFrame({
                    'Feature': ['Extraction Ratio', 'Stress Level', 'Recharge (Deficit)', 'Total Extraction'],
                    'Impact': [extraction_ratio * 40, stress_level * 30, (1/annual_recharge)*1000 if annual_recharge>0 else 0, annual_extraction * 0.05]
                })
                fig_feat = px.bar(impact_df, x='Impact', y='Feature', orientation='h', title='Feature Impact Analysis', color_discrete_sequence=['#0B3C5D'])
                feature_chart = make_responsive(fig_feat)
                
                # Radar Chart (Advanced Comparison)
                categories_radar = ['Recharge', 'Extraction', 'Stress Level', 'Extraction Ratio']
                national_recharge_max = df['annual_recharge'].max() if not df.empty and df['annual_recharge'].max() > 0 else 1
                national_extract_max = df['annual_extraction'].max() if not df.empty and df['annual_extraction'].max() > 0 else 1
                
                # Normalize values for radar (0 to 1 scale)
                r_district = [
                    annual_recharge / national_recharge_max,
                    annual_extraction / national_extract_max,
                    stress_level,
                    extraction_ratio if extraction_ratio <= 1 else 1
                ]
                r_national = [
                    (df['annual_recharge'].mean() / national_recharge_max) if not df.empty else 0.5,
                    (df['annual_extraction'].mean() / national_extract_max) if not df.empty else 0.5,
                    df['stress_level'].astype(float).mean() if not df.empty and 'stress_level' in df else 0.5,
                    df['extraction_ratio'].mean() if not df.empty and df['extraction_ratio'].mean() <= 1 else 1
                ]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_district, theta=categories_radar, fill='toself', name='Your Input', 
                    line_color='#FFB300', fillcolor='rgba(255, 179, 0, 0.6)', line_width=3
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_national, theta=categories_radar, fill='toself', name='National Avg', 
                    line_color='#0B3C5D', fillcolor='rgba(11, 60, 93, 0.4)', line_width=3
                ))
                fig_radar.update_layout(
                    title={'text': "Input vs National Average", 'font': {'color': '#0B3C5D', 'size': 18, 'weight': 'bold'}},
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                        angularaxis=dict(tickfont=dict(size=13, color='#0B3C5D', weight='bold'))
                    ),
                    showlegend=True,
                    template='plotly_white',
                    margin=dict(l=50, r=50, t=60, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                
                history_chart = pio.to_html(
                    fig_radar, full_html=False, 
                    config={'responsive': True, 'displayModeBar': False}
                )
                    
                # AI Suggestion
                if result > 0.66:
                    ai_suggestion = "Immediate 20% reduction in industrial extraction recommended. Mandatory rainwater harvesting should be implemented to restore balance."
                elif result > 0.33:
                    ai_suggestion = "Groundwater extraction is approaching unsustainable levels. Consider optimizing agricultural usage and monitoring water tables closely."
                else:
                    ai_suggestion = "Current extraction is sustainable. Focus on maintaining natural recharge zones to prevent future stress."

            else:
                flash("Model not loaded!", "error")
                
        except Exception as e:
            flash(f"Error: Fill all fields correctly. ({e})", "error")
            print(f"Prediction error: {e}")

    return render_template('prediction.html', prediction=result, state_district_map=state_district_map, gauge_chart=gauge_chart, feature_chart=feature_chart, history_chart=history_chart, ai_suggestion=ai_suggestion)

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ✅ DASHBOARD ROUTE
@app.route('/dashboard')
def dashboard():
    if login_required():
        return redirect(url_for('auth', mode='login'))
        
    df_copy = apply_filters(df)
    kpis = [
        {"title": "Total Districts", "value": f"{df_copy['district'].nunique()}"},
        {"title": "Total Recharge (MCM)", "value": f"{df_copy['annual_recharge'].sum():,.2f}" if 'annual_recharge' in df_copy else "N/A"},
        {"title": "Total Extraction (MCM)", "value": f"{df_copy['annual_extraction'].sum():,.2f}" if 'annual_extraction' in df_copy else "N/A"},
        {"title": "Avg Risk Score", "value": f"{df_copy['risk_score'].mean():.2f}" if 'risk_score' in df_copy else "N/A"}
    ]
    
    # New Graphs for Dashboard Home
    # Graph 1: State-wise Extraction vs Recharge (Grouped Bar)
    if not df_copy.empty and 'annual_recharge' in df_copy:
        state_agg = df_copy.groupby('state')[['annual_recharge', 'annual_extraction']].sum().reset_index()
        fig1 = px.bar(state_agg, x='state', y=['annual_recharge', 'annual_extraction'], barmode='group',
                      color_discrete_sequence=['#FFB300', '#0B3C5D'])
        graph1 = make_responsive(fig1)
    else:
        graph1 = None
        
    # Graph 2: Donut Chart Risk Levels
    if not df_copy.empty and 'risk_score' in df_copy:
        risk_cuts = pd.cut(df_copy['risk_score'], bins=[-1, 0.33, 0.66, 1.5], labels=['Low', 'Medium', 'High'])
        risk_counts = risk_cuts.value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig2 = px.pie(risk_counts, names='Risk Level', values='Count', hole=0.6, 
                      color_discrete_sequence=['#FFB300', '#1F7A8C', '#0B3C5D'])
        graph2 = make_responsive(fig2)
    else:
        graph2 = None
        
    # Graph 3: Stress Level by State (Area/Line)
    if not df_copy.empty and 'stress_level' in df_copy:
        # Convert to numeric if it's not
        df_copy['stress_level_num'] = pd.to_numeric(df_copy['stress_level'], errors='coerce')
        stress_agg = df_copy.groupby('state')['stress_level_num'].mean().reset_index()
        fig3 = px.area(stress_agg, x='state', y='stress_level_num', 
                    color_discrete_sequence=['#FFB300'])
        graph3 = make_responsive(fig3)
    else:
        graph3 = None
        
    # Graph 4: Safe vs Critical vs Overexploited
    if not df_copy.empty and 'category' in df_copy:
        cat_counts = df_copy['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig4 = px.bar(cat_counts, x='Category', y='Count', 
                    color_discrete_sequence=['#0B3C5D'])
        graph4 = make_responsive(fig4)
    else:
        graph4 = None
    
    return render_template('dashboard_home.html', kpis=kpis, graph1=graph1, graph2=graph2, graph3=graph3, graph4=graph4)

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for('home'))

# ================= NEW ROUTE: DISTRICT PROFILER =================
@app.route('/district-profiler')
def district_profiler():
    if login_required():
        return redirect(url_for('auth', mode='login'))
        
    df_copy = apply_filters(df)
    
    # Needs a specific district selected, otherwise take the first one or overall
    selected_district = request.args.get('district')
    
    # Pass districts list to template for dropdown
    districts = sorted(df_copy['district'].dropna().unique().tolist()) if 'district' in df_copy else []
    
    if not selected_district and districts:
        selected_district = districts[0]
        
    if selected_district:
        district_data = df_copy[df_copy['district'] == selected_district]
        state_data = df_copy[df_copy['state'] == district_data['state'].iloc[0]] if not district_data.empty else pd.DataFrame()
    else:
        district_data = pd.DataFrame()
        state_data = pd.DataFrame()
        
    kpis = []
    graph1 = graph2 = graph3 = None
    
    if not district_data.empty:
        d_row = district_data.iloc[0]
        s_avg_recharge = state_data['annual_recharge'].mean() if 'annual_recharge' in state_data else 0
        s_avg_extraction = state_data['annual_extraction'].mean() if 'annual_extraction' in state_data else 0
        s_avg_stress = pd.to_numeric(state_data['stress_level'], errors='coerce').mean() if 'stress_level' in state_data else 0
        s_avg_risk = state_data['risk_score'].mean() if 'risk_score' in state_data else 0
        
        # Calculate water balance
        water_bal = d_row.get('annual_recharge', 0) - d_row.get('annual_extraction', 0)
        
        kpis = [
            {"title": "Selected District", "value": f"{selected_district}"},
            {"title": "Water Balance", "value": f"{water_bal:,.2f}"},
            {"title": "Risk Score", "value": f"{d_row.get('risk_score', 0):.2f}"},
            {"title": "Category", "value": f"{d_row.get('category', 'Unknown')}"}
        ]
        
        # ================= GRAPH 1: RADAR CHART =================
        categories = ['Recharge', 'Extraction', 'Stress Level', 'Risk Score']
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatterpolar(
            r=[
                d_row.get('annual_recharge', 0) / df_copy['annual_recharge'].max() if df_copy['annual_recharge'].max() > 0 else 0,
                d_row.get('annual_extraction', 0) / df_copy['annual_extraction'].max() if df_copy['annual_extraction'].max() > 0 else 0,
                pd.to_numeric(d_row.get('stress_level', 0), errors='coerce'),
                d_row.get('risk_score', 0)
            ],
            theta=categories,
            fill='toself',
            name=f'District: {selected_district}',
            line_color='#FFB300'
        ))
        
        fig1.add_trace(go.Scatterpolar(
            r=[
                s_avg_recharge / df_copy['annual_recharge'].max() if df_copy['annual_recharge'].max() > 0 else 0,
                s_avg_extraction / df_copy['annual_extraction'].max() if df_copy['annual_extraction'].max() > 0 else 0,
                s_avg_stress,
                s_avg_risk
            ],
            theta=categories,
            fill='toself',
            name='State Average',
            line_color='#0B3C5D'
        ))
        
        fig1.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title=f"{selected_district} vs State Average Profile",
            template='plotly_white'
        )
        graph1 = make_responsive(fig1)
        
        # ================= GRAPH 2: EXTRACTION GAUGE =================
        extr_ratio = d_row.get('extraction_ratio', 0)
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = extr_ratio,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Extraction Ratio"},
            gauge = {
                'axis': {'range': [None, 2]},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, 0.7], 'color': "#2CA58D"},
                    {'range': [0.7, 1.0], 'color': "#FFB300"},
                    {'range': [1.0, 2.0], 'color': "#e74c3c"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': extr_ratio
                }
            }
        ))
        graph2 = make_responsive(fig2)
        
        # ================= GRAPH 3: WATER BALANCE REPORT CARD =================
        fig3 = go.Figure(data=[
            go.Bar(name='Recharge', x=['Water'], y=[d_row.get('annual_recharge', 0)], marker_color='#2CA58D'),
            go.Bar(name='Extraction', x=['Water'], y=[d_row.get('annual_extraction', 0)], marker_color='#e74c3c')
        ])
        fig3.update_layout(barmode='group', title='Water Balance (MCM)')
        graph3 = make_responsive(fig3)
        
    return render_template('district_profiler.html', 
                        kpis=kpis, 
                        districts=districts,
                        selected_district=selected_district,
                        graph1=graph1, graph2=graph2, graph3=graph3)

# ================= NEW ROUTE: AI CORRELATION =================
@app.route('/ai-correlation')
def ai_correlation():
    if login_required():
        return redirect(url_for('auth', mode='login'))
        
    df_copy = apply_filters(df)
    
    kpis = [
        {"title": "Total Features", "value": len(df_copy.select_dtypes(include=['number']).columns)},
        {"title": "Records Evaluated", "value": len(df_copy)},
        {"title": "Max Correlation", "value": "Extraction ↔ Risk"},
        {"title": "AI Confidence", "value": "92%"}
    ]
    
    # ================= GRAPH 1: CORRELATION HEATMAP =================
    num_df = df_copy.select_dtypes(include=['number'])
    if not num_df.empty:
        corr_matrix = num_df.corr().round(2)
        fig1 = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='AI Feature Correlation Heatmap'
        )
        graph1 = make_responsive(fig1)
    else:
        graph1 = None
        
    # ================= GRAPH 2: 3D SCATTER PLOT =================
    if all(col in df_copy.columns for col in ['annual_recharge', 'annual_extraction', 'risk_score', 'category']):
        fig2 = px.scatter_3d(
            df_copy, 
            x='annual_recharge', 
            y='annual_extraction', 
            z='risk_score',
            color='category',
            size='utilization_rate' if 'utilization_rate' in df_copy else None,
            hover_name='district',
            color_discrete_map={'Safe': '#2CA58D', 'Critical': '#FFB300', 'Over-Exploited': '#e74c3c'},
            title='3D Feature Projection (Recharge vs Extraction vs Risk)'
        )
        # Tweak for 3d scatter margin
        fig2.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        graph2 = pio.to_html(fig2, full_html=False, config={'responsive': True})
    else:
        graph2 = None
        
    # ================= GRAPH 3: SCATTER WITH MARGINAL HISTOGRAMS =================
    if all(col in df_copy.columns for col in ['annual_recharge', 'annual_extraction', 'category']):
        fig3 = px.scatter(
            df_copy, 
            x="annual_recharge", 
            y="annual_extraction", 
            color="category",
            marginal_x="histogram", 
            marginal_y="histogram",
            color_discrete_map={'Safe': '#2CA58D', 'Critical': '#FFB300', 'Over-Exploited': '#e74c3c'},
            title="Recharge vs Extraction Density Distribution"
        )
        graph3 = make_responsive(fig3)
    else:
        graph3 = None

    return render_template('ai_correlation.html', kpis=kpis, graph1=graph1, graph2=graph2, graph3=graph3)

# ================= NEW ROUTE: SCENARIO SIMULATOR =================
@app.route('/scenario-simulator', methods=['GET', 'POST'])
def scenario_simulator():
    if login_required():
        return redirect(url_for('auth', mode='login'))
        
    df_copy = apply_filters(df)
    
    # Defaults
    rainfall_change = 0
    extraction_change = 0
    
    if request.method == 'POST':
        rainfall_change = float(request.form.get('rainfall_change', 0))
        extraction_change = float(request.form.get('extraction_change', 0))
        
    # Apply simulation
    sim_df = df_copy.copy()
    if 'annual_recharge' in sim_df and 'annual_extraction' in sim_df:
        sim_df['sim_recharge'] = sim_df['annual_recharge'] * (1 + rainfall_change/100)
        sim_df['sim_extraction'] = sim_df['annual_extraction'] * (1 + extraction_change/100)
        
        sim_df['sim_ratio'] = sim_df['sim_extraction'] / sim_df['sim_recharge'].replace(0, 0.0001)
        
        def sim_category(ratio):
            if ratio <= 0.7: return 'Safe'
            elif ratio <= 0.9: return 'Semi-Critical'
            elif ratio <= 1.0: return 'Critical'
            else: return 'Over-Exploited'
            
        sim_df['sim_category'] = sim_df['sim_ratio'].apply(sim_category)
    
    # KPIS
    orig_safe = len(df_copy[df_copy['category'] == 'Safe']) if 'category' in df_copy else 0
    sim_safe = len(sim_df[sim_df['sim_category'] == 'Safe']) if 'sim_category' in sim_df else 0
    orig_over = len(df_copy[df_copy['category'] == 'Over-Exploited']) if 'category' in df_copy else 0
    sim_over = len(sim_df[sim_df['sim_category'] == 'Over-Exploited']) if 'sim_category' in sim_df else 0
    
    kpis = [
        {"title": "Original Safe Districts", "value": orig_safe},
        {"title": "Simulated Safe Districts", "value": sim_safe},
        {"title": "Original Over-Exploited", "value": orig_over},
        {"title": "Simulated Over-Exploited", "value": sim_over}
    ]
    
    # ================= GRAPH 1: ORIGINAL VS SIMULATED PIE =================
    graph1 = graph2 = None
    if 'category' in df_copy and 'sim_category' in sim_df:
        orig_counts = df_copy['category'].value_counts().reset_index()
        orig_counts.columns = ['Category', 'Count']
        orig_counts['Type'] = 'Original'
        
        sim_counts = sim_df['sim_category'].value_counts().reset_index()
        sim_counts.columns = ['Category', 'Count']
        sim_counts['Type'] = 'Simulated'
        
        comb = pd.concat([orig_counts, sim_counts])
        
        fig1 = px.bar(
            comb, 
            x='Category', 
            y='Count', 
            color='Type', 
            barmode='group',
            color_discrete_map={'Original': '#0B3C5D', 'Simulated': '#FFB300'},
            title='Category Distribution Change'
        )
        graph1 = make_responsive(fig1)
        
        # ================= GRAPH 2: SUNBURST =================
        sim_tree = sim_df.groupby(['state', 'sim_category']).size().reset_index(name='count')
        fig2 = px.sunburst(
            sim_tree,
            path=['state', 'sim_category'],
            values='count',
            color='sim_category',
            color_discrete_map={'Safe': '#2CA58D', 'Critical': '#FFB300', 'Over-Exploited': '#e74c3c'},
            title='Simulated Risk Flow by State'
        )
        graph2 = make_responsive(fig2)
        
    return render_template(
        'scenario_simulator.html', 
        kpis=kpis, 
        rainfall_change=rainfall_change, 
        extraction_change=extraction_change,
        graph1=graph1, 
        graph2=graph2
    )

# ✅ NEW RAW DATA TABLE ROUTE
@app.route('/data-table')
def data_table():
    if login_required():
        return redirect(url_for('auth', mode='login'))
        
    df_copy = apply_filters(df)
    
    columns_to_show = ['state', 'district', 'annual_recharge', 'annual_extraction', 'water_balance', 'category', 'risk_score']
    cols = [c for c in columns_to_show if c in df_copy.columns]
    
    # Convert to dict for rendering. Limit to 1000 rows to prevent overwhelming the browser
    table_data = df_copy[cols].head(1000).to_dict(orient='records')
    
    return render_template('data_table.html', data=table_data, columns=cols)

# =========================
# 🤖 AI CHAT SUPPORT (WORKING)
# =========================
@app.route('/ai-chat', methods=['POST'])
def ai_chat():
    user_msg = request.form.get('message', '').lower()

    response = "Sorry, I didn't understand that. Try asking about prediction, dataset, or groundwater."

    if "prediction" in user_msg:
        response = "AI predicts groundwater using rainfall and extraction data with ML logic."

    elif "dataset" in user_msg:
        response = "We use historical groundwater CSV dataset for training and visualization."

    elif "help" in user_msg:
        response = "You can ask about prediction, dataset, overview, or system usage."

    elif "hello" in user_msg or "hi" in user_msg:
        response = "Hello 👋 How can I help you today?"

    elif "groundwater" in user_msg:
        response = "Groundwater is analyzed based on extraction vs rainfall trends."

    return {"reply": response}


if __name__ == '__main__':
    app.run(debug=True)















































