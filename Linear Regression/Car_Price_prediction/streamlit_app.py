import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import altair as alt
import os

# ========== Page Configuration ==========
st.set_page_config(
    page_title="🚗 VIP Car Hub",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS for VIP Design ==========
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #0d0d0d 0%, #323232 100%);
        color: #f0e6d2;
        font-family: 'Roboto', sans-serif;
    }
    .sidebar .sidebar-content { 
        background-color: #1f1f1f; 
        border-right: 3px solid #FFD700; 
    }
    .stButton>button { 
        background-color: #FFD700; 
        color: #1f1f1d; 
        border: none; 
        padding: 0.8em 1.6em; 
        border-radius: 0.5em; 
        font-weight: bold; 
        text-transform: uppercase;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #e6c200; 
    }
    h1, h2, h3 { 
        color: #FFD700; 
        text-shadow: 1px 1px 2px #000000; 
    }
    .stMetric-label, .stMetric-value { 
        color: #f0e6d2; 
    }
    .stDataFrame table { 
        background-color: #2b2b2b; 
        color: #f0e6d2; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Load Model ==========
@st.cache_resource
def load_model():
    return joblib.load("best_random_forest_model.pkl")
model = load_model()

# ========== Load and Prepare Data ==========
def load_data():
    """
    هنا نستخدم ملف "Car_Price.csv" لأنه الملف الأصلي الذي يحتوي على النصوص كما هي،
    وليس الملف المُنظف الذي قد يكون به ترميز مُسبق.
    """
    df = pd.read_csv("Car_Price.csv")
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    return df

# النسخة الأصلية للعرض مع النصوص كما هي
original_df = load_data()

# إنشاء نسخة لتشفير البيانات لغرض التنبؤ (سيُستخدم النموذج)
df_encoded = original_df.copy()

# ========== Label Encoding ==========
label_encoders = {}
# لكل عمود نصي نقوم بإنشاء LabelEncoder لتشفير البيانات (لنستخدمها فقط للتنبؤ)
for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# ========== Identify target column ==========
target_col = df_encoded.columns[-1]

# دالة للمساعدة على فك التشفير عند الحاجة (وليس ضروري استخدامها هنا عند العرض)
def decode_value(col_name, encoded_value):
    encoder = label_encoders[col_name]
    return encoder.inverse_transform([encoded_value])[0]

# ========== Sidebar ==========
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)
else:
    st.sidebar.title("🚗 VIP Car Hub")

menu = ["Home", "Overview", "Predict Price", "Batch Predict", "Model Eval"]
choice = st.sidebar.radio("Navigation", menu)

# ========== Home ==========
if choice == "Home":
    st.title("Welcome to VIP Car Hub 🚗✨")
    if os.path.exists("welcome_banner.png"):
        st.image("welcome_banner.png", use_container_width=True)
    else:
        st.markdown("***Experience the ultimate car buying journey.***")
    st.markdown("""
        - Discover market insights
        - Predict your dream car's price
        - Make informed decisions with confidence
    """)
    st.write("---")

# ========== Overview ==========
elif choice == "Overview":
    st.header("🔍 Dataset Insights")
    # عرض البيانات الأصلية مع النصوص كما هي
    st.dataframe(original_df.head())
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", original_df.shape[0])
    col2.metric("Features", original_df.shape[1] - 1)
    price_min = original_df[target_col].min()
    price_max = original_df[target_col].max()
    col3.metric(f"{target_col} Range", f"{price_min:,.0f} - {price_max:,.0f}")

    st.subheader(f"📈 {target_col} Distribution")
    chart1 = alt.Chart(original_df).mark_area(opacity=0.7, color="#FFD700").encode(
        alt.X(f"{target_col}:Q", bin=alt.Bin(maxbins=40)),
        alt.Y('count()', title='Count')
    ).properties(height=250)
    st.altair_chart(chart1, use_container_width=True)

    # استخراج قائمة السنوات المتوفرة لعرضها في المخطط
    unique_years = original_df["year"].dropna().unique().tolist()
    unique_years = sorted(unique_years)

    st.subheader(f"🕒 {target_col} vs year")
    chart2 = alt.Chart(original_df).mark_circle(size=60, color="#f0e6d2").encode(
        alt.X("year:N", title='Year', scale=alt.Scale(domain=unique_years)),
        alt.Y(f"{target_col}:Q", title=target_col),
        tooltip=['year', target_col]
    ).interactive().properties(height=300)
    st.altair_chart(chart2, use_container_width=True)

    st.subheader("📌 Correlation Heatmap")
    corr = df_encoded.corr()
    corr_df = corr.reset_index().melt('index')
    heatmap = alt.Chart(corr_df).mark_rect().encode(
        alt.X('index:O', title=None),
        alt.Y('variable:O', title=None),
        alt.Color('value:Q', title='Correlation')
    ).properties(height=400)
    st.altair_chart(heatmap, use_container_width=True)

# ========== Predict Price ==========
elif choice == "Predict Price":
    with st.container():
        st.markdown("""
            <div style='padding: 1rem; background: #1e1e1e; border: 2px solid #FFD700; border-radius: 12px;'>
                <h2 style='color: #FFD700; margin-bottom: 0.5rem;'>💰 VIP Selling Price Estimation</h2>
                <p style='color: #f0e6d2;'>Customize your car details below and get an instant prediction:</p>
            </div>
        """, unsafe_allow_html=True)

    left, right = st.columns(2)
    input_data = {}

    for i, col in enumerate(original_df.drop(columns=[target_col]).columns):
        container = left if i % 2 == 0 else right
        with container:
            box_style = """
                <style>
                div[data-baseweb="select"] > div { background-color: #2c2c2c !important; color: #f0e6d2 !important; }
                </style>
            """
            st.markdown(box_style, unsafe_allow_html=True)

            display_col = col.replace("_", " ").title()

            if col.lower() == "seats":
                unique_seats = sorted(original_df[col].dropna().astype(int).unique())
                selected = st.selectbox("🪑 Number of Seats", unique_seats)
                input_data[col] = selected

            elif col.lower() == "year":
                unique_years = sorted(original_df[col].dropna().astype(int).unique(), reverse=True)
                selected = st.selectbox("📅 Model Year", unique_years)
                input_data[col] = selected

            elif col.lower() == "max_power":
                power_col = pd.to_numeric(original_df[col], errors='coerce')
                min_val = float(power_col.min())
                max_val = float(power_col.max())
                mean_val = float(power_col.mean())
                input_data[col] = st.slider("⚡ Max Power", min_val, max_val, mean_val)

            elif original_df[col].dtype == 'object':
                values = original_df[col].dropna().astype(str).unique()
                selected = st.selectbox(f"🧾 {display_col}", sorted(values))
                input_data[col] = label_encoders[col].transform([selected])[0]

            else:
                numeric_col = pd.to_numeric(original_df[col], errors='coerce')
                min_val = float(numeric_col.min())
                max_val = float(numeric_col.max())
                mean_val = float(numeric_col.mean())
                input_data[col] = st.slider(f"📊 {display_col}", min_val, max_val, mean_val)

    st.markdown("""<hr style='margin: 2rem 0; border-color: #FFD700;'>""", unsafe_allow_html=True)

    center = st.container()
    with center:
        if st.button("Estimate Price 🚀"):
            input_df = pd.DataFrame([input_data])

            # إعادة تسمية الأعمدة لتطابق أسماء الأعمدة المستخدمة أثناء التدريب
            rename_map = {
                "engine": "Engine",
                "mileage(km/ltr/kg)": "Mileage",
                "max_power": "Power"
            }
            input_df.rename(columns=rename_map, inplace=True)

            price = model.predict(input_df)[0]
            st.success(f"🔖 VIP Estimated {target_col}: **{price:,.2f}**")
            st.balloons()


# ========== Batch Predict ==========
elif choice == "Batch Predict":
    st.header("📁 Bulk Predictions")
    st.markdown("Upload your CSV (with same columns) to receive VIP estimates in bulk.")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        # قراءة الملف المُحمّل وتخزين نسخة من البيانات الأصلية
        data_original = pd.read_csv(file)
        
        # إنشاء نسخة لتطبيق عملية الترميز عليها فقط لغرض التنبؤ
        data_encoded = data_original.copy()
        for col, le in label_encoders.items():
            if col in data_encoded.columns:
                try:
                    data_encoded[col] = le.transform(data_encoded[col])
                except Exception as e:
                    st.error(f"Error in column {col}: {e}")
        
        # إجراء التنبؤ باستخدام النسخة المُشفّرة
        data_encoded[f'Predicted_{target_col}'] = model.predict(data_encoded)
        
        # استرجاع النصوص الأصلية إذا ظهرت كأرقام
        for col, le in label_encoders.items():
            if col in data_original.columns and pd.api.types.is_numeric_dtype(data_original[col]):
                try:
                    data_original[col] = le.inverse_transform(data_encoded[col])
                except Exception as e:
                    st.error(f"Error in reverting column {col}: {e}")
        
        # دمج عمود التنبؤ مع النسخة الأصلية ليظهر الجدول النهائي بالنصوص الأصلية
        data_original[f'Predicted_{target_col}'] = data_encoded[f'Predicted_{target_col}']
        
        st.dataframe(data_original)
        csv = data_original.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, f"vip_predictions_{target_col}.csv")

# ========== Model Eval ==========
elif choice == "Model Eval":
    st.header("📋 Model Performance")
    eval_df = pd.read_csv("model_evaluation_results.csv")
    st.dataframe(eval_df)
    st.subheader("📌 Feature Importance")
    st.image("feature_importance.png", use_container_width=True)
    st.subheader("📊 Actual vs Predicted")
    st.image("prediction_actual_vs_pred.png", use_container_width=True)
    st.subheader("📉 Error Distribution")
    st.image("prediction_error_distribution.png", use_container_width=True)
