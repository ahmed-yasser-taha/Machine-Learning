import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import altair as alt
from PIL import Image
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Page Configuration ==========
st.set_page_config(
    page_title="🚗 VIP Car Hub",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS for Professional Design ==========
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;700;900&family=Playfair+Display:wght@700&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #121212 0%, #1e1e1e 100%);
        color: #ffffff;
        font-family: 'Montserrat', sans-serif;
    }
    .sidebar .sidebar-content { 
        background-color: #121212; 
        border-right: 1px solid #444; 
    }
    .stButton>button { 
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
        color: #121212; 
        border: none; 
        padding: 0.8em 2em; 
        border-radius: 8px; 
        font-weight: bold; 
        font-size: 1em;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    h1, h2, h3, h4 { 
        color: #FFD700; 
        font-family: 'Playfair Display', serif;
        margin-bottom: 0.5em;
    }
    .stMetric-label { 
        color: #AAAAAA; 
        font-size: 0.9em;
    }
    .stMetric-value { 
        color: #FFD700; 
        font-size: 1.5em;
    }
    .stDataFrame { 
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame table { 
        background-color: #1e1e1e;
    }
    .stDataFrame th { 
        background-color: #2a2a2a !important;
        color: #FFD700 !important;
    }
    .stDataFrame tr:hover { 
        background-color: #2a2a2a;
    }
    .css-1aumxhk { /* Main container */
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .css-1v3fvcr { /* Sidebar items */
        color: #ffffff;
    }
    .css-1v3fvcr:hover { 
        color: #FFD700;
    }
    .st-b7 { /* Text input */
        background-color: #2a2a2a;
    }
    .stSelectbox div div { /* Select box */
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    .stSlider div div div div { /* Slider */
        background: #FFD700 !important;
    }
    .stAlert { /* Alert boxes */
        border-radius: 8px;
    }
    .stMarkdown { 
        line-height: 1.6;
    }
    .css-1q8dd3e { /* Number input */
        background-color: #2a2a2a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Helper Functions ==========
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_string.decode()});
                background-size: cover;
                background-attachment: fixed;
                background-color: rgba(0, 0, 0, 0.8);
                background-blend-mode: overlay;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image not found. Using default background.")

def add_logo(logo_path, width=200):
    try:
        logo = Image.open(logo_path)
        st.sidebar.image(logo, width=width)
    except FileNotFoundError:
        st.sidebar.title("🚗 VIP Car Hub")

# ========== Load Assets ==========
add_bg_from_local("background.jpg")
add_logo("logo.png")

# ========== Load Model ==========
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_random_forest_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_random_forest_model.pkl' is in the directory.")
        return None
model = load_model()
if model is None:
    st.stop()

# ========== Load and Prepare Data ==========
def load_data():
    try:
        df = pd.read_csv("Car_Price.csv")
        # Clean max_power
        if 'max_power' in df.columns:
            df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
        # Ensure other numeric columns
        for col in ['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'seats', 'selling_price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Dataset file 'Car_Price.csv' not found.")
        return None

original_df = load_data()
if original_df is None:
    st.stop()

df_encoded = original_df.copy()

# ========== Label Encoding ==========
label_encoders = {}
for col in ['name', 'fuel', 'seller_type', 'transmission', 'owner']:
    if col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# ========== Identify target column ==========
target_col = 'selling_price'

# ========== Sidebar ==========
menu = ["🏠 Home", "📊 Overview", "💰 Predict Price", "📁 Batch Predict", "📈 Model Eval"]
choice = st.sidebar.radio("Navigation", menu)

# ========== Home Page ==========
if choice == "🏠 Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.title("VIP Car Hub")
        st.markdown("""
            <div style='font-size: 1.2em; margin-bottom: 2em;'>
                Your premier destination for luxury car valuation and market insights.
                Experience the future of car pricing with our AI-powered platform.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: rgba(30, 30, 30, 0.7); padding: 1.5rem; border-radius: 8px; border-left: 4px solid #FFD700;'>
                <h4 style='color: #FFD700; margin-top: 0;'>Why Choose VIP Car Hub?</h4>
                <ul style='padding-left: 1.2em;'>
                    <li>🔍 Accurate price predictions</li>
                    <li>📈 Real-time market trends</li>
                    <li>🚀 Instant valuation reports</li>
                    <li>💎 Premium user experience</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            st.image("luxury_car.jpg", use_column_width=True, caption="Your Dream Car Awaits")
        except:
            st.image("https://images.unsplash.com/photo-1494905998402-395d579af36f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
                    use_column_width=True, caption="Your Dream Car Awaits")

    st.markdown("---")
    
    features = st.columns(3)
    with features[0]:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: rgba(30, 30, 30, 0.7); border-radius: 8px;'>
                <h4 style='color: #FFD700;'>🚀 Instant Estimates</h4>
                <p>Get accurate price predictions in seconds with our advanced AI model.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with features[1]:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: rgba(30, 30, 30, 0.7); border-radius: 8px;'>
                <h4 style='color: #FFD700;'>📊 Market Insights</h4>
                <p>Discover trends and patterns in the luxury car market.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with features[2]:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: rgba(30, 30, 30, 0.7); border-radius: 8px;'>
                <h4 style='color: #FFD700;'>💎 Premium Experience</h4>
                <p>Designed for discerning users who demand the best.</p>
            </div>
        """, unsafe_allow_html=True)

# ========== Overview Page ==========
# ========== Overview Page ==========
elif choice == "📊 Overview":
    st.header("📊 Dataset Insights")
    
    with st.expander("🔍 View Raw Data", expanded=False):
        st.dataframe(original_df.head(10).style
                    .background_gradient(cmap='Oranges')
                    .set_properties(**{'color': 'white', 'background-color': '#1e1e1e'}))
    
    st.markdown("---")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{original_df.shape[0]:,}", help="Total cars in our database")
    
    with col2:
        st.metric("Features Available", original_df.shape[1] - 1, help="Different attributes we analyze")
    
    with col3:
        price_min = original_df[target_col].min()
        price_max = original_df[target_col].max()
        st.metric(f"{target_col} Range", f"${price_min:,.0f} - ${price_max:,.0f}")
    
    st.markdown("---")
    
    # Price Distribution
    st.subheader(f"📈 {target_col} Distribution")
    chart1 = alt.Chart(original_df).transform_density(
        target_col,
        as_=[target_col, 'density'],
    ).mark_area(opacity=0.7, interpolate='monotone', line=True, color='#FFD700').encode(
        alt.X(f'{target_col}:Q', title='Price', axis=alt.Axis(format='$,.0f')),
        alt.Y('density:Q', title='Density'),
        tooltip=[alt.Tooltip(f'{target_col}:Q', format='$,.0f')]
    ).properties(height=400)
    st.altair_chart(chart1, use_container_width=True)
    
    st.markdown("---")
    
    # Price vs Year
    st.subheader(f"🕒 {target_col} by Model Year")
    chart2 = alt.Chart(original_df).mark_circle(size=60, color='#FFD700', opacity=0.7).encode(
        x=alt.X('year:O', title='Model Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{target_col}:Q', title='Price', axis=alt.Axis(format='$,.0f')),
        tooltip=['year', alt.Tooltip(target_col, format='$,.0f')]
    ).properties(height=400)
    st.altair_chart(chart2, use_container_width=True)
# ========== Predict Price Page ==========
elif choice == "💰 Predict Price":
    st.header("💰 VIP Price Prediction")
    
    with st.container():
        st.markdown("""
            <div style='padding: 2rem; background: rgba(30, 30, 30, 0.7); border-radius: 12px; border-left: 4px solid #FFD700;'>
                <h2 style='color: #FFD700; margin-top: 0;'>Customize Your Car</h2>
                <p style='font-size: 1.1em;'>Enter the details below to get an instant valuation for your luxury vehicle.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Form columns
    left, right = st.columns(2)
    input_data = {}
    
    # Form fields
    with left:
        st.subheader("Basic Information")
        
        # Name
        name_options = original_df["name"].dropna().astype(str).unique()
        selected_name = st.selectbox("🚗 Car Model", sorted(name_options))
        input_data["name"] = label_encoders["name"].transform([selected_name])[0]
        
        # Year
        unique_years = sorted(original_df["year"].dropna().astype(int).unique(), reverse=True)
        selected_year = st.selectbox("📅 Model Year", unique_years, index=0)
        input_data["year"] = selected_year
        
        # Fuel
        fuel_types = original_df["fuel"].dropna().astype(str).unique()
        selected_fuel = st.selectbox("⛽ Fuel Type", sorted(fuel_types))
        input_data["fuel"] = label_encoders["fuel"].transform([selected_fuel])[0]
        
        # Transmission
        transmission_types = original_df["transmission"].dropna().astype(str).unique()
        selected_transmission = st.selectbox("⚙️ Transmission", sorted(transmission_types))
        input_data["transmission"] = label_encoders["transmission"].transform([selected_transmission])[0]
        
        # Seller Type
        seller_types = original_df["seller_type"].dropna().astype(str).unique()
        selected_seller_type = st.selectbox("👤 Seller Type", sorted(seller_types))
        input_data["seller_type"] = label_encoders["seller_type"].transform([selected_seller_type])[0]
        
        # Owner
        owner_types = original_df["owner"].dropna().astype(str).unique()
        selected_owner = st.selectbox("🔑 Owner", sorted(owner_types))
        input_data["owner"] = label_encoders["owner"].transform([selected_owner])[0]
    
    with right:
        st.subheader("Technical Specifications")
        
        # Mileage
        input_data["mileage(km/ltr/kg)"] = st.number_input("⛽ Mileage (km/ltr/kg)", min_value=0.0, value=15.0, step=0.1)
        
        # Engine
        input_data["engine"] = st.number_input("🔧 Engine Size (cc)", min_value=0.0, value=2000.0, step=100.0)
        
        # Max Power
        input_data["max_power"] = st.number_input("⚡ Max Power (bhp)", min_value=0.0, value=120.0, step=10.0)
        
        # Seats
        unique_seats = sorted(original_df["seats"].dropna().astype(int).unique())
        selected_seats = st.selectbox("🪑 Number of Seats", unique_seats)
        input_data["seats"] = selected_seats
        
        # Km Driven
        input_data["km_driven"] = st.number_input("🛣️ Km Driven", min_value=0.0, value=50000.0, step=1000.0)
    
    st.markdown("---")
    
    # Prediction button and results
    center = st.container()
    with center:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🚀 Get VIP Valuation", use_container_width=True):
                with st.spinner('Calculating your premium valuation...'):
                    input_df = pd.DataFrame([input_data])
                    
                    # Ensure column names match the model's training data
                    expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df_encoded.drop(columns=[target_col]).columns
                    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
                    
                    try:
                        price = model.predict(input_df)[0]
                        
                        st.success("Valuation completed successfully!")
                        st.markdown(f"""
                            <div style='text-align: center; padding: 1.5rem; background: rgba(30, 30, 30, 0.8); border-radius: 12px; border: 1px solid #FFD700;'>
                                <h3 style='color: #FFD700; margin-top: 0;'>VIP Valuation Result</h3>
                                <p style='font-size: 1.2em;'>Your customized valuation:</p>
                                <h2 style='color: #FFD700;'>₹{price:,.2f}</h2>
                                <p style='font-size: 0.9em; color: #AAAAAA;'>Based on current market trends and your specifications</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                        
                        try:
                            st.image("celebrate.jpg", use_column_width=True, caption="Congratulations on your valuation!")
                        except:
                            pass
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

# ========== Batch Predict Page ==========
elif choice == "📁 Batch Predict":
    st.header("📁 Bulk Valuation Processing")
    
    with st.container():
        st.markdown("""
            <div style='padding: 1.5rem; background: rgba(30, 30, 30, 0.7); border-radius: 12px; border-left: 4px solid #FFD700;'>
                <h3 style='color: #FFD700; margin-top: 0;'>Upload Your Inventory</h3>
                <p>Process multiple valuations at once by uploading your CSV file with vehicle specifications.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    file = st.file_uploader("📤 Upload CSV File", type="csv")
    
    if file:
        with st.spinner('Processing your luxury vehicle inventory...'):
            # Read uploaded file
            data_original = pd.read_csv(file)
            
            # Clean max_power
            if 'max_power' in data_original.columns:
                data_original['max_power'] = pd.to_numeric(data_original['max_power'], errors='coerce')
            
            # Create encoded copy for prediction
            data_encoded = data_original.copy()
            for col, le in label_encoders.items():
                if col in data_encoded.columns:
                    try:
                        data_encoded[col] = le.transform(data_encoded[col].astype(str))
                    except Exception as e:
                        st.error(f"Error encoding column {col}: {str(e)}")
                        st.stop()
            
            # Make predictions
            try:
                expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df_encoded.drop(columns=[target_col]).columns
                data_encoded = data_encoded.reindex(columns=expected_columns, fill_value=0)
                data_encoded[f'Predicted_{target_col}'] = model.predict(data_encoded)
                
                # Merge predictions with original data
                data_original[f'Predicted_{target_col}'] = data_encoded[f'Predicted_{target_col}']
                
                # Format the predicted price
                data_original[f'Predicted_{target_col}'] = data_original[f'Predicted_{target_col}'].apply(lambda x: f"₹{x:,.2f}")
                
                # Show results
                st.success("✅ Valuation complete! Preview of your results:")
                st.dataframe(data_original.style.format({
                    f'Predicted_{target_col}': lambda x: x.replace('₹', '').replace(',', '')
                }).applymap(lambda x: 'color: #FFD700' if isinstance(x, str) and x.startswith('₹') else ''))
                
                # Download button
                csv = data_original.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Report",
                    data=csv,
                    file_name=f"vip_car_valuations_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv',
                    help="Download your complete valuation report"
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {str(e)}")

# ========== Model Eval Page ==========
elif choice == "📈 Model Eval":
    st.header("📈 Model Performance Evaluation")
    
    with st.container():
        st.markdown("""
            <div style='padding: 1.5rem; background: rgba(30, 30, 30, 0.7); border-radius: 12px; border-left: 4px solid #FFD700;'>
                <h3 style='color: #FFD700; margin-top: 0;'>Model Insights</h3>
                <p>Explore the performance metrics and feature importance of our AI-powered valuation model.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load evaluation results
    try:
        results_df = pd.read_csv("model_evaluation_results.csv", index_col=0)
        st.subheader("Model Performance Metrics")
        st.dataframe(results_df.style
                    .background_gradient(cmap='Oranges')
                    .set_properties(**{'color': 'white', 'background-color': '#1e1e1e'}))
    except FileNotFoundError:
        st.warning("Evaluation results file 'model_evaluation_results.csv' not found.")
    
    st.markdown("---")
    
    # Feature Importance Visualization
    st.subheader("Feature Importance")
    try:
        importance_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='Oranges')
        ax.set_title("Feature Importance (Random Forest)", color='#FFD700')
        ax.set_xlabel("Importance", color='#AAAAAA')
        ax.set_ylabel("Feature", color='#AAAAAA')
        ax.tick_params(colors='#AAAAAA')
        ax.set_facecolor('#1e1e1e')
        fig.set_facecolor('#1e1e1e')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")
    
    st.markdown("---")
    
    # Display saved visualizations
    st.subheader("Exploratory Data Analysis Visualizations")
    for img in ["eda_price_distribution.png", "eda_price_vs_year.png", "eda_correlation_heatmap.png"]:
        try:
            st.image(img, caption=img.replace('.png', '').replace('eda_', '').replace('_', ' ').title(), use_column_width=True)
        except FileNotFoundError:
            st.warning(f"Image {img} not found.")