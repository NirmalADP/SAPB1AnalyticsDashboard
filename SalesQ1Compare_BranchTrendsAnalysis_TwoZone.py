# Integrating High Frequency Buyers Module into Customer Analysis Tab

# main.py - Enhanced Dashboard with HFB Integration
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import timedelta, datetime
import os
import numpy as np
from sqlalchemy import create_engine
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env file

# For Streamlit secrets.toml
try:
    DB_HOST = st.secrets["connections"]["DB_HOST"]
    DB_PORT = st.secrets["connections"]["DB_PORT"]
    DB_USER = st.secrets["connections"]["DB_USER"]
    DB_PASSWORD = st.secrets["connections"]["DB_PASSWORD"]
except (KeyError, FileNotFoundError):
    # Fallback to environment variables or defaults
    DB_HOST = os.getenv('DB_HOST', '192.168.238.7')
    DB_PORT = int(os.getenv('DB_PORT', '30015'))
    DB_USER = os.getenv('DB_USER', 'SYSTEM')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'Welcome@1')
# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="SAP B1 Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: grey;
    color: black;
    padding-top: 20px;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: white; /* yellow-400 */
    font-weight: bold;
}

/* Navigation links */
[data-testid="stSidebarNav"] a {
    font-size: 16px;
    padding: 8px 16px;
    border-radius: 8px;
    color: white; /* gray-300 */
    text-decoration: none;
    transition: all 0.3s ease-in-out;
}

/* Hover effect */
[data-testid="stSidebarNav"] a:hover {
    background-color: #374151; /* gray-700 */
    color: white;
}

/* Active page */
[data-testid="stSidebarNav"] a.active {
    background: #facc15; /* yellow highlight */
    color: white;
    font-weight: bold;
}

.main-header {
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    padding: 0.5rem;
    border-radius: 8px;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}
.main-header h1 {
    margin: 0;
    font-size: 1.5rem;
}
.connection-status {
    padding: 8px;
    border-radius: 5px;
    margin: 8px 0;
    font-size: 0.9rem;
}
.status-success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.status-error {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}
.filter-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.inventory-card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    border-left: 4px solid #2c3e50;
}
.action-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border: 1px solid #dee2e6;
}
.recommendation-box {
    background-color: #e8f5e8;
    border: 1px solid #28a745;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.critical-box {
    background-color: #f8d7da;
    border: 1px solid #dc3545;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.hfb-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Utility Functions
# ----------------------------
def in_lakhs(val):
    """Format value in lakhs"""
    if val == 0:
        return "₹0.00 L"
    return f"₹{val/100000:,.2f} L"

def shift_year(date_val, years=1):
    """Shift date by years"""
    return date_val - timedelta(days=365 * years)

def get_period_comparison(df, period_type="MOM"):
    """Calculate period over period comparison"""
    if df.empty:
        return 0, "No data"

    df_sorted = df.sort_values('InvDate')

    if period_type == "MOM":
        cutoff_date = df_sorted['InvDate'].max() - timedelta(days=30)
    elif period_type == "QOQ":
        cutoff_date = df_sorted['InvDate'].max() - timedelta(days=90)
    else:  # YOY
        cutoff_date = df_sorted['InvDate'].max() - timedelta(days=365)

    recent_sales = df[df['InvDate'] > cutoff_date]['SalesValue'].sum()
    older_sales = df[df['InvDate'] <= cutoff_date]['SalesValue'].sum()

    if older_sales > 0:
        growth = ((recent_sales - older_sales) / older_sales) * 100
        return growth, f"{growth:+.1f}%"
    return 0, "n/a"

def generate_stock_recommendations(row):
    """Generate stock recommendations based on stock levels and coverage"""
    recommendations = []

    # Stock level recommendations
    if row.get('Stock_Status') == 'Understock':
        if row.get('Stock_Coverage_Days', 0) < 7:
            recommendations.append("🚨 URGENT: Raise PO immediately - less than 1 week coverage")
        elif row.get('Stock_Coverage_Days', 0) < 15:
            recommendations.append("⚠️ Raise PO - less than 2 weeks coverage")
        else:
            recommendations.append("📝 Consider raising PO")
    elif row.get('Stock_Status') == 'Overstock':
        recommendations.append("📦 Consider transfer to high-demand branch")

    # Age-based recommendations
    if row.get('Oldest_Item_Age_Days', 0) > 90:
        recommendations.append("🕐 Aged stock - consider clearance sale")
    elif row.get('Oldest_Item_Age_Days', 0) > 60:
        recommendations.append("⏰ Monitor for aging")

    return " | ".join(recommendations) if recommendations else "✅ No immediate action needed"

def calculate_stock_status(current_stock, avg_daily_sales, safety_days=7):
    """Calculate stock status based on coverage days"""
    if avg_daily_sales <= 0:
        return 'Unknown', 999

    coverage_days = current_stock / avg_daily_sales if avg_daily_sales > 0 else 999

    if coverage_days < safety_days:
        return 'Understock', coverage_days
    elif coverage_days > 60:  # More than 60 days coverage
        return 'Overstock', coverage_days
    else:
        return 'Optimal', coverage_days

# ----------------------------
# Database Configuration
# ----------------------------
class DatabaseConnection:
    def __init__(self):
        self.host = os.getenv('DB_HOST', '192.168.238.7')
        self.port = int(os.getenv('DB_PORT', '30015'))
        self.username = os.getenv('DB_USER', 'SYSTEM')
        self.password = os.getenv('DB_PASSWORD', 'Welcome@1')
        self.connection = None
        self.timeout = 30

    def get_connection(self):
        """Get a fresh connection for each operation"""
        try:
            import hdbcli.dbapi as dbapi
            connection = dbapi.connect(
                address=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                autocommit=True,
                timeout=self.timeout,
                encrypt=False,
                sslValidateCertificate=False,
                reconnect=True,
                communicationTimeout=30000
            )
            return connection
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None

    def test_connection(self):
        """Test database connection"""
        try:
            import socket
            # Test basic network connectivity first
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            
            if result != 0:
                return False, f"Network connection failed to {self.host}:{self.port}"

            conn = self.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM DUMMY")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                return True, f"Connected ({result[0]})"
            return False, "Failed to establish connection"
        except Exception as e:
            return False, f"Connection test failed: {e}"

    def execute_query(self, query, params=None):
        """Execute query and return DataFrame"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return pd.DataFrame()

            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=columns)
            return df

        except Exception as e:
            st.error(f"Query execution failed: {e}")
            return pd.DataFrame()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

@st.cache_resource
def get_db_instance():
    return DatabaseConnection()

# ----------------------------
# Data Loading
# ----------------------------
@st.cache_data(ttl=300)
def load_filter_values():
    """Load filter values for dropdowns"""
    db = get_db_instance()
    try:
        # Test connection first
        is_connected, msg = db.test_connection()
        if not is_connected:
            st.error(f"Cannot connect to database: {msg}")
            return {
                "min_date": datetime.now().date() - timedelta(days=30),
                "max_date": datetime.now().date(),
                "zones": [], "branches": [], "item_groups": [], "brands": []
            }

        # Date range
        date_query = """
        SELECT MIN("InvDate") AS MIN_DATE, MAX("InvDate") AS MAX_DATE
        FROM "AVA_Integration"."SALES_AI"
        """
        date_df = db.execute_query(date_query)
        if date_df.empty:
            min_date = datetime.now().date() - timedelta(days=365)
            max_date = datetime.now().date()
        else:
            min_date = pd.to_datetime(date_df.iloc[0, 0]).date()
            max_date = pd.to_datetime(date_df.iloc[0, 1]).date()

        def safe_query(query, col):
            try:
                r = db.execute_query(query)
                if not r.empty and col in r.columns:
                    return sorted([str(z).strip() for z in r[col].dropna().unique()
                                   if str(z).strip() not in ('', 'nan')])
                return []
            except:
                return []

        zones = safe_query('SELECT DISTINCT "StateLoc" FROM "AVA_Integration"."SALES_AI" WHERE "StateLoc" IS NOT NULL', "StateLoc")
        branches = safe_query('SELECT DISTINCT "U_AVA_Branch" FROM "AVA_Integration"."SALES_AI" WHERE "U_AVA_Branch" IS NOT NULL', "U_AVA_Branch")
        item_groups = safe_query('SELECT DISTINCT "Item Group Name" FROM "AVA_Integration"."SALES_AI" WHERE "Item Group Name" IS NOT NULL', "Item Group Name")
        brands = safe_query('SELECT DISTINCT "Brand Name" FROM "AVA_Integration"."SALES_AI" WHERE "Brand Name" IS NOT NULL', "Brand Name")

        return {"min_date": min_date, "max_date": max_date,
                "zones": zones, "branches": branches,
                "item_groups": item_groups, "brands": brands}
    except Exception as e:
        st.error(f"Error loading filter values: {e}")
        return {
            "min_date": datetime.now().date() - timedelta(days=30),
            "max_date": datetime.now().date(),
            "zones": [], "branches": [], "item_groups": [], "brands": []
        }

@st.cache_data(ttl=180)
def load_sales_data(start_date, end_date, zone, branch, item_group, brand):
    """Load sales data with filters"""
    db = get_db_instance()
    try:
        base_query = """
        SELECT "InvDate","InvNo","StateLoc","U_AVA_Branch",
               "Item Group Name","Brand Name","ItemName",
               "CustomerName","SlpName" AS "Salesperson",
               "Quantity","SalesValue","GrossSalesValue",
               "GPNew" AS "GP_AMT","GPNew%" AS "GP_PCT"
        FROM "AVA_Integration"."SALES_AI"
        WHERE "InvDate" BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        conditions = []
        if zone and zone != "All":
            conditions.append('"StateLoc" = ?'); params.append(zone)
        if branch and branch != "All":
            conditions.append('"U_AVA_Branch" = ?'); params.append(branch)
        if item_group and item_group != "All":
            conditions.append('"Item Group Name" = ?'); params.append(item_group)
        if brand and brand != "All":
            conditions.append('"Brand Name" = ?'); params.append(brand)
        query = base_query + (" AND " + " AND ".join(conditions) if conditions else "")

        df = db.execute_query(query, params)
        if df.empty:
            return df

        df["InvDate"] = pd.to_datetime(df["InvDate"], errors="coerce")

        for col in ["SalesValue", "GrossSalesValue", "GP_AMT", "GP_PCT", "Quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for col in ["StateLoc", "U_AVA_Branch", "Item Group Name", "Brand Name",
                    "ItemName", "CustomerName", "Salesperson", "InvNo"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")

        # Extra safety for Top Customers/Employees
        if "CustomerName" in df.columns:
            df["CustomerName"] = df["CustomerName"].replace({"": "Unknown"})
        if "Salesperson" in df.columns:
            df["Salesperson"] = df["Salesperson"].replace({"": "Unknown"})

        return df
    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_inventory_data():
    """Load inventory data - demo using sales patterns as proxy"""
    db = get_db_instance()
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=60)

        sales_query = """
        SELECT 
            "ItemName" as "ItemCode",
            "ItemName",
            "U_AVA_Branch" as "Branch",
            "StateLoc" as "Zone",
            "Item Group Name" as "ItemGroup",
            "Brand Name" as "Brand",
            SUM("Quantity") as "Sales_60_Days",
            COUNT(DISTINCT "InvDate") as "Sales_Days"
        FROM "AVA_Integration"."SALES_AI"
        WHERE "InvDate" >= ?
        GROUP BY "ItemName", "U_AVA_Branch", "StateLoc", "Item Group Name", "Brand Name"
        HAVING SUM("Quantity") > 0
        """
        df = db.execute_query(sales_query, [start_date])
        if df.empty:
            return df

        import numpy as np
        np.random.seed(42)

        df['Avg_Daily_Sales'] = df['Sales_60_Days'] / 60
        df['Current_Stock'] = np.random.randint(0, 100, size=len(df))
        df['Total_Stock'] = df['Current_Stock'] + np.random.randint(0, 50, size=len(df))
        df['Oldest_Item_Age_Days'] = np.random.randint(1, 180, size=len(df))

        results = []
        for _, row in df.iterrows():
            status, coverage = calculate_stock_status(row['Current_Stock'], row['Avg_Daily_Sales'])
            results.append({'Stock_Status': status, 'Stock_Coverage_Days': coverage})

        status_df = pd.DataFrame(results)
        df = pd.concat([df, status_df], axis=1)

        df['Age_Status'] = df['Oldest_Item_Age_Days'].apply(
            lambda x: 'Fresh' if x < 30 else 'Aging' if x < 90 else 'Aged Stock'
        )
        df['Recommendations'] = df.apply(generate_stock_recommendations, axis=1)
        return df

    except Exception as e:
        st.error(f"Error loading inventory data: {e}")
        return pd.DataFrame()

# ----------------------------
# HFB Data Loading Functions
# ----------------------------
@st.cache_data(ttl=3600)
def load_hfb_data_original(period_days=365):
    """Load High Frequency Buyers data"""
    try:
        # Create SQLAlchemy engine
        engine = create_engine("hana+hdbcli://SYSTEM:Welcome%401@192.168.238.7:30015")
        
        query = f"""
        WITH CustomerItems AS (
            SELECT 
                "CardCode",
                STRING_AGG("ItemGroup", ', ') AS "ItemGroups"
            FROM (
                SELECT DISTINCT
                    "CardCode",
                    "Item Group Name" AS "ItemGroup"
                FROM 
                    "AVA_Integration"."SALES_AI"
                WHERE 
                    "InvDate" >= ADD_DAYS(CURRENT_DATE, :days_param)
                )
            GROUP BY 
                "CardCode"
        ),
        CustomerStats AS (
            SELECT 
                S."CardCode" AS "CustomerID",
                S."CustomerName",
                S."Mobile No" AS "Mobile",
                W."ZONE" AS "Zone",
                COUNT(DISTINCT S."InvNo") AS "TotalOrders",
                SUM(S."Quantity") AS "TotalItems",
                ROUND(SUM(S."SalesValue"), 0) AS "TotalSpend",
                ROUND(DAYS_BETWEEN(MAX(S."InvDate"), CURRENT_DATE), 0) AS "DaysSinceLastPurchase",
                CASE 
                    WHEN COUNT(DISTINCT S."InvNo") > 1 
                    THEN ROUND(DAYS_BETWEEN(MIN(S."InvDate"), MAX(S."InvDate")) / NULLIF((COUNT(DISTINCT S."InvNo") - 1), 0), 0)
                    ELSE NULL 
                END AS "AvgDaysBetweenOrders",
                ROUND(SUM(S."Quantity") / NULLIF(COUNT(DISTINCT S."InvNo"), 0), 0) AS "AvgItemsPerOrder",
                CASE 
                    WHEN COUNT(DISTINCT S."InvNo") > 0 
                    THEN ROUND(COUNT(DISTINCT S."InvNo") / (-:days_param/30), 2)
                    ELSE 0 
                END AS "PurchaseFrequency",
                CI."ItemGroups" AS "PreferredCategories"
            FROM 
                "AVA_Integration"."SALES_AI" S
                JOIN "AVA_Integration"."WH_ZONE_AI" W ON S."WhsCode" = W."WHSCODE"
                LEFT JOIN CustomerItems CI ON S."CardCode" = CI."CardCode"
            WHERE 
                S."InvDate" >= ADD_DAYS(CURRENT_DATE, :days_param)
            GROUP BY 
                S."CardCode", S."CustomerName", S."Mobile No", W."ZONE", CI."ItemGroups"
        )
        SELECT * FROM CustomerStats
        ORDER BY "TotalOrders" DESC
        """
        
        df = pd.read_sql(
            query, 
            engine,
            params={'days_param': -int(period_days)}
        )
        return df
    except Exception as e:
        st.error(f"Error loading HFB data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_hfb_product_data(period_days=365):
    """Load product data for HFB customers"""
    try:
        engine = create_engine("hana+hdbcli://SYSTEM:Welcome%401@192.168.238.7:30015")
        
        query = """
        WITH CustomerProductStats AS (
            SELECT 
                "CardCode" AS "CustomerID",
                "ItemCode",
                "ItemName",
                "Item Group Name",
                SUM("Quantity") AS "TotalQuantity",
                COUNT(DISTINCT "InvNo") AS "OrderCount",
                ROW_NUMBER() OVER (PARTITION BY "CardCode" ORDER BY SUM("Quantity") DESC) AS "ProductRank"
            FROM 
                "AVA_Integration"."SALES_AI"
            WHERE 
                "InvDate" >= ADD_DAYS(CURRENT_DATE, :days_param)
            GROUP BY 
                "CardCode", "ItemCode", "ItemName", "Item Group Name"
        )
        SELECT * FROM CustomerProductStats
        WHERE "ProductRank" <= 10
        """
        
        df = pd.read_sql(
            query, 
            engine,
            params={'days_param': -int(period_days)}
        )
        return df
    except Exception as e:
        st.error(f"Error loading HFB product data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_hfb_data(period_days=365):
    """Load and process High Frequency Buyers data with product recommendations"""
    # Load data with optimized query
    customer_df = load_hfb_data_original(period_days)
    product_df = load_hfb_product_data(period_days)
    
    if product_df.empty or customer_df.empty:
        customer_df['PurchasedProducts'] = ""
        customer_df['RecommendedProducts'] = ""
        return customer_df
    
    # 1. Get top purchased products
    purchased_products = (
        product_df.sort_values(['CustomerID', 'TotalQuantity'], ascending=[True, False])
        .groupby('CustomerID').head(3)
        .groupby('CustomerID')
        .apply(lambda x: ", ".join(f"{row['ItemName']} ({row['TotalQuantity']})" 
              for _, row in x.iterrows()))
        .reset_index(name='PurchasedProducts')
    )
    
    # 2. Pre-compute popular items by category
    popular_items = (
        product_df.groupby(['Item Group Name', 'ItemCode', 'ItemName'])
        ['TotalQuantity'].sum()
        .reset_index()
        .sort_values(['Item Group Name', 'TotalQuantity'], ascending=[True, False])
    )
    
    # 3. Generate recommendations
    customer_items = product_df.groupby('CustomerID')['ItemCode'].apply(set).to_dict()
    customer_categories = product_df.groupby('CustomerID')['Item Group Name'].apply(lambda x: list(x.unique())).to_dict()
    
    recommendations = []
    for customer_id in customer_df['CustomerID']:
        purchased = customer_items.get(customer_id, set())
        categories = customer_categories.get(customer_id, [])
        
        if len(categories) == 0:
            recommendations.append({'CustomerID': customer_id, 'RecommendedProducts': ''})
            continue
            
        # Get top 3 recommended items from same categories not purchased
        rec_items = popular_items[
            (popular_items['Item Group Name'].isin(categories)) &
            (~popular_items['ItemCode'].isin(purchased))
        ].head(3)['ItemName'].tolist()
        
        recommendations.append({
            'CustomerID': customer_id,
            'RecommendedProducts': ", ".join(rec_items) if rec_items else ''
        })
    
    # Merge results
    recommendations_df = pd.DataFrame(recommendations)
    final_df = customer_df.merge(purchased_products, on='CustomerID', how='left')
    final_df = final_df.merge(recommendations_df, on='CustomerID', how='left')
    
    return final_df.fillna({'PurchasedProducts': '', 'RecommendedProducts': ''})

# ----------------------------
# Navigation
# ----------------------------
def render_navigation():
    """Render the compact navigation menu"""
    st.sidebar.markdown("### 🏢 SAP B1 Analytics")

    db = get_db_instance()
    is_connected, status_msg = db.test_connection()

    if is_connected:
        st.sidebar.markdown('<div class="connection-status status-success">✅ Connected</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="connection-status status-error">❌ Disconnected</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Sales Report'

    menu_items = {
        "📊 Sales": ["Sales Report", "Branch Trends", "Customer Analysis"],
        "📦 Inventory": ["Stock Analysis", "Movement Report"],
        "💰 Purchase": ["Purchase Report", "Vendor Analysis"],
        "📈 Analytics": ["Business Intelligence", "Forecasting"],
        "⚙️ Settings": ["Database Config"]
    }

    for section, items in menu_items.items():
        st.sidebar.markdown(f"**{section}**")
        for item in items:
            if st.sidebar.button(item, key=f"nav_{item}", use_container_width=True):
                st.session_state.current_page = item
        st.sidebar.markdown("")

    return st.session_state.current_page

# ----------------------------
# Sales Report Page
# ----------------------------
def render_sales_report():
    """Enhanced sales report page"""
    st.markdown('<div class="main-header"><h1>📈 Retail Sales Intelligence</h1></div>', unsafe_allow_html=True)

    db = get_db_instance()
    is_connected, status_msg = db.test_connection()
    if not is_connected:
        st.error(f"Database connection failed: {status_msg}")
        return

    with st.spinner("Loading filter options..."):
        F = load_filter_values()

    if not F["zones"] and not F["branches"]:
        st.error("No data found in the database.")
        return

    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        st.markdown("**🔎 Filters**")

        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1.5, 1.5, 1.5, 1])

        with col1:
            start_date = st.date_input("Start Date", F["min_date"],
                                      min_value=F["min_date"], max_value=F["max_date"],
                                      key="start_date")
        with col2:
            end_date = st.date_input("End Date", F["max_date"],
                                    min_value=F["min_date"], max_value=F["max_date"],
                                    key="end_date")
        with col3:
            zone = st.selectbox("Zone", ["All"] + F["zones"], key="zone")
        with col4:
            branch = st.selectbox("Branch", ["All"] + F["branches"], key="branch")
        with col5:
            item_group = st.selectbox("Item Group", ["All"] + F["item_groups"], key="item_group")
        with col6:
            load_btn = st.button("🔄 Load", type="primary", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    if not load_btn:
        st.info("Select filters and click **Load** to generate the report.")
        return

    with st.spinner("Loading data..."):
        df = load_sales_data(start_date, end_date, zone, branch, item_group, "All")

    if df.empty:
        st.warning("No data found for the selected filters.")
        return

    st.success(f"✅ Loaded {len(df):,} records | Period: {start_date} to {end_date}")

    total_sales = df["SalesValue"].sum()
    distinct_invoices = df["InvNo"].nunique()
    calendar_days = max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1, 1)
    avg_daily_sales = total_sales / calendar_days
    avg_bill_value = total_sales / distinct_invoices if distinct_invoices else 0

    mom_growth, mom_display = get_period_comparison(df, "MOM")
    qoq_growth, qoq_display = get_period_comparison(df, "QOQ")
    yoy_growth, yoy_display = get_period_comparison(df, "YOY")

    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.subheader("📊 Key Performance Indicators")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("💰 Total Sales", in_lakhs(total_sales))
    with k2:
        st.metric("📅 Daily Avg", in_lakhs(avg_daily_sales))
    with k3:
        st.metric("🛒 Avg Bill", in_lakhs(avg_bill_value))
    with k4:
        st.metric("📈 MOM", mom_display, f"{mom_growth:+.1f}%" if mom_growth != 0 else None)
    with k5:
        st.metric("📊 QOQ", qoq_display, f"{qoq_growth:+.1f}%" if qoq_growth != 0 else None)
    with k6:
        st.metric("📆 YOY", yoy_display, f"{yoy_growth:+.1f}%" if yoy_growth != 0 else None)

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📈 Quick Insights")
    c1, c2 = st.columns(2)

    with c1:
        store_perf = df.groupby("U_AVA_Branch")["SalesValue"].sum().reset_index().sort_values("SalesValue", ascending=False).head(10)
        if not store_perf.empty:
            fig_store = px.bar(store_perf, x="U_AVA_Branch", y="SalesValue",
                               title="🏬 Top 10 Branches by Sales")
            fig_store.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_store, use_container_width=True)

    with c2:
        brand_contrib = (
            df.groupby("Brand Name")["SalesValue"]
            .sum()
            .reset_index()
            .sort_values("SalesValue", ascending=False)
            .head(10)
        )
        if not brand_contrib.empty:
            fig_brand = px.pie(brand_contrib, names="Brand Name", values="SalesValue",
                               title="🏷️ Top 10 Brand Contribution")
            fig_brand.update_layout(height=400)
            st.plotly_chart(fig_brand, use_container_width=True)

    st.subheader("🔍 Detailed Analysis Dashboard")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Top Products",
        "👥 Top Customers",
        "🧑‍💼 Top Employees",
        "📊 Performance Metrics",
        "📈 Trend Analysis"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🏆 Top 15 Products by Sales Value**")
            top_products = df.groupby("ItemName")[["SalesValue", "Quantity"]].sum().reset_index().sort_values("SalesValue", ascending=False).head(15)
            if not top_products.empty:
                top_products_display = top_products.copy()
                top_products_display["SalesValue"] = top_products_display["SalesValue"].apply(lambda x: in_lakhs(x))
                top_products_display["Quantity"] = top_products_display["Quantity"].apply(lambda x: f"{x:,.0f}")
                top_products_display.columns = ["Product Name", "Sales Value", "Qty Sold"]
                st.dataframe(top_products_display, use_container_width=True, hide_index=True, height=400)

    with tab2:
        st.markdown("**👥 Top 15 Customers by Sales Value**")
        top_customers = df.groupby("CustomerName")[["SalesValue", "Quantity"]].sum().reset_index().sort_values("SalesValue", ascending=False).head(15)
        if not top_customers.empty:
            top_customers_display = top_customers.copy()
            top_customers_display["SalesValue"] = top_customers_display["SalesValue"].apply(lambda x: in_lakhs(x))
            top_customers_display["Quantity"] = top_customers_display["Quantity"].apply(lambda x: f"{x:,.0f}")
            top_customers_display.columns = ["Customer Name", "Sales Value", "Qty Purchased"]
            st.dataframe(top_customers_display, use_container_width=True, hide_index=True, height=400)

    with tab3:
        st.markdown("**🧑‍💼 Top 15 Employees by Sales Value**")
        top_employees = df.groupby("Salesperson")[["SalesValue", "Quantity"]].sum().reset_index().sort_values("SalesValue", ascending=False).head(15)
        if not top_employees.empty:
            top_employees_display = top_employees.copy()
            top_employees_display["SalesValue"] = top_employees_display["SalesValue"].apply(lambda x: in_lakhs(x))
            top_employees_display["Quantity"] = top_employees_display["Quantity"].apply(lambda x: f"{x:,.0f}")
            top_employees_display.columns = ["Salesperson", "Sales Value", "Qty Sold"]
            st.dataframe(top_employees_display, use_container_width=True, hide_index=True, height=400)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Sales by Zone**")
            zone_sales = df.groupby("StateLoc")["SalesValue"].sum().reset_index().sort_values("SalesValue", ascending=False)
            if not zone_sales.empty:
                fig_zone = px.pie(zone_sales, names="StateLoc", values="SalesValue", title="Sales Distribution by Zone")
                st.plotly_chart(fig_zone, use_container_width=True)

        with col2:
            st.markdown("**📊 Sales by Item Group**")
            item_group_sales = df.groupby("Item Group Name")["SalesValue"].sum().reset_index().sort_values("SalesValue", ascending=False).head(10)
            if not item_group_sales.empty:
                fig_item = px.bar(item_group_sales, x="Item Group Name", y="SalesValue", title="Top 10 Item Groups by Sales")
                fig_item.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_item, use_container_width=True)

    with tab5:
        st.markdown("**📈 Sales Trend Over Time**")
        daily_sales = df.groupby("InvDate")["SalesValue"].sum().reset_index()
        if not daily_sales.empty:
            fig_trend = px.line(daily_sales, x="InvDate", y="SalesValue", title="Daily Sales Trend")
            st.plotly_chart(fig_trend, use_container_width=True)
            
# ----------------------------
# Branch Trends Page
# ----------------------------
def render_branch_trends():
    """Sales trends analysis page"""
    st.markdown('<div class="main-header"><h1>📈 Branch Trends Analysis</h1></div>', unsafe_allow_html=True)

    db = get_db_instance()
    is_connected, status_msg = db.test_connection()
    if not is_connected:
        st.error(f"Database connection failed: {status_msg}")
        return

    with st.spinner("Loading filter options..."):
        F = load_filter_values()

    if not F["zones"] and not F["branches"]:
        st.error("No data found in the database.")
        return

    # Initialize session state for dropdowns if not exists
    if 'period_type' not in st.session_state:
        st.session_state.period_type = "Month-over-Month (MOM)"
    if 'first_period' not in st.session_state:
        st.session_state.first_period = None
    if 'second_period' not in st.session_state:
        st.session_state.second_period = None

    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        st.markdown("**🔎 Filters**")

        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1.5, 1.5, 1.5, 1])

        with col1:
            start_date = st.date_input("Start Date", F["min_date"],
                                      min_value=F["min_date"], max_value=F["max_date"],
                                      key="trend_start_date")
        with col2:
            end_date = st.date_input("End Date", F["max_date"],
                                    min_value=F["min_date"], max_value=F["max_date"],
                                    key="trend_end_date")
        with col3:
            zone = st.selectbox("Zone", ["All"] + F["zones"], key="trend_zone")
        with col4:
            branch = st.selectbox("Branch", ["All"] + F["branches"], key="trend_branch")
        with col5:
            item_group = st.selectbox("Item Group", ["All"] + F["item_groups"], key="trend_item_group")
        with col6:
            load_btn = st.button("🔄 Load", type="primary", use_container_width=True, key="trend_load")

        st.markdown('</div>', unsafe_allow_html=True)

    if not load_btn and 'sales_trend_df' not in st.session_state:
        st.info("Select filters and click **Load** to generate the trend analysis.")
        return

    # Load data only when Load button is clicked or if we already have data
    if load_btn or 'sales_trend_df' in st.session_state:
        if load_btn:
            with st.spinner("Loading data..."):
                df = load_sales_data(start_date, end_date, zone, branch, item_group, "All")
                st.session_state.sales_trend_df = df
                st.session_state.trend_filters = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'zone': zone,
                    'branch': branch,
                    'item_group': item_group
                }
                # Reset period selections when new data is loaded
                st.session_state.first_period = None
                st.session_state.second_period = None
        else:
            # Use cached data
            df = st.session_state.sales_trend_df
            filters = st.session_state.trend_filters
            start_date, end_date, zone, branch, item_group = (
                filters['start_date'], filters['end_date'], filters['zone'], 
                filters['branch'], filters['item_group']
            )

        if df.empty:
            st.warning("No data found for the selected filters.")
            if 'sales_trend_df' in st.session_state:
                del st.session_state.sales_trend_df
            return

        st.success(f"✅ Loaded {len(df):,} records | Period: {start_date} to {end_date}")

        # Convert dates to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(df['InvDate']):
            df['InvDate'] = pd.to_datetime(df['InvDate'])

        # 1. Top 10 branches by zone-wise
        st.markdown("## 🏆 Top 10 Branches by Zone")
        
        # Get all zones if "All" is selected
        if zone == "All":
            zones_to_show = F["zones"]
        else:
            zones_to_show = [zone]
        
        for zone_name in zones_to_show:
            zone_df = df[df['StateLoc'] == zone_name] if zone_name != "All" else df
            
            if not zone_df.empty:
                branch_sales = zone_df.groupby('U_AVA_Branch')['SalesValue'].sum().reset_index()
                branch_sales = branch_sales.sort_values('SalesValue', ascending=False).head(10)
                
                if not branch_sales.empty:
                    st.markdown(f"### {zone_name}")
                    fig = px.bar(branch_sales, x='U_AVA_Branch', y='SalesValue',
                                title=f'Top 10 Branches in {zone_name}',
                                labels={'U_AVA_Branch': 'Branch', 'SalesValue': 'Sales Value'})
                    fig.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No branch data available for {zone_name}")
            else:
                st.info(f"No data available for {zone_name}")

        # 2. Last month vs previous month sales growth by zone
        st.markdown("## 📊 Month-over-Month Sales Growth by Zone")
        
        # Calculate last month and previous month
        end_date_dt = pd.to_datetime(end_date)
        last_month_start = (end_date_dt.replace(day=1))
        last_month_end = end_date_dt
        
        # Calculate previous month
        prev_month_end = last_month_start - pd.DateOffset(days=1)
        prev_month_start = prev_month_end.replace(day=1)
        
        # Filter data for last month and previous month
        last_month_df = df[(df['InvDate'] >= last_month_start) & (df['InvDate'] <= last_month_end)]
        previous_month_df = df[(df['InvDate'] >= prev_month_start) & (df['InvDate'] <= prev_month_end)]
        
        # Group by zone for MOM comparison
        last_month_zone = last_month_df.groupby('StateLoc')['SalesValue'].sum().reset_index()
        previous_month_zone = previous_month_df.groupby('StateLoc')['SalesValue'].sum().reset_index()
        
        # Merge the data for comparison
        mom_comparison = pd.merge(previous_month_zone, last_month_zone, on='StateLoc', 
                                 how='outer', suffixes=('_prev', '_last'))
        mom_comparison = mom_comparison.fillna(0)
        
        # Calculate growth percentage
        mom_comparison['Growth_Percent'] = mom_comparison.apply(
            lambda x: ((x['SalesValue_last'] - x['SalesValue_prev']) / x['SalesValue_prev'] * 100) 
            if x['SalesValue_prev'] > 0 else 0, axis=1
        )
        
        # Display MOM comparison by zone
        if not mom_comparison.empty:
            fig_mom_zone = px.bar(mom_comparison, x='StateLoc', y='Growth_Percent',
                                 title='Month-over-Month Growth by Zone (%)',
                                 labels={'StateLoc': 'Zone', 'Growth_Percent': 'Growth Percentage'},
                                 color='Growth_Percent',
                                 color_continuous_scale=['red', 'white', 'green'])
            fig_mom_zone.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_mom_zone, use_container_width=True)
            
            # Also show the actual sales values by zone
            mom_sales_comparison = pd.melt(mom_comparison, id_vars=['StateLoc'], 
                                          value_vars=['SalesValue_prev', 'SalesValue_last'],
                                          var_name='Period', value_name='SalesValue')
            mom_sales_comparison['Period'] = mom_sales_comparison['Period'].replace({
                'SalesValue_prev': 'Previous Month',
                'SalesValue_last': 'Last Month'
            })
            
            fig_mom_sales = px.bar(mom_sales_comparison, x='StateLoc', y='SalesValue', color='Period',
                                  title='Monthly Sales Comparison by Zone',
                                  labels={'StateLoc': 'Zone', 'SalesValue': 'Sales Value'},
                                  barmode='group')
            fig_mom_sales.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_mom_sales, use_container_width=True)
        else:
            st.info("No data available for MOM comparison by zone")

        # 3. Quarter-over-Quarter comparison by zone
        st.markdown("## 📊 Quarter-over-Quarter Sales Growth by Zone")
        
        # Calculate current quarter and previous quarter
        current_quarter = pd.Period(end_date_dt, freq='Q')
        previous_quarter = current_quarter - 1
        
        # Get start and end dates for current and previous quarters
        current_quarter_start = current_quarter.start_time
        current_quarter_end = current_quarter.end_time
        previous_quarter_start = previous_quarter.start_time
        previous_quarter_end = previous_quarter.end_time
        
        # Filter data for current and previous quarters
        current_quarter_df = df[(df['InvDate'] >= current_quarter_start) & (df['InvDate'] <= current_quarter_end)]
        previous_quarter_df = df[(df['InvDate'] >= previous_quarter_start) & (df['InvDate'] <= previous_quarter_end)]
        
        # Group by zone for QOQ comparison
        current_quarter_zone = current_quarter_df.groupby('StateLoc')['SalesValue'].sum().reset_index()
        previous_quarter_zone = previous_quarter_df.groupby('StateLoc')['SalesValue'].sum().reset_index()
        
        # Merge the data for QOQ comparison
        qoq_comparison = pd.merge(previous_quarter_zone, current_quarter_zone, on='StateLoc', 
                                 how='outer', suffixes=('_prev', '_current'))
        qoq_comparison = qoq_comparison.fillna(0)
        
        # Calculate growth percentage
        qoq_comparison['Growth_Percent'] = qoq_comparison.apply(
            lambda x: ((x['SalesValue_current'] - x['SalesValue_prev']) / x['SalesValue_prev'] * 100) 
            if x['SalesValue_prev'] > 0 else 0, axis=1
        )
        
        # Display QOQ comparison by zone
        if not qoq_comparison.empty:
            # QOQ Growth percentage
            fig_qoq_zone = px.bar(qoq_comparison, x='StateLoc', y='Growth_Percent',
                                 title='Quarter-over-Quarter Growth by Zone (%)',
                                 labels={'StateLoc': 'Zone', 'Growth_Percent': 'Growth Percentage'},
                                 color='Growth_Percent',
                                 color_continuous_scale=['red', 'white', 'green'])
            fig_qoq_zone.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_qoq_zone, use_container_width=True)
            
            # Also show the actual sales values by zone
            qoq_sales_comparison = pd.melt(qoq_comparison, id_vars=['StateLoc'], 
                                          value_vars=['SalesValue_prev', 'SalesValue_current'],
                                          var_name='Period', value_name='SalesValue')
            qoq_sales_comparison['Period'] = qoq_sales_comparison['Period'].replace({
                'SalesValue_prev': f'Q{previous_quarter.quarter} {previous_quarter.year}',
                'SalesValue_current': f'Q{current_quarter.quarter} {current_quarter.year}'
            })
            
            fig_qoq_sales = px.bar(qoq_sales_comparison, x='StateLoc', y='SalesValue', color='Period',
                                  title='Quarterly Sales Comparison by Zone',
                                  labels={'StateLoc': 'Zone', 'SalesValue': 'Sales Value'},
                                  barmode='group')
            fig_qoq_sales.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_qoq_sales, use_container_width=True)
        else:
            st.info("No data available for QOQ comparison by zone")

        # 4. Custom period comparison
        st.markdown("## 🔍 Custom Period Comparison")
        
        # Get unique months and quarters from the data
        df['YearMonth'] = df['InvDate'].dt.strftime('%Y-%m')
        df['YearQuarter'] = df['InvDate'].dt.to_period('Q').astype(str)
        
        available_months = sorted(df['YearMonth'].unique())
        available_quarters = sorted(df['YearQuarter'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use session state to maintain the selected period type
            period_type = st.selectbox(
                "Select Period Type",
                options=["Month-over-Month (MOM)", "Quarter-over-Quarter (QOQ)"],
                index=0 if st.session_state.period_type == "Month-over-Month (MOM)" else 1,
                key="period_type_select"
            )
            
            # Update session state when selection changes
            if period_type != st.session_state.period_type:
                st.session_state.period_type = period_type
                st.session_state.first_period = None
                st.session_state.second_period = None
                st.rerun()
        
        with col2:
            if st.session_state.period_type == "Month-over-Month (MOM)":
                if len(available_months) >= 2:
                    # Set default values if not set
                    if st.session_state.first_period is None:
                        st.session_state.first_period = available_months[-2] if len(available_months) >= 2 else available_months[0]
                    if st.session_state.second_period is None:
                        st.session_state.second_period = available_months[-1] if len(available_months) >= 1 else available_months[0]
                    
                    # Create selectboxes for both periods
                    col_a, col_b = st.columns(2)
                    with col_a:
                        first_period = st.selectbox(
                            "Select First Month",
                            options=available_months,
                            index=available_months.index(st.session_state.first_period),
                            key="first_month_select"
                        )
                    with col_b:
                        second_period = st.selectbox(
                            "Select Second Month",
                            options=available_months,
                            index=available_months.index(st.session_state.second_period),
                            key="second_month_select"
                        )
                    
                    # Update session state when selections change
                    if first_period != st.session_state.first_period:
                        st.session_state.first_period = first_period
                        st.rerun()
                    
                    if second_period != st.session_state.second_period:
                        st.session_state.second_period = second_period
                        st.rerun()
                    
                    st.info(f"Comparing {st.session_state.first_period} vs {st.session_state.second_period}")
                else:
                    st.info("Not enough months data available for comparison")
            else:  # QOQ
                if len(available_quarters) >= 2:
                    # Set default values if not set
                    if st.session_state.first_period is None:
                        st.session_state.first_period = available_quarters[-2] if len(available_quarters) >= 2 else available_quarters[0]
                    if st.session_state.second_period is None:
                        st.session_state.second_period = available_quarters[-1] if len(available_quarters) >= 1 else available_quarters[0]
                    
                    # Create selectboxes for both periods
                    col_a, col_b = st.columns(2)
                    with col_a:
                        first_period = st.selectbox(
                            "Select First Quarter",
                            options=available_quarters,
                            index=available_quarters.index(st.session_state.first_period),
                            key="first_quarter_select"
                        )
                    with col_b:
                        second_period = st.selectbox(
                            "Select Second Quarter",
                            options=available_quarters,
                            index=available_quarters.index(st.session_state.second_period),
                            key="second_quarter_select"
                        )
                    
                    # Update session state when selections change
                    if first_period != st.session_state.first_period:
                        st.session_state.first_period = first_period
                        st.rerun()
                    
                    if second_period != st.session_state.second_period:
                        st.session_state.second_period = second_period
                        st.rerun()
                    
                    st.info(f"Comparing {st.session_state.first_period} vs {st.session_state.second_period}")
                else:
                    st.info("Not enough quarters data available for comparison")
        
        # Calculate and display the growth based on selection
        if st.session_state.period_type == "Month-over-Month (MOM)" and st.session_state.first_period and st.session_state.second_period:
            # Filter data for selected months
            first_df = df[df['YearMonth'] == st.session_state.first_period]
            second_df = df[df['YearMonth'] == st.session_state.second_period]
            
            # Group by zone for custom MOM comparison
            first_zone = first_df.groupby('StateLoc')['SalesValue'].sum().reset_index()
            second_zone = second_df.groupby('StateLoc')['SalesValue'].sum().reset_index()
            
            # Merge the data for comparison
            custom_mom_comparison = pd.merge(first_zone, second_zone, on='StateLoc', 
                                           how='outer', suffixes=('_first', '_second'))
            custom_mom_comparison = custom_mom_comparison.fillna(0)
            
            # Calculate growth percentage
            custom_mom_comparison['Growth_Percent'] = custom_mom_comparison.apply(
                lambda x: ((x['SalesValue_second'] - x['SalesValue_first']) / x['SalesValue_first'] * 100) 
                if x['SalesValue_first'] > 0 else 0, axis=1
            )
            
            # Display custom MOM comparison by zone
            if not custom_mom_comparison.empty:
                fig_custom_mom = px.bar(custom_mom_comparison, x='StateLoc', y='Growth_Percent',
                                       title=f'Custom MOM Growth by Zone: {st.session_state.first_period} vs {st.session_state.second_period} (%)',
                                       labels={'StateLoc': 'Zone', 'Growth_Percent': 'Growth Percentage'},
                                       color='Growth_Percent',
                                       color_continuous_scale=['red', 'white', 'green'])
                fig_custom_mom.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_custom_mom, use_container_width=True)
                
                # Also show the actual sales values by zone
                custom_mom_sales = pd.melt(custom_mom_comparison, id_vars=['StateLoc'], 
                                          value_vars=['SalesValue_first', 'SalesValue_second'],
                                          var_name='Period', value_name='SalesValue')
                custom_mom_sales['Period'] = custom_mom_sales['Period'].replace({
                    'SalesValue_first': st.session_state.first_period,
                    'SalesValue_second': st.session_state.second_period
                })
                
                fig_custom_sales = px.bar(custom_mom_sales, x='StateLoc', y='SalesValue', color='Period',
                                         title=f'Sales Comparison by Zone: {st.session_state.first_period} vs {st.session_state.second_period}',
                                         labels={'StateLoc': 'Zone', 'SalesValue': 'Sales Value'},
                                         barmode='group')
                fig_custom_sales.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_custom_sales, use_container_width=True)
            else:
                st.info("No data available for custom MOM comparison by zone")
            
        elif st.session_state.period_type == "Quarter-over-Quarter (QOQ)" and st.session_state.first_period and st.session_state.second_period:
            # Filter data for selected quarters
            first_df = df[df['YearQuarter'] == st.session_state.first_period]
            second_df = df[df['YearQuarter'] == st.session_state.second_period]
            
            # Group by zone for custom QOQ comparison
            first_zone = first_df.groupby('StateLoc')['SalesValue'].sum().reset_index()
            second_zone = second_df.groupby('StateLoc')['SalesValue'].sum().reset_index()
            
            # Merge the data for comparison
            custom_qoq_comparison = pd.merge(first_zone, second_zone, on='StateLoc', 
                                           how='outer', suffixes=('_first', '_second'))
            custom_qoq_comparison = custom_qoq_comparison.fillna(0)
            
            # Calculate growth percentage
            custom_qoq_comparison['Growth_Percent'] = custom_qoq_comparison.apply(
                lambda x: ((x['SalesValue_second'] - x['SalesValue_first']) / x['SalesValue_first'] * 100) 
                if x['SalesValue_first'] > 0 else 0, axis=1
            )
            
            # Display custom QOQ comparison by zone
            if not custom_qoq_comparison.empty:
                fig_custom_qoq = px.bar(custom_qoq_comparison, x='StateLoc', y='Growth_Percent',
                                       title=f'Custom QOQ Growth by Zone: {st.session_state.first_period} vs {st.session_state.second_period} (%)',
                                       labels={'StateLoc': 'Zone', 'Growth_Percent': 'Growth Percentage'},
                                       color='Growth_Percent',
                                       color_continuous_scale=['red', 'white', 'green'])
                fig_custom_qoq.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_custom_qoq, use_container_width=True)
                
                # Also show the actual sales values by zone
                custom_qoq_sales = pd.melt(custom_qoq_comparison, id_vars=['StateLoc'], 
                                          value_vars=['SalesValue_first', 'SalesValue_second'],
                                          var_name='Period', value_name='SalesValue')
                custom_qoq_sales['Period'] = custom_qoq_sales['Period'].replace({
                    'SalesValue_first': st.session_state.first_period,
                    'SalesValue_second': st.session_state.second_period
                })
                
                fig_custom_sales = px.bar(custom_qoq_sales, x='StateLoc', y='SalesValue', color='Period',
                                         title=f'Sales Comparison by Zone: {st.session_state.first_period} vs {st.session_state.second_period}',
                                         labels={'StateLoc': 'Zone', 'SalesValue': 'Sales Value'},
                                         barmode='group')
                fig_custom_sales.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_custom_sales, use_container_width=True)
            else:
                st.info("No data available for custom QOQ comparison by zone")

        # Additional trend analysis
        st.markdown("## 📈 Sales Trend Over Time by Zone")
        
        # Group by month and zone for trend analysis
        monthly_trend_zone = df.groupby(['YearMonth', 'StateLoc'])['SalesValue'].sum().reset_index()
        
        if not monthly_trend_zone.empty:
            fig = px.line(monthly_trend_zone, x='YearMonth', y='SalesValue', color='StateLoc',
                         title='Monthly Sales Trend by Zone',
                         labels={'SalesValue': 'Sales Value', 'YearMonth': 'Month', 'StateLoc': 'Zone'})
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.markdown("## 📥 Export Data")
        if st.button("📄 Export Trend Data to CSV", use_container_width=True):
            # Prepare data for export
            export_df = df.copy()
            monthly_export = export_df.groupby(['YearMonth', 'StateLoc', 'U_AVA_Branch'])['SalesValue'].sum().reset_index()
            
            csv_data = monthly_export.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"sales_trend_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ----------------------------
# Customer Analysis Page with HFB Integration
# ----------------------------
def render_customer_analysis():
    """Customer Analysis page with High Frequency Buyers integration"""
    st.markdown('<div class="main-header"><h1>👥 Customer Analysis</h1></div>', unsafe_allow_html=True)

    db = get_db_instance()
    is_connected, status_msg = db.test_connection()
    if not is_connected:
        st.error(f"Database connection failed: {status_msg}")
        return

    # HFB Analysis Section
    st.markdown("## 📊 High Frequency Buyers Analysis")
    st.markdown("Identify and analyze your most frequent customers by zone")

    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        st.markdown("**🔎 HFB Filters**")

        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            period_days = st.selectbox(
                "Analysis Period:",
                options=[
                    {'label': 'Last 3 Months', 'value': 90},
                    {'label': 'Last 6 Months', 'value': 180},
                    {'label': 'Last 12 Months', 'value': 365},
                    {'label': 'Last 2 Years', 'value': 730},
                    {'label': 'Last 3 Years', 'value': 1095}
                ],
                format_func=lambda x: x['label'],
                index=2,  # Default to 12 months
                key="hfb_period"
            )['value']

        with col2:
            min_orders = st.number_input(
                "Minimum Orders:",
                min_value=1,
                value=5,
                step=1,
                key="hfb_min_orders"
            )

        with col3:
            # Load zones for filter
            zones = []
            try:
                zones_query = 'SELECT DISTINCT "ZONE" FROM "AVA_Integration"."WH_ZONE_AI" WHERE "ZONE" IS NOT NULL'
                zones_df = db.execute_query(zones_query)
                if not zones_df.empty:
                    zones = sorted([str(z).strip() for z in zones_df['ZONE'].dropna().unique() if str(z).strip() not in ('', 'nan')])
            except:
                pass
            
            selected_zones = st.multiselect(
                "Zone Filter:",
                options=zones,
                placeholder="All Zones",
                key="hfb_zone_filter"
            )

        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            load_hfb_btn = st.button("🔄 Load HFB Data", type="primary", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    if not load_hfb_btn:
        st.info("Configure filters and click **Load HFB Data** to analyze high frequency buyers.")
        
        # Show basic customer insights from sales data
        st.markdown("## 📈 Customer Overview")
        with st.spinner("Loading customer overview..."):
            try:
                # Load recent sales data for basic customer insights
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=90)
                sales_df = load_sales_data(start_date, end_date, "All", "All", "All", "All")
                
                if not sales_df.empty:
                    total_customers = sales_df["CustomerName"].nunique()
                    repeat_customers = len(sales_df.groupby("CustomerName").filter(lambda x: len(x) > 1)["CustomerName"].unique())
                    avg_order_value = sales_df.groupby("InvNo")["SalesValue"].sum().mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", f"{total_customers:,}")
                    with col2:
                        st.metric("Repeat Customers", f"{repeat_customers:,}", f"{(repeat_customers/total_customers*100):.1f}%" if total_customers > 0 else "0%")
                    with col3:
                        st.metric("Avg Order Value", in_lakhs(avg_order_value))
            except Exception as e:
                st.warning(f"Could not load customer overview: {e}")
        
        return

    # Load HFB Data
    with st.spinner("Loading High Frequency Buyers data..."):
        hfb_df = load_hfb_data(period_days)

    if hfb_df.empty:
        st.warning("No High Frequency Buyers data found for the selected period.")
        return

    st.success(f"✅ Loaded {len(hfb_df):,} high frequency buyers")

    # Apply filters
    filtered_hfb = hfb_df.copy()
    if selected_zones:
        filtered_hfb = filtered_hfb[filtered_hfb['Zone'].isin(selected_zones)]
    if min_orders > 1:
        filtered_hfb = filtered_hfb[filtered_hfb['TotalOrders'] >= min_orders]

    # Display Metrics
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.subheader("📊 HFB Performance Metrics")

    if not filtered_hfb.empty:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total HFBs", f"{len(filtered_hfb):,}")
        with m2:
            st.metric("Avg Orders", f"{filtered_hfb['TotalOrders'].mean():.1f}")
        with m3:
            st.metric("Avg Spend", in_lakhs(filtered_hfb['TotalSpend'].mean()))
        with m4:
            st.metric("Avg Items/Order", f"{filtered_hfb['AvgItemsPerOrder'].mean():.1f}")
        with m5:
            st.metric("Avg Days Between", f"{filtered_hfb['AvgDaysBetweenOrders'].mean():.1f}")
    else:
        st.info("No high frequency buyers match the selected filters.")

    st.markdown('</div>', unsafe_allow_html=True)

    # HFB Table
    st.markdown("## 📋 High Frequency Buyers Details")
    
    # Prepare data for display
    display_df = filtered_hfb.copy()
    display_df = display_df[[
        'CustomerName', 'Mobile', 'Zone', 'TotalOrders', 'TotalSpend',
        'AvgDaysBetweenOrders', 'AvgItemsPerOrder', 'PurchaseFrequency',
        'DaysSinceLastPurchase', 'PurchasedProducts', 'RecommendedProducts',
        'PreferredCategories'
    ]]
    
    display_df.columns = [
        'Customer', 'Mobile', 'Zone', 'Total Orders', 'Total Spend',
        'Avg Days Between Orders', 'Avg Items/Order', 'Purchase Frequency/Month',
        'Last Purchase (Days)', 'Purchased Products', 'Recommended Products',
        'Preferred Categories'
    ]
    
    # Format numeric columns
    display_df['Total Spend'] = display_df['Total Spend'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Avg Days Between Orders'] = display_df['Avg Days Between Orders'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    display_df['Purchase Frequency/Month'] = display_df['Purchase Frequency/Month'].apply(lambda x: f"{x:.1f}")

    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "Customer": st.column_config.TextColumn("Customer", width="medium"),
            "Mobile": st.column_config.TextColumn("Mobile", width="small"),
            "Zone": st.column_config.TextColumn("Zone", width="small"),
            "Total Orders": st.column_config.NumberColumn("Total Orders", format="%d"),
            "Total Spend": st.column_config.TextColumn("Total Spend"),
            "Avg Days Between Orders": st.column_config.TextColumn("Avg Days"),
            "Avg Items/Order": st.column_config.NumberColumn("Avg Items", format="%.1f"),
            "Purchase Frequency/Month": st.column_config.TextColumn("Freq/Month"),
            "Last Purchase (Days)": st.column_config.NumberColumn("Last Purchase", format="%d"),
            "Purchased Products": st.column_config.TextColumn("Purchased Products", width="large"),
            "Recommended Products": st.column_config.TextColumn("Recommended", width="large"),
            "Preferred Categories": st.column_config.TextColumn("Categories", width="medium")
        }
    )

    # Visualizations
    st.markdown("## 📈 Customer Insights")

    if not filtered_hfb.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Top customers by frequency
            top_freq = filtered_hfb.nlargest(10, 'PurchaseFrequency')
            fig_freq = px.bar(
                top_freq,
                x='CustomerName',
                y='PurchaseFrequency',
                title='Top 10 Customers by Purchase Frequency',
                labels={'PurchaseFrequency': 'Orders per Month', 'CustomerName': 'Customer'}
            )
            fig_freq.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_freq, use_container_width=True)

        with col2:
            # Top customers by spend
            top_spend = filtered_hfb.nlargest(10, 'TotalSpend')
            fig_spend = px.bar(
                top_spend,
                x='CustomerName',
                y='TotalSpend',
                title='Top 10 Customers by Total Spend',
                labels={'TotalSpend': 'Total Spend (₹)', 'CustomerName': 'Customer'}
            )
            fig_spend.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_spend, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            # Zone distribution
            zone_dist = filtered_hfb['Zone'].value_counts().reset_index()
            zone_dist.columns = ['Zone', 'Count']
            fig_zone = px.pie(
                zone_dist,
                names='Zone',
                values='Count',
                title='HFB Distribution by Zone'
            )
            st.plotly_chart(fig_zone, use_container_width=True)

        with col4:
            # Purchase frequency distribution
            fig_hist = px.histogram(
                filtered_hfb,
                x='PurchaseFrequency',
                nbins=20,
                title='Purchase Frequency Distribution',
                labels={'PurchaseFrequency': 'Orders per Month'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # Export functionality
    st.markdown("## 📥 Export Data")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Export HFB Data to CSV", use_container_width=True):
            csv_data = filtered_hfb.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"hfb_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:
        if st.button("🎯 Export Recommendations", use_container_width=True):
            rec_data = filtered_hfb[['CustomerName', 'Mobile', 'Zone', 'RecommendedProducts', 'PreferredCategories']]
            csv_data = rec_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Recommendations",
                data=csv_data,
                file_name=f"customer_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Actionable Insights
    st.markdown("## 🎯 Actionable Insights")

    if not filtered_hfb.empty:
        # Customers who haven't purchased recently
        inactive_customers = filtered_hfb[filtered_hfb['DaysSinceLastPurchase'] > 90]
        if not inactive_customers.empty:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown(f"**⏰ {len(inactive_customers)} Inactive High-Value Customers**")
            st.markdown("Customers who haven't purchased in over 90 days:")
            for _, customer in inactive_customers.head(5).iterrows():
                st.markdown(f"• {customer['CustomerName']} - {customer['DaysSinceLastPurchase']} days since last purchase")
            st.markdown('</div>', unsafe_allow_html=True)

        # Most frequent buyers
        frequent_buyers = filtered_hfb[filtered_hfb['PurchaseFrequency'] > 4]
        if not frequent_buyers.empty:
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown(f"**⭐ {len(frequent_buyers)} Highly Engaged Customers**")
            st.markdown("Consider loyalty programs or exclusive offers:")
            for _, customer in frequent_buyers.head(5).iterrows():
                st.markdown(f"• {customer['CustomerName']} - {customer['PurchaseFrequency']:.1f} orders/month")
            st.markdown('</div>', unsafe_allow_html=True)

        # High spenders
        high_spenders = filtered_hfb[filtered_hfb['TotalSpend'] > filtered_hfb['TotalSpend'].quantile(0.75)]
        if not high_spenders.empty:
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown(f"**💰 {len(high_spenders)} High-Value Spenders**")
            st.markdown("Premium customers for personalized attention:")
            for _, customer in high_spenders.head(5).iterrows():
                st.markdown(f"• {customer['CustomerName']} - ₹{customer['TotalSpend']:,.0f} total spend")
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# NEW: Inventory Stock Analysis Page
# ----------------------------
def render_stock_analysis():
    """Comprehensive inventory stock analysis page"""
    st.markdown('<div class="main-header"><h1>📦 Inventory Stock Analysis</h1></div>', unsafe_allow_html=True)

    db = get_db_instance()
    is_connected, status_msg = db.test_connection()
    if not is_connected:
        st.error(f"Database connection failed: {status_msg}")
        return

    with st.spinner("Loading inventory data..."):
        inventory_df = load_inventory_data()

    if inventory_df.empty:
        st.warning("No inventory data available.")
        return

    st.success(f"✅ Loaded {len(inventory_df):,} inventory items")

    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    st.markdown("**🔎 Inventory Filters**")

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])

    with col1:
        zone_filter = st.selectbox("Zone", ["All"] + sorted(inventory_df["Zone"].unique()), key="inv_zone")
    with col2:
        branch_filter = st.selectbox("Branch", ["All"] + sorted(inventory_df["Branch"].unique()), key="inv_branch")
    with col3:
        status_filter = st.selectbox("Stock Status", ["All", "Understock", "Optimal", "Overstock"], key="inv_status")
    with col4:
        age_filter = st.selectbox("Age Status", ["All", "Fresh", "Aging", "Aged Stock"], key="inv_age")
    with col5:
        refresh_btn = st.button("🔄 Refresh", type="primary", use_container_width=True, key="inv_refresh")

    st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = inventory_df.copy()
    if zone_filter != "All":
        filtered_df = filtered_df[filtered_df["Zone"] == zone_filter]
    if branch_filter != "All":
        filtered_df = filtered_df[filtered_df["Branch"] == branch_filter]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["Stock_Status"] == status_filter]
    if age_filter != "All":
        filtered_df = filtered_df[filtered_df["Age_Status"] == age_filter]

    total_items = len(filtered_df)
    understock_items = len(filtered_df[filtered_df["Stock_Status"] == "Understock"])
    overstock_items = len(filtered_df[filtered_df["Stock_Status"] == "Overstock"])
    aged_items = len(filtered_df[filtered_df["Age_Status"] == "Aged Stock"])
    critical_items = len(filtered_df[filtered_df["Stock_Coverage_Days"] < 7])
    total_stock_units = filtered_df["Current_Stock"].sum()

    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.subheader("📊 Inventory Summary")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("📦 Total Items", f"{total_items:,}")
    with m2:
        st.metric("🔴 Understock", f"{understock_items:,}", f"{(understock_items/total_items*100):.1f}%" if total_items > 0 else None)
    with m3:
        st.metric("🟡 Overstock", f"{overstock_items:,}", f"{(overstock_items/total_items*100):.1f}%" if total_items > 0 else None)
    with m4:
        st.metric("🕐 Aged Stock", f"{aged_items:,}", f"{(aged_items/total_items*100):.1f}%" if total_items > 0 else None)
    with m5:
        st.metric("🚨 Critical (<7 days)", f"{critical_items:,}")
    with m6:
        st.metric("📊 Total Stock Units", f"{total_stock_units:,.0f}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🎯 Immediate Actions Required")
    col1, col2, col3 = st.columns(3)

    with col1:
        critical_df = filtered_df[filtered_df["Stock_Coverage_Days"] < 7]
        st.markdown('<div class="critical-box">', unsafe_allow_html=True)
        st.markdown(f"**🚨 {len(critical_df)} Critical Items**")
        st.markdown("Items with less than 7 days coverage")
        for _, item in critical_df.head(3).iterrows():
            st.markdown(f"• {item['ItemCode'][:20]}... - {item['Stock_Coverage_Days']:.1f} days")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        aged_df = filtered_df[filtered_df["Age_Status"] == "Aged Stock"]
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(f"**⏰ {len(aged_df)} Aged Items**")
        st.markdown("Items older than 90 days")
        for _, item in aged_df.head(3).iterrows():
            st.markdown(f"• {item['ItemCode'][:20]}... - {item['Oldest_Item_Age_Days']} days")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        overstock_df = filtered_df[filtered_df["Stock_Status"] == "Overstock"]
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown(f"**📦 {len(overstock_df)} Overstock Items**")
        st.markdown("Consider transfer or promotions")
        for _, item in overstock_df.head(3).iterrows():
            st.markdown(f"• {item['ItemCode'][:20]}... - {item['Stock_Coverage_Days']:.1f} days")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📈 Inventory Analytics")
    a1, a2 = st.columns(2)

    with a1:
        status_counts = filtered_df['Stock_Status'].value_counts()
        if not status_counts.empty:
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Stock Status Distribution",
                color_discrete_map={'Understock': '#e74c3c', 'Optimal': '#2ecc71', 'Overstock': '#f39c12'}
            )
            fig_status.update_layout(height=350)
            st.plotly_chart(fig_status, use_container_width=True)

    with a2:
        age_counts = filtered_df['Age_Status'].value_counts()
        if not age_counts.empty:
            fig_age = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                title="Inventory Age Distribution"
            )
            fig_age.update_layout(height=350)
            st.plotly_chart(fig_age, use_container_width=True)

    b1, b2 = st.columns(2)

    with b1:
        zone_analysis = filtered_df.groupby(['Zone', 'Stock_Status']).size().unstack(fill_value=0)
        if not zone_analysis.empty:
            fig_zone = px.bar(
                zone_analysis,
                title="Stock Status by Zone",
                barmode='stack',
                color_discrete_map={'Understock': '#e74c3c', 'Optimal': '#2ecc71', 'Overstock': '#f39c12'}
            )
            fig_zone.update_layout(height=350)
            st.plotly_chart(fig_zone, use_container_width=True)

    with b2:
        # ✅ FIXED: use nbins (not 'bins') for Plotly Express histogram
        fig_coverage = px.histogram(
            filtered_df,
            x="Stock_Coverage_Days",
            nbins=20,
            title="Stock Coverage Distribution (Days)"
        )
        fig_coverage.add_vline(x=7, line_dash="dash", line_color="red", annotation_text="Critical Level")
        fig_coverage.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="Optimal Level")
        fig_coverage.update_layout(height=350)
        st.plotly_chart(fig_coverage, use_container_width=True)

    st.subheader("🔍 Detailed Inventory Analysis")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Item Details",
        "🎯 Recommendations",
        "📊 Performance Metrics",
        "📈 Trend Analysis"
    ])

    with tab1:
        st.markdown("**📋 Inventory Item Details**")
        display_df = filtered_df.copy()
        display_df = display_df[[
            'ItemCode', 'ItemName', 'Branch', 'Zone', 'Current_Stock',
            'Stock_Coverage_Days', 'Stock_Status', 'Age_Status',
            'Oldest_Item_Age_Days', 'Sales_60_Days'
        ]]
        display_df['Stock_Coverage_Days'] = display_df['Stock_Coverage_Days'].apply(lambda x: f"{x:.1f}")
        display_df['Sales_60_Days'] = display_df['Sales_60_Days'].apply(lambda x: f"{x:.0f}")
        display_df.columns = [
            'Item Code', 'Item Name', 'Branch', 'Zone', 'Current Stock',
            'Coverage (Days)', 'Stock Status', 'Age Status', 'Age (Days)', '60-Day Sales'
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("**🎯 Smart Recommendations**")
        purchase_recs = filtered_df[filtered_df["Stock_Status"] == "Understock"].head(10)
        if not purchase_recs.empty:
            st.markdown("**🛒 Purchase Recommendations**")
            for _, item in purchase_recs.iterrows():
                urgency = "🚨 URGENT" if item['Stock_Coverage_Days'] < 7 else "⚠️ HIGH" if item['Stock_Coverage_Days'] < 15 else "📝 MEDIUM"
                st.markdown(f"• {urgency}: **{item['ItemCode']}** - Coverage: {item['Stock_Coverage_Days']:.1f} days")

        transfer_recs = filtered_df[filtered_df["Stock_Status"] == "Overstock"].head(10)
        if not transfer_recs.empty:
            st.markdown("**📦 Transfer/Clearance Recommendations**")
            for _, item in transfer_recs.iterrows():
                action = "🕐 CLEARANCE" if item['Age_Status'] == 'Aged Stock' else "📦 TRANSFER"
                st.markdown(f"• {action}: **{item['ItemCode']}** - Coverage: {item['Stock_Coverage_Days']:.1f} days, Age: {item['Oldest_Item_Age_Days']} days")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 Zone Performance**")
            zone_metrics = filtered_df.groupby('Zone').agg({
                'Current_Stock': 'sum',
                'Stock_Coverage_Days': 'mean',
                'Oldest_Item_Age_Days': 'mean'
            }).round(1)
            zone_metrics.columns = ['Total Stock', 'Avg Coverage (Days)', 'Avg Age (Days)']
            st.dataframe(zone_metrics, use_container_width=True)
        with c2:
            st.markdown("**🏪 Branch Performance**")
            branch_metrics = filtered_df.groupby('Branch').agg({
                'Current_Stock': 'sum',
                'Stock_Coverage_Days': 'mean'
            }).round(1).head(10)
            branch_metrics.columns = ['Total Stock', 'Avg Coverage (Days)']
            st.dataframe(branch_metrics, use_container_width=True)

    with tab4:
        d1, d2 = st.columns(2)
        with d1:
            category_stock = filtered_df.groupby('ItemGroup')['Current_Stock'].sum().reset_index()
            category_stock = category_stock.sort_values('Current_Stock', ascending=False).head(10)
            fig_category = px.bar(
                category_stock,
                x='ItemGroup', y='Current_Stock',
                title='Stock by Item Category'
            )
            fig_category.update_layout(xaxis_tickangle=-45, height=350)
            st.plotly_chart(fig_category, use_container_width=True)
        with d2:
            fig_scatter = px.scatter(
                filtered_df,
                x='Sales_60_Days',
                y='Current_Stock',
                color='Stock_Status',
                title='Sales vs Current Stock',
                hover_data=['ItemCode', 'Branch'],
                color_discrete_map={'Understock': '#e74c3c', 'Optimal': '#2ecc71', 'Overstock': '#f39c12'}
            )
            fig_scatter.update_layout(height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📥 Export Inventory Analysis")
    e1, e2 = st.columns(2)
    with e1:
        if st.button("📄 Export to CSV", use_container_width=True):
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"inventory_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    with e2:
        if st.button("🎯 Export Recommendations", use_container_width=True):
            recommendations_data = filtered_df[['ItemCode', 'ItemName', 'Branch', 'Zone', 'Stock_Status', 'Recommendations']]
            csv_data = recommendations_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Recommendations",
                data=csv_data,
                file_name=f"inventory_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ----------------------------
# Database Configuration Page
# ----------------------------
def render_database_config():
    """Database configuration page"""
    st.markdown('<div class="main-header"><h1>🔧 Database Configuration</h1></div>', unsafe_allow_html=True)

    db = get_db_instance()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Configuration")
        st.code(f"""
Host: {db.host}
Port: {db.port}
Username: {db.username}
Password: {"*" * len(db.password)}
        """)
        if st.button("🔍 Test Connection", type="primary"):
            with st.spinner("Testing..."):
                is_connected, status_msg = db.test_connection()
                if is_connected:
                    st.success(f"✅ {status_msg}")
                else:
                    st.error(f"❌ {status_msg}")

    with col2:
        st.subheader("Quick Diagnostics")
        if st.button("🔍 Test Table Access"):
            with st.spinner("Testing table access..."):
                try:
                    test_query = 'SELECT COUNT(*) as RECORD_COUNT FROM "AVA_Integration"."SALES_AI"'
                    result_df = db.execute_query(test_query)
                    if not result_df.empty:
                        record_count = result_df.iloc[0, 0]
                        st.success(f"✅ Table accessible - Records: {record_count:,}")
                    else:
                        st.error("❌ Table query returned no results")
                except Exception as e:
                    st.error(f"❌ Table access failed: {e}")

# ----------------------------
# Placeholder
# ----------------------------
def render_placeholder_page(page_name):
    st.markdown(f'<div class="main-header"><h1>🚧 {page_name}</h1></div>', unsafe_allow_html=True)
    st.info(f"The **{page_name}** page is under development. Coming soon!")

# ----------------------------
# Main
# ----------------------------
def main():
    try:
        current_page = render_navigation()
        if current_page == "Sales Report":
            render_sales_report()
        elif current_page == "Branch Trends":
            render_branch_trends()  # Add this line
        elif current_page == "Customer Analysis":
            render_customer_analysis()
        elif current_page == "Stock Analysis":
            render_stock_analysis()
        elif current_page == "Database Config":
            render_database_config()
        else:
            render_placeholder_page(current_page)

        st.markdown("---")
        st.markdown("*SAP B1 Analytics Dashboard | Powered by Streamlit*")
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or check the Database Config.")

if __name__ == "__main__":
    main()