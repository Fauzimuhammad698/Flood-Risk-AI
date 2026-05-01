"""
Sistem Prediksi Risiko Banjir AI - Updated Version
Integrasi: BMKG, GEE, Open-Meteo, Hybrid Scorer
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- GEE IMPORT - WAJIB UNTUK REALTIME ---
try:
    import ee
    GEE_AVAILABLE = True
except ImportError as e:
    GEE_AVAILABLE = False
    st.error("❌ earthengine-api gagal diinstall. GEE wajib untuk mode real-time.")
    st.stop()
except Exception as e:
    GEE_AVAILABLE = False
    st.error(f"❌ GEE import error: {e}")
    st.stop()

# Try to import SHAP for explanation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not available. Install with: pip install shap")

# Page configuration
st.set_page_config(
    page_title="Sistem Prediksi Risiko Banjir AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.3);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .search-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
    }
    .risk-low { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); color: #065f46; }
    .risk-medium { background: linear-gradient(135deg, #fee140 0%, #fa709a 100%); color: #92400e; }
    .risk-high { background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%); color: white; }
    
    .metric-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.1); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1e293b; }
    .metric-label { font-size: 0.9rem; color: #64748b; margin-top: 0.5rem; }
    
    .analyze-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    .analyze-btn:hover { transform: scale(1.02); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
    
    .explanation-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Mobile Responsive Text Styles Only */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        .main-header p {
            font-size: 0.9rem !important;
        }
        .risk-badge {
            font-size: 1.2rem !important;
            padding: 0.5rem 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'current_time' not in st.session_state:
    st.session_state.current_time = datetime.now()

# --- GEE INITIALIZATION - WAJIB UNTUK REALTIME ---
GEE_ENABLED = False

if not GEE_AVAILABLE:
    st.error("❌ GEE library tidak tersedia. Mode real-time memerlukan GEE.")
    st.stop()

try:
    # Coba Service Account dari Streamlit Secrets
    if 'gee' in st.secrets:
        from oauth2client.service_account import ServiceAccountCredentials
        
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            {
                "type": "service_account",
                "project_id": "deteksi-banjir-492803",
                "private_key": st.secrets["gee"]["private_key"],
                "client_email": st.secrets["gee"]["service_account_email"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            },
            scopes=['https://www.googleapis.com/auth/earthengine.readonly']
        )
        ee.Initialize(credentials)
        GEE_ENABLED = True
        st.success("✅ GEE Real-time Connected via Service Account")
    else:
        # Local development - pakai default (hanya untuk local)
        ee.Initialize(project='deteksi-banjir-492803')
        GEE_ENABLED = True
        st.info("✅ GEE Real-time Connected (Local Mode)")
        
except Exception as e:
    st.error(f"❌ GEE initialization gagal: {e}")
    st.error("📋 Cek: 1) Service Account Key sudah diisi di Secrets, 2) Email sudah di-add ke GEE ACL")
    st.stop()  # STOP aplikasi jika GEE gagal

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open('final_production_flood_model.pkl', 'rb') as f:
            assets = pickle.load(f)
        return assets
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

assets = load_assets()
if assets is None:
    st.error("Failed to load model. Please check if 'final_production_flood_model.pkl' exists.")
    st.stop()

model = assets['model']
scaler = assets['scaler']
encoder = assets.get('encoder', None)
features = assets['features']

print(f"[OK] Model loaded. Features: {features}")

# --- HELPER FUNCTIONS ---

def format_waktu_indonesia(dt):
    """Format waktu ke bahasa Indonesia"""
    bulan_indonesia = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    hari_indonesia = {
        0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis',
        4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'
    }
    hari = hari_indonesia[dt.weekday()]
    bulan = bulan_indonesia[dt.month]
    return f"{hari}, {dt.day} {bulan} {dt.year} {dt.strftime('%H:%M')} WIB"


def get_coordinates(location_name):
    """Get coordinates from location name using Nominatim"""
    try:
        geolocator = Nominatim(user_agent="flood_risk_app_v2")
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None, None


def get_bmkg_nowcast_alert(lat, lon, location_name=None):
    """
    Get BMKG Nowcast Alert (Peringatan Dini) for a location
    Priority 2 in 3-tier fallback system
    
    Returns alert status and warning level if location is under active warning
    """
    try:
        print(f"[BMKG-Nowcast] Checking alerts for lat={lat}, lon={lon}")
        
        # BMKG Nowcast API endpoint
        url = "https://api.bmkg.go.id/publik/peringatan-dini-cuaca"
        
        headers = {
            'User-Agent': 'FloodRiskAI/1.0 (Research Purpose - Attribution to BMKG)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"[BMKG-Nowcast] HTTP Error: {response.status_code}")
            return None
        
        data = response.json()
        
        # Parse nowcast alerts
        if 'data' in data:
            alerts = data['data']
            
            # Check if location is mentioned in any active warning
            for alert in alerts:
                alert_areas = alert.get('areas', [])
                
                for area in alert_areas:
                    # Simple string matching for location name
                    if location_name and location_name.lower() in area.lower():
                        print(f"[BMKG-Nowcast] [ALERT] Location in warning area!")
                        
                        # Determine warning level and rainfall threshold
                        alert_type = alert.get('type', '')
                        warning_rainfall = 40.0  # Default warning threshold
                        
                        if 'Ekstrem' in alert_type or 'Lebat' in alert_type:
                            warning_rainfall = 100.0
                        elif 'Sedang' in alert_type:
                            warning_rainfall = 50.0
                        elif 'Ringan' in alert_type:
                            warning_rainfall = 25.0
                        
                        return {
                            'alert_active': True,
                            'alert_type': alert_type,
                            'warning_rainfall': warning_rainfall,
                            'description': alert.get('description', ''),
                            'area': area
                        }
            
            print(f"[BMKG-Nowcast] [OK] No active alerts")
            return {'alert_active': False}
        
        return None
            
    except Exception as e:
        print(f"[BMKG-Nowcast] Error: {e}")
        return None


def get_bmkg_weather(adm4_code=None, lat=None, lon=None):
    """
    Get weather data from BMKG API
    Supports both adm4_code-based and coordinate-based endpoints
    
    Priority:
    1. Use adm4_code if provided
    2. Fallback to coordinate-based nearest station
    3. Return None if both fail (trigger Open-Meteo fallback)
    
    Note: BMKG API endpoints may vary. This function tries multiple approaches.
    """
    try:
        # Approach 1: Try BMKG Cuaca endpoint (if available)
        # This is the public weather API endpoint
        
        if adm4_code:
            # Using adm4 code (administrative area level 4)
            url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4_code}"
            print(f"[BMKG] Trying adm4_code endpoint: {adm4_code}")
        elif lat is not None and lon is not None:
            # Using coordinates - try different endpoint formats
            # Format 1: Standard lat/lon
            url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?lat={lat}&lon={lon}"
            print(f"[BMKG] Trying coordinate endpoint: {lat:.4f}, {lon:.4f}")
        else:
            print("[BMKG] No adm4_code or coordinates provided")
            return None
        
        response = requests.get(url, timeout=10)
        print(f"[BMKG] Response status: {response.status_code}")
        
        # If 404, try alternative endpoint
        if response.status_code == 404:
            print(f"[BMKG] Primary endpoint 404, trying alternative...")
            # Try alternative BMKG endpoint (maritim or cuaca berbasis wilayah)
            # This is a placeholder - actual BMKG endpoints may differ
            alt_url = f"https://maritim.bmkg.go.id/ajax/curah-hujan?lat={lat}&lon={lon}&format=json"
            response = requests.get(alt_url, timeout=10)
            print(f"[BMKG] Alternative endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Parse BMKG API response structure
            if 'data' in data and len(data['data']) > 0:
                weather_data = data['data'][0]
                
                # Extract rainfall data from various possible fields
                rainfall_mm = 0.0
                
                # Try different field names based on BMKG API structure
                if 'precipitation' in weather_data:
                    rainfall_mm = float(weather_data['precipitation'])
                elif 'rainfall' in weather_data:
                    rainfall_mm = float(weather_data['rainfall'])
                elif 'curah_hujan' in weather_data:
                    rainfall_mm = float(weather_data['curah_hujan'])
                elif 'weather' in weather_data and 'precipitation' in weather_data['weather']:
                    rainfall_mm = float(weather_data['weather']['precipitation'])
                elif 'cuaca' in weather_data:
                    cuaca_data = weather_data['cuaca']
                    if isinstance(cuaca_data, list) and len(cuaca_data) > 0:
                        rainfall_mm = float(cuaca_data[0].get('precipitation', 0))
                
                # Also try to get forecast data for accumulation
                rainfall_3d = rainfall_mm * 2.5  # Estimate if not available
                rainfall_7d = rainfall_mm * 5    # Estimate if not available
                
                print(f"[BMKG] [OK] Success - Rainfall: {rainfall_mm:.1f}mm")
                
                return {
                    'rainfall_curr': rainfall_mm,
                    'rainfall_3d': rainfall_3d,
                    'rainfall_7d': rainfall_7d,
                    'source': 'BMKG'
                }
            else:
                print(f"[BMKG] Response OK but no data field: {list(data.keys())}")
        else:
            print(f"[BMKG] HTTP Error: {response.status_code}")
            print(f"[BMKG] Response: {response.text[:100]}")
        
        return None
        
    except requests.Timeout:
        print(f"[BMKG] [TIMEOUT] API took too long to respond")
        return None
    except requests.RequestException as e:
        print(f"[BMKG] [NETWORK] Error: {e}")
        return None
    except Exception as e:
        print(f"[BMKG] [ERROR] {e}")
        return None


def get_openmeteo_weather(lat, lon):
    """
    Get weather data from Open-Meteo as primary or fallback
    Returns: rainfall_curr, rainfall_3d, rainfall_7d
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=rain_sum&timezone=auto&forecast_days=7"
        response = requests.get(url, timeout=10)
        res = response.json()
        
        current = res.get('current_weather', {})
        daily = res.get('daily', {})
        rain_sum_list = daily.get('rain_sum', [])
        
        # Calculate accumulations
        rain_sum_7d = sum(rain_sum_list[:7]) if len(rain_sum_list) >= 7 else sum(rain_sum_list)
        rain_sum_3d = sum(rain_sum_list[:3]) if len(rain_sum_list) >= 3 else sum(rain_sum_list)
        
        # Convert weather code to rainfall estimate
        weather_code = current.get('weathercode', 0)
        weather_code_mapping = {
            0: 0, 1: 1, 2: 2, 3: 5, 45: 10, 48: 15,
            51: 5, 53: 10, 55: 15, 56: 10, 57: 20,
            61: 10, 63: 20, 65: 35, 66: 25, 67: 40,
            71: 5, 73: 15, 75: 30, 77: 20,
            80: 20, 81: 35, 82: 50, 85: 30, 86: 45,
            95: 40, 96: 60, 99: 80
        }
        rainfall_curr = weather_code_mapping.get(weather_code, 10)
        
        return {
            'rainfall_curr': rainfall_curr,
            'rainfall_3d': rain_sum_3d,
            'rainfall_7d': rain_sum_7d,
            'source': 'Open-Meteo'
        }
    except Exception as e:
        print(f"Open-Meteo API error: {e}")
        return {
            'rainfall_curr': 0,
            'rainfall_3d': 0,
            'rainfall_7d': 0,
            'source': 'Error'
        }


def get_gee_data(lat, lon):
    """
    Get spatial data from Google Earth Engine - REALTIME ONLY
    Raises exception if failed (no fallback)
    """
    if not GEE_ENABLED:
        raise Exception("GEE tidak terinisialisasi. Real-time mode memerlukan GEE.")
    
    point = ee.Geometry.Point([lon, lat])
    
    # 1. SLOPE from SRTM
    srtm = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(srtm)
    slope_value = slope.sample(point, 30).first().get('slope').getInfo()
    
    if slope_value is None:
        raise Exception("GEE: Gagal mengambil data slope")
    
    # 2. NDVI from Sentinel-2
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()
    
    ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi_value = ndvi.sample(point, 30).first().get('NDVI').getInfo()
    
    if ndvi_value is None:
        raise Exception("GEE: Gagal mengambil data NDVI")
    
    # 3. LAND COVER from ESA WorldCover
    worldcover = ee.Image('ESA/WorldCover/v200/2021')
    land_cover_code = worldcover.sample(point, 10).first().get('Map').getInfo()
    
    if land_cover_code is None:
        raise Exception("GEE: Gagal mengambil data Land Cover")
    
    land_cover_mapping = {
        10: 'Forest', 20: 'Forest', 30: 'Agri', 40: 'Agri',
        50: 'Urban', 60: 'Barren', 70: 'Snow', 80: 'Water',
        90: 'Wetland', 95: 'Mangrove', 100: 'Moss'
    }
    land_cover = land_cover_mapping.get(land_cover_code, 'Urban')
    
    return {
        'slope': float(slope_value),
        'ndvi': float(ndvi_value),
        'land_cover': land_cover,
        'source': 'GEE-REALTIME'
    }


def fetch_comprehensive_data(lat, lon, adm4_code=None, location_name=None):
    """
    Fetch all data from multiple sources with 3-tier fallback:
    Priority 1: BMKG Forecast API (via adm4_code)
    Priority 2: BMKG Nowcast Alert (if location in active warning)
    Priority 3: Open-Meteo Archive/Forecast API
    """
    rainfall_data = None
    
    # PRIORITY 1: BMKG Forecast API
    print(f"[FETCH] Priority 1: Trying BMKG Forecast API for lat={lat}, lon={lon}")
    if adm4_code:
        bmkg_data = get_bmkg_weather(adm4_code=adm4_code)
        if bmkg_data and bmkg_data.get('rainfall_curr') is not None:
            rainfall_data = {
                'rainfall_curr': bmkg_data['rainfall_curr'],
                'rainfall_3d': bmkg_data['rainfall_3d'],
                'rainfall_7d': bmkg_data['rainfall_7d'],
                'weather_desc': bmkg_data.get('weather_desc', ''),
                'weather_source': 'BMKG-Forecast'
            }
            print(f"[FETCH] [OK] Priority 1 SUCCESS: BMKG Forecast - {rainfall_data['rainfall_curr']:.1f}mm")
    else:
        print("[FETCH] [WARN] Priority 1: No adm4_code provided, skipping BMKG Forecast")
    
    # PRIORITY 2: BMKG Nowcast Alert (if Priority 1 failed)
    if rainfall_data is None:
        print(f"[FETCH] Priority 2: Trying BMKG Nowcast Alert for location: {location_name}")
        nowcast_data = get_bmkg_nowcast_alert(lat, lon, location_name)
        if nowcast_data and nowcast_data.get('alert_active'):
            # Location is under warning - set warning threshold
            rainfall_data = {
                'rainfall_curr': nowcast_data.get('warning_rainfall', 40.0),  # Warning threshold
                'rainfall_3d': nowcast_data.get('warning_rainfall', 40.0) * 2.5,
                'rainfall_7d': nowcast_data.get('warning_rainfall', 40.0) * 5,
                'weather_desc': nowcast_data.get('alert_type', 'Peringatan Dini'),
                'weather_source': 'BMKG-Nowcast'
            }
            print(f"[FETCH] [OK] Priority 2 SUCCESS: BMKG Nowcast Alert - {rainfall_data['rainfall_curr']:.1f}mm")
        else:
            print(f"[FETCH] [WARN] Priority 2: No active nowcast alert for this location")
    
    # PRIORITY 3: Open-Meteo (final fallback)
    if rainfall_data is None:
        print(f"[FETCH] Priority 3: Using Open-Meteo as final fallback")
        rainfall_data = get_openmeteo_weather(lat, lon)
        rainfall_data['weather_source'] = 'Open-Meteo'
        print(f"[FETCH] [OK] Priority 3: Open-Meteo - {rainfall_data['rainfall_curr']:.1f}mm")
    
    print(f"[FETCH] Final weather source: {rainfall_data['weather_source']}")
    
    # Get GEE data
    gee_data = get_gee_data(lat, lon)
    
    # Get elevation
    try:
        el_url = f'https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}'
        response = requests.get(el_url, timeout=10)
        elevation = response.json()['results'][0]['elevation']
    except:
        elevation = 50.0
    
    # Combine all data
    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'rainfall_curr': rainfall_data['rainfall_curr'],
        'rainfall_3d': rainfall_data['rainfall_3d'],
        'rainfall_7d': rainfall_data['rainfall_7d'],
        'slope': gee_data['slope'],
        'ndvi': gee_data['ndvi'],
        'land_cover': gee_data['land_cover'],
        'elevation': elevation,
        'weather_source': rainfall_data['weather_source'],
        'spatial_source': gee_data['source']
    }
    
    return result


def preprocess_input(raw_data):
    """
    Preprocess input data for model prediction
    Creates: is_rainy_season, land_cover_enc, and ensures exact feature order
    """
    df = pd.DataFrame([raw_data])
    
    print(f"[DEBUG] Input features: {list(df.columns)}")
    
    # 1. Create is_rainy_season feature (Oct-Apr = 1, else 0)
    current_month = datetime.now().month
    df['is_rainy_season'] = 1 if current_month in [10, 11, 12, 1, 2, 3, 4] else 0
    print(f"[DEBUG] Month: {current_month}, is_rainy_season: {df['is_rainy_season'].iloc[0]}")
    
    # 2. Encode land_cover
    if 'land_cover' in df.columns and encoder is not None:
        try:
            df['land_cover_enc'] = encoder.transform([df['land_cover'].iloc[0]])[0]
        except ValueError:
            # Unknown category, use default
            df['land_cover_enc'] = 0
    else:
        df['land_cover_enc'] = 0
    
    # 3. Get expected features from scaler
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = list(scaler.feature_names_in_)
    else:
        expected_features = ['rainfall_curr', 'rainfall_3d', 'rainfall_7d', 'slope', 'ndvi', 'land_cover_enc', 'is_rainy_season']
    
    print(f"[DEBUG] Expected features: {expected_features}")
    
    # 4. Ensure all expected features exist
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
            print(f"[DEBUG] Added missing feature '{col}' with value 0")
    
    # 5. Reindex to exact feature order
    X = df.reindex(columns=expected_features, fill_value=0)
    
    print(f"[DEBUG] Final feature order: {list(X.columns)}")
    print(f"[DEBUG] Feature values: {X.iloc[0].to_dict()}")
    
    # 6. Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, X, df


def hybrid_scorer(raw_data, model_prob):
    """
    Hybrid Scorer: 80% Physical + 20% Historical
    Anti-paranoid approach - requires multiple factors for high risk
    """
    from datetime import datetime
    
    # Get current month for rainy season check
    current_month = datetime.now().month
    
    # 80% Physical Score Component
    rainfall_score = min(raw_data['rainfall_curr'] / 100, 1.0) * 0.3 + \
                     min(raw_data['rainfall_3d'] / 300, 1.0) * 0.25 + \
                     min(raw_data['rainfall_7d'] / 500, 1.0) * 0.25
    
    vulnerability_score = (1 - min(raw_data['slope'] / 30, 1.0)) * 0.1 + \
                          (1 - raw_data['ndvi']) * 0.1
    
    # Land cover vulnerability
    land_vuln = {'Urban': 0.8, 'Barren': 0.7, 'Agri': 0.4, 'Forest': 0.2, 'Water': 0.5}
    land_factor = land_vuln.get(raw_data['land_cover'], 0.5)
    
    physical_score = (rainfall_score + vulnerability_score + land_factor * 0.1) / 0.8
    physical_score = min(physical_score, 1.0)
    
    # 20% Model Score Component
    model_score = model_prob
    
    # Combined Score (80-20 weighting)
    final_prob = (physical_score * 0.8) + (model_score * 0.2)
    
    # Anti-paranoid adjustment: Require multiple factors for high risk
    # Musim hujan Indonesia: Oktober-April (10,11,12,1,2,3,4)
    is_rainy_season = raw_data.get('is_rainy_season', current_month in [10, 11, 12, 1, 2, 3, 4])
    
    risk_factors = sum([
        raw_data['rainfall_curr'] > 50,
        raw_data['rainfall_3d'] > 150,
        raw_data['slope'] < 10,
        raw_data['ndvi'] < 0.3,
        raw_data['land_cover'] in ['Urban', 'Barren'],
        is_rainy_season
    ])
    
    # If only 1-2 risk factors, cap the probability
    if risk_factors <= 2:
        final_prob = min(final_prob, 0.4)  # Cap at medium risk
    elif risk_factors <= 4:
        final_prob = min(final_prob, 0.7)  # Cap at high risk threshold
    
    return min(final_prob, 1.0)


def get_risk_level(prob):
    """Convert probability to risk level"""
    if prob < 0.3:
        return "Rendah", "low", "✅", "#10b981", "green"
    elif prob < 0.7:
        return "Sedang", "medium", "⚠️", "#f59e0b", "orange"
    else:
        return "Tinggi", "high", "🚨", "#ef4444", "red"


def generate_ai_explanation(raw_data, risk_level, top_features):
    """Generate AI explanation for the risk assessment"""
    explanations = []
    
    # Rainfall analysis
    if raw_data['rainfall_curr'] > 80:
        explanations.append("🌧️ Curah hujan ekstrem saat ini membebani sistem drainase")
    elif raw_data['rainfall_curr'] > 50:
        explanations.append("🌧️ Curah hujan tinggi, perlu waspada genangan")
    
    if raw_data['rainfall_7d'] > 300:
        explanations.append("📊 Akumulasi hujan 7 hari tinggi - tanah jenuh")
    elif raw_data['rainfall_3d'] > 150:
        explanations.append("📊 Hujan 3 hari berturut-turut meningkatkan risiko")
    
    # Terrain analysis
    if raw_data['slope'] < 5:
        explanations.append("⛰️ Lereng datar - air mudah menggenang")
    elif raw_data['slope'] > 20:
        explanations.append("⛰️ Lereng curam - aliran air cepat")
    
    # Vegetation analysis
    if raw_data['ndvi'] < 0.2:
        explanations.append("🌿 Vegetasi jarang - permeabilitas tanah rendah")
    elif raw_data['ndvi'] > 0.6:
        explanations.append("🌿 Vegetasi lebat - membantu absorbsi air")
    
    # Land cover
    if raw_data['land_cover'] == 'Urban':
        explanations.append("🏙️ Kawasan urban - permukaan tidak tembus air")
    elif raw_data['land_cover'] == 'Forest':
        explanations.append("🌲 Kawasan hutan - risiko genangan rendah")
    
    # Seasonal
    current_month = datetime.now().month
    if current_month in [10, 11, 12, 1, 2, 3, 4]:
        explanations.append("📅 Musim hujan - potensi banjir meningkat")
    
    if not explanations:
        if risk_level == "Rendah":
            explanations.append("✅ Semua faktor kondisi aman")
        else:
            explanations.append("⚠️ Kombinasi faktor meningkatkan risiko")
    
    return explanations


def create_folium_map(lat, lon, address, risk_level, risk_color):
    """
    Create interactive Folium map with Google Maps-like UI
    Features: Multiple tile layers, fullscreen, layer control
    """
    from folium.plugins import Fullscreen
    
    # Create base map with OpenStreetMap (default - clear and detailed)
    m = folium.Map(
        location=[lat, lon],
        zoom_start=14,  # Closer zoom for better detail
        tiles='OpenStreetMap',
        control_scale=True,  # Show scale bar
        prefer_canvas=True   # Better performance
    )
    
    # Add multiple tile layers (like Google Maps options)
    # 1. CartoDB Positron (clean, light)
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Peta Bersih',
        attr='©OpenStreetMap, ©CartoDB'
    ).add_to(m)
    
    # 2. CartoDB Dark Matter (dark mode)
    folium.TileLayer(
        tiles='CartoDB dark_matter',
        name='Mode Gelap',
        attr='©OpenStreetMap, ©CartoDB'
    ).add_to(m)
    
    # 3. OpenTopoMap (topographic)
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        name='Terrain',
        attr='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)'
    ).add_to(m)
    
    # 4. Satellite-like tiles (Esri World Imagery)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        name='Satelit (Esri)',
        attr='Esri',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add Layer Control (switch between map types)
    folium.LayerControl(position='topright', collapsed=True).add_to(m)
    
    # Add Fullscreen button
    Fullscreen(position='topleft', title='Layar Penuh', title_cancel='Keluar').add_to(m)
    
    # Add custom styled popup with better UI
    popup_html = f"""
    <div style="font-family: 'Inter', sans-serif; padding: 10px; min-width: 200px;">
        <h4 style="margin: 0 0 8px 0; color: #1e293b; font-size: 16px;">📍 {address[:50]}</h4>
        <div style="background: {'#fee2e2' if risk_color=='red' else '#fef3c7' if risk_color=='orange' else '#d1fae5'}; 
                    padding: 8px 12px; border-radius: 8px; margin: 8px 0;">
            <strong style="color: {'#dc2626' if risk_color=='red' else '#d97706' if risk_color=='orange' else '#059669'};">
                🌊 Risiko {risk_level}
            </strong>
        </div>
        <p style="margin: 5px 0; font-size: 12px; color: #64748b;">
            Koordinat: {lat:.4f}°, {lon:.4f}°
        </p>
    </div>
    """
    
    # Add marker with custom icon
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=300, min_width=200),
        tooltip=f"📍 {address[:40]}...",
        icon=folium.Icon(
            color=risk_color,
            icon='warning' if risk_level == "Tinggi" else 'info-sign',
            prefix='glyphicon',
            icon_color='white'
        )
    ).add_to(m)
    
    # Add multiple risk zones with different opacities
    # Inner circle (strong risk)
    folium.Circle(
        location=[lat, lon],
        radius=1000,  # 1km
        popup='Zona Risiko Tinggi',
        color=risk_color,
        fill=True,
        fill_opacity=0.25,
        weight=3
    ).add_to(m)
    
    # Outer circle (moderate risk)
    folium.Circle(
        location=[lat, lon],
        radius=2000,  # 2km
        popup='Zona Risiko Sedang',
        color=risk_color,
        fill=True,
        fill_opacity=0.1,
        weight=2,
        dash_array='5, 10'  # Dashed line
    ).add_to(m)
    
    # Add mini map (overview) in corner
    from folium.plugins import MiniMap
    minimap = MiniMap(
        tile_layer='CartoDB positron',
        position='bottomright',
        width=150,
        height=150,
        collapsed_width=25,
        collapsed_height=25,
        zoom_level_offset=-5
    )
    m.add_child(minimap)
    
    return m


# --- MAIN UI ---

# Header with current time
current_time_str = format_waktu_indonesia(datetime.now())

st.markdown(f"""
<div style="text-align: center; background: linear-gradient(90deg, #667eea, #764ba2); 
            padding: 10px 20px; border-radius: 25px; margin-bottom: 15px;">
    <span style="color: white; font-size: 1rem; font-weight: 600;">🕐 {current_time_str}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌊 Sistem Prediksi Risiko Banjir AI</h1>
    <p>Integrasi BMKG, GEE & Hybrid Scorer untuk Prediksi Real-time</p>
</div>
""", unsafe_allow_html=True)

# Model Info Banner
st.markdown("""
<div style="background: linear-gradient(90deg, #10b981, #059669); 
            padding: 12px 20px; border-radius: 12px; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
        <span style="font-size: 1.5rem;">🧠</span>
        <div style="text-align: center;">
            <span style="color: white; font-weight: 700; font-size: 1rem;">Model Hybrid</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# GEE Status
if GEE_ENABLED:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #3b82f6, #1d4ed8); 
                padding: 8px 15px; border-radius: 8px; margin-bottom: 15px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 1.2rem;">🛰️</span>
            <span style="color: white; font-weight: 600; font-size: 0.9rem;">Google Earth Engine: TERHUBUNG (Slope & NDVI Real-time)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ GEE tidak terhubung - menggunakan data default")

# Search Section
st.markdown("<div class='search-card'>", unsafe_allow_html=True)
st.markdown("### 📍 Cari Lokasi")

col1, col2 = st.columns([3, 1])
with col1:
    location_input = st.text_input(
        "Masukkan nama kota/daerah:",
        placeholder="Contoh: Jakarta, Bandung, Surabaya, Yogyakarta...",
        label_visibility="collapsed"
    )
with col2:
    analyze_clicked = st.button("🔍 Analisis Risiko", use_container_width=True, type="primary")

st.markdown("</div>", unsafe_allow_html=True)

# Analysis Results
if analyze_clicked and location_input:
    with st.spinner('🔍 Mencari koordinat...'):
        lat, lon, address = get_coordinates(location_input)
    
    if lat and lon:
        # Interactive Loading UI
        loading_container = st.empty()
        
        loading_html = """
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 1rem 0;">
            <div style="font-size: 3rem; animation: pulse 1.5s infinite;">🌊</div>
            <h3 style="color: white; margin: 1rem 0;">Menganalisis Risiko Banjir...</h3>
            <div style="width: 80%; height: 8px; background: rgba(255,255,255,0.3); border-radius: 10px; margin: 1rem auto; overflow: hidden;">
                <div id="loading-bar" style="width: 0%; height: 100%; background: white; border-radius: 10px; transition: width 0.3s ease;"></div>
            </div>
            <p id="loading-status" style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-top: 0.5rem;">🛰️ Mengambil data spasial...</p>
        </div>
        <style>
            @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
        </style>
        """
        loading_container.markdown(loading_html, unsafe_allow_html=True)
        
        # Simulate animated steps
        steps = [
            ("🛰️ Mengambil data GEE (Slope & NDVI)...", 0.2),
            ("🌧️ Mengambil data cuaca BMKG/Open-Meteo...", 0.4),
            ("🤖 Memproses model AI & Hybrid Scorer...", 0.6),
            ("📊 Analisis risiko & generate peta...", 0.8),
            ("✅ Finalisasi hasil prediksi...", 1.0)
        ]
        
        for status, progress in steps:
            loading_container.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 1rem 0;">
                <div style="font-size: 3rem; animation: pulse 1.5s infinite;">🌊</div>
                <h3 style="color: white; margin: 1rem 0;">Menganalisis Risiko Banjir...</h3>
                <div style="width: 80%; height: 8px; background: rgba(255,255,255,0.3); border-radius: 10px; margin: 1rem auto; overflow: hidden;">
                    <div style="width: {progress*100}%; height: 100%; background: white; border-radius: 10px; transition: width 0.5s ease;"></div>
                </div>
                <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-top: 0.5rem;">{status}</p>
            </div>
            <style>@keyframes pulse {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.1); }} }}</style>
            """, unsafe_allow_html=True)
            time.sleep(0.4)
        
        loading_container.empty()
        
        # Fetch data
        raw_data = fetch_comprehensive_data(lat, lon)
        raw_data['lat'] = lat
        raw_data['lon'] = lon
        
        # Preprocess
        X_scaled, X_raw, df_processed = preprocess_input(raw_data)
        
        # Get model probability
        model_prob = float(model.predict_proba(X_scaled)[0][1])
        
        # SHAP Explanation (if available)
        shap_values = None
        if SHAP_AVAILABLE and hasattr(model, 'predict_proba'):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
                print(f"[SHAP] [OK] Explanation computed successfully")
            except Exception as e:
                print(f"[SHAP] [WARN] Error computing SHAP: {e}")
        
        # Apply hybrid scorer
        final_prob = hybrid_scorer(raw_data, model_prob)
        
        # Get risk level
        risk_level, risk_class, emoji, color, map_color = get_risk_level(final_prob)
        
        # --- RESULT DISPLAY ---
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        
        # Risk Badge
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">{emoji}</div>
            <div class="risk-badge risk-{risk_class}">Risiko {risk_level}</div>
            <div style="font-size: 1.8rem; font-weight: 600; color: {color}; margin-top: 1rem;">
                Probabilitas: {final_prob:.1%}
            </div>
            <div style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
                Prediksi diambil dari 100 hybrid data real
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.markdown("<h4>Tingkat Keyakinan:</h4>", unsafe_allow_html=True)
        st.progress(final_prob)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- METRICS GRID ---
        st.markdown("<h3 style='margin-top: 2rem;'>📊 Data Lingkungan Real-time</h3>", unsafe_allow_html=True)
        
        cols = st.columns(4)
        metrics = [
            ("🌧️ Curah Hujan", f"{raw_data['rainfall_curr']:.1f} mm", f"Sumber: {raw_data['weather_source']}"),
            ("📊 Hujan 7 Hari", f"{raw_data['rainfall_7d']:.1f} mm", "Akumulasi"),
            ("⛰️ Kemiringan", f"{raw_data['slope']:.1f}°", f"Sumber: {raw_data['spatial_source']}"),
            ("🌿 NDVI", f"{raw_data['ndvi']:.2f}", "Vegetasi")
        ]
        
        for col, (label, value, help_text) in zip(cols, metrics):
            with col:
                st.metric(label, value, help=help_text)
        
        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🗺️ Peta Risiko", "🧠 Analisis AI", "📋 Data Lengkap"])
        
        with tab1:
            m = create_folium_map(lat, lon, address, risk_level, map_color)
            st_folium(m, width=1200, height=700, returned_objects=[])
        
        with tab2:
            st.markdown("<div class='explanation-card'>", unsafe_allow_html=True)
            st.markdown("### 🤖 AI Analysis Summary")
            
            explanations = generate_ai_explanation(raw_data, risk_level, None)
            for exp in explanations:
                st.markdown(f"- {exp}")
            
            # SHAP Explanation (if available)
            st.markdown("\n#### 📈 Top Faktor Risiko (AI-powered):")
            
            if shap_values is not None and SHAP_AVAILABLE:
                try:
                    # Create SHAP force plot data
                    feature_names = assets['features'] if 'features' in assets else ['rainfall_curr', 'rainfall_3d', 'rainfall_7d', 'slope', 'ndvi', 'land_cover_enc', 'is_rainy_season']
                    
                    # Get SHAP values for the positive class (flood risk)
                    if isinstance(shap_values, list):
                        shap_vals = shap_values[1][0]  # For binary classification, index 1 is positive class
                    else:
                        shap_vals = shap_values[0]
                    
                    # Sort features by absolute SHAP value
                    feature_importance = list(zip(feature_names, shap_vals))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Display top 5 most important features
                    top_features = feature_importance[:5]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    names, values = zip(*top_features)
                    colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]
                    
                    ax.barh(names, values, color=colors)
                    ax.set_xlabel('SHAP Value (Impact on Prediction)')
                    ax.set_title('Faktor yang Mempengaruhi Risiko Banjir (Model AI)')
                    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show detailed SHAP values
                    st.markdown("\n**Detail Kontribusi Fitur:**")
                    for name, value in top_features:
                        direction = "↑ Meningkatkan" if value > 0 else "↓ Menurunkan"
                        st.markdown(f"- **{name}**: {direction} risiko (score: {value:.3f})")
                    
                except Exception as e:
                    st.warning(f"⚠️ SHAP visualization error: {e}")
                    # Fallback to simplified explanation
                    show_simplified_explanation(raw_data, st)
            else:
                # Fallback to simplified explanation if SHAP not available
                show_simplified_explanation(raw_data, st)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='explanation-card'>", unsafe_allow_html=True)
            st.markdown(f"📍 **{address}**")
            st.markdown(f"🌐 Koordinat: {lat:.6f}°, {lon:.6f}°")
            st.markdown(f"📅 Waktu Analisis: {current_time_str}")
            
            st.markdown("\n**Detail Data:**")
            data_table = {
                "Parameter": ["Curah Hujan", "Hujan 3 Hari", "Hujan 7 Hari", "Kemiringan", "NDVI", "Tutupan Lahan", "Musim Hujan", "Elevasi"],
                "Nilai": [f"{raw_data['rainfall_curr']:.1f} mm", f"{raw_data['rainfall_3d']:.1f} mm", f"{raw_data['rainfall_7d']:.1f} mm", 
                         f"{raw_data['slope']:.1f}°", f"{raw_data['ndvi']:.2f}", raw_data['land_cover'], 
                         "Ya" if datetime.now().month in [10, 11, 12, 1, 2, 3, 4] else "Tidak", f"{raw_data['elevation']:.0f} m"],
                "Sumber": [raw_data['weather_source'], raw_data['weather_source'], raw_data['weather_source'], 
                          raw_data['spatial_source'], raw_data['spatial_source'], raw_data['spatial_source'], "Sistem", "Open-Elevation"]
            }
            st.table(pd.DataFrame(data_table))
            st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        st.error("❌ Lokasi tidak ditemukan. Coba masukkan nama kota yang lebih spesifik.")
        st.info("💡 Tips: Gunakan 'Jakarta, Indonesia' atau 'Bandung, Jawa Barat'")

# Footer
st.markdown("""
<div class="footer">
    <p>🌊 Flood Risk AI v2.0 | Integrasi BMKG, GEE, Open-Meteo & Hybrid Scorer</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">Data cuaca real-time dari BMKG/Open-Meteo | Data spasial dari Google Earth Engine</p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem; color: #94a3b8;">Dibuat oleh Fauzi Muhammad</p>
</div>
""", unsafe_allow_html=True)
