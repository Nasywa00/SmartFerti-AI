import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

st.set_page_config(
    page_title="SmartFerti AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        body {
            background: radial-gradient(circle at top left, #f4ede0 0%, #d6b494 45%, #a98357 100%);
            color: #3f3122;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.92);
            padding: 18px 18px 36px 18px;
            border-radius: 12px;
        }
        .main-header {
            background: linear-gradient(135deg, #8d6b4b 0%, #b89976 100%);
            border-radius: 30px;
            padding: 36px;
            color: #f7efdf;
            box-shadow: 0 24px 80px rgba(63, 49, 34, 0.18);
            margin-bottom: 28px;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.94);
            border-radius: 24px;
            padding: 26px;
            box-shadow: 0 20px 80px rgba(63, 49, 34, 0.08);
            border: 1px solid rgba(141, 107, 75, 0.16);
            margin-bottom: 20px;
        }
        .metric-card {
            background: #fff8f0;
            border-radius: 22px;
            padding: 22px;
            border-left: 5px solid #8d6b4b;
            box-shadow: 0 14px 40px rgba(63, 49, 34, 0.06);
            margin-bottom: 18px;
        }
        .metric-title {
            color: #6c5233;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .metric-value {
            color: #322514;
            font-size: 2.4rem;
            font-weight: 800;
            margin-top: 10px;
        }
        .metric-unit {
            color: #7f6b57;
            font-size: 0.95rem;
            margin-top: 6px;
        }
        .fertilizer-card {
            background: linear-gradient(180deg, #fbf2e8 0%, #f3e3cf 100%);
            border-radius: 22px;
            padding: 22px;
            border: 1px solid rgba(141, 107, 75, 0.18);
            box-shadow: 0 14px 40px rgba(63, 49, 34, 0.06);
            margin-bottom: 18px;
        }
        .fertilizer-title {
            color: #533d29;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .fertilizer-value {
            color: #7f5b34;
            font-size: 2rem;
            font-weight: 800;
            margin-top: 12px;
        }
        .fertilizer-sub {
            color: #7b6655;
            font-size: 0.95rem;
            margin-top: 6px;
        }
        .info-box {
            background: rgba(139, 107, 75, 0.1);
            border-radius: 20px;
            padding: 20px;
            color: #4d3626;
            margin-bottom: 20px;
            border: 1px solid rgba(139, 107, 75, 0.22);
        }
        .success-box {
            background: #eef4e8;
            padding: 18px;
            border-radius: 18px;
            color: #3f593d;
            border-left: 5px solid #8d6b4b;
            font-size: 0.95rem;
            margin-top: 16px;
        }
        .stButton>button {
            border-radius: 14px;
            background: linear-gradient(135deg, #8d6b4b 0%, #b4997a 100%);
            color: #fff;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #916d4a 0%, #b89d83 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('model.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, encoders
    except Exception:
        return None, None

model, encoders = load_model_and_encoders()

if model is None or encoders is None:
    st.markdown(
        """
        <div class='section-card'>
            <h2>Model belum tersedia</h2>
            <p>Silakan jalankan <code>train.py</code> terlebih dahulu untuk membuat <code>model.pkl</code> dan <code>encoders.pkl</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


def encode_input(crop, soil_condition, season):
    try:
        crop_encoded = encoders['crop_encoder'].transform([crop])[0]
        soil_encoded = encoders['soil_encoder'].transform([soil_condition])[0]
        season_encoded = encoders['season_encoder'].transform([season])[0]
        return crop_encoded, soil_encoded, season_encoded
    except Exception as e:
        st.error(f"Error pada encoding: {e}")
        return None, None, None


def predict_npk(crop, soil_condition, season, age_days, land_area_ha):
    crop_enc, soil_enc, season_enc = encode_input(crop, soil_condition, season)
    if crop_enc is None:
        return None
    X = np.array([[crop_enc, soil_enc, season_enc, age_days, land_area_ha]])
    prediction = model.predict(X)[0]
    return {
        'N': float(prediction[0]),
        'P': float(prediction[1]),
        'K': float(prediction[2]),
    }


def convert_to_fertilizer(npk):
    return {
        'Urea': round(npk['N'] / 0.46, 2),
        'SP36': round(npk['P'] / 0.36, 2),
        'KCL': round(npk['K'] / 0.60, 2),
    }


def fertilizer_info():
    return {
        'Urea': {'content': '46% N', 'color': '#8d6b4b'},
        'SP36': {'content': '36% P2O5', 'color': '#b15f27'},
        'KCL': {'content': '60% K2O', 'color': '#9b4123'},
    }


def metric_card(title, value, unit):
    return f"""
    <div class='metric-card'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value'>{value}</div>
        <div class='metric-unit'>{unit}</div>
    </div>
    """


def fertilizer_card(title, amount, content):
    return f"""
    <div class='fertilizer-card'>
        <div class='fertilizer-title'>{title}</div>
        <div class='fertilizer-value'>{amount} kg/ha</div>
        <div class='fertilizer-sub'>{content}</div>
    </div>
    """

st.sidebar.title('🧑‍🌾 Input Parameter')
st.sidebar.write('Isi parameter lahan dan tanaman untuk rekomendasi pupuk presisi.')

crop = st.sidebar.selectbox('Jenis Tanaman', ['padi', 'jagung', 'cabai', 'tomat'])
soil_condition = st.sidebar.selectbox(
    'Kondisi Tanah',
    ['berat', 'sedang', 'ringan'],
    help='berat: tanah lempung/compact (drainase buruk); sedang: tekstur seimbang; ringan: berpasir (drainase baik)'
)
season = st.sidebar.selectbox('Musim', ['musim hujan', 'musim kemarau', 'musim transisi'])
age_days = st.sidebar.slider('Umur Tanaman (hari)', 30, 120, 60, 5)
land_area_ha = st.sidebar.number_input('Luas Lahan (hektar)', min_value=0.5, max_value=5.0, value=1.0, step=0.1, format='%.1f')

st.sidebar.markdown('---')
if st.sidebar.button('🔮 Prediksi Sekarang'):
    result = predict_npk(crop, soil_condition, season, age_days, land_area_ha)
    if result is not None:
        st.session_state.npk_result = result
        st.session_state.fertilizer_result = convert_to_fertilizer(result)
        st.session_state.input_data = {
            'crop': crop,
            'soil_condition': soil_condition,
            'season': season,
            'age_days': age_days,
            'land_area_ha': land_area_ha,
        }
        st.session_state.prediction_done = True

if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

st.markdown(
    """
    <div class='main-header'>
        <h1>SmartFerti AI</h1>
        <p>Dashboard agritech dengan rekomendasi pupuk presisi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.4, 1], gap='large')

with col1:
    if st.session_state.prediction_done:
        input_data = st.session_state.input_data
        npk_result = st.session_state.npk_result
        fertilizer_result = st.session_state.fertilizer_result

        st.markdown(
            f"""
            <div class='section-card'>
                <div class='info-box'>
                    <strong>Input terakhir:</strong> {input_data['crop'].upper()} | {input_data['soil_condition']} | {input_data['season']} | {input_data['age_days']} hari | {input_data['land_area_ha']} ha
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='section-card'>
                <h2>Hasil Prediksi Nutrisi</h2>
                <p>Kebutuhan N, P, dan K dalam satuan kg/ha.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics = [
            ('Nitrogen (N)', f"{npk_result['N']:.2f}", 'kg/ha'),
            ('Phosphorus (P)', f"{npk_result['P']:.2f}", 'kg/ha'),
            ('Potassium (K)', f"{npk_result['K']:.2f}", 'kg/ha'),
        ]
        metric_cols = st.columns(3, gap='large')
        for col, item in zip(metric_cols, metrics):
            col.markdown(metric_card(*item), unsafe_allow_html=True)

        st.markdown(
            """
            <div class='section-card'>
                <h2>Visualisasi NPK</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        npk_df = pd.DataFrame({
            'Nutrisi': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'],
            'Nilai': [npk_result['N'], npk_result['P'], npk_result['K']],
        })
        bar_chart = alt.Chart(npk_df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
            x=alt.X('Nutrisi:N', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Nilai:Q', title='kg/ha'),
            color=alt.Color('Nutrisi:N', scale=alt.Scale(range=['#8d6b4b', '#b15f27', '#9b4123']), legend=None),
            tooltip=[alt.Tooltip('Nutrisi:N'), alt.Tooltip('Nilai:Q', format='.2f')],
        ).properties(height=320)
        st.altair_chart(bar_chart, use_container_width=True)

        st.markdown(
            """
            <div class='section-card'>
                <h2>Rekomendasi Pupuk</h2>
                <p>Dosis rekomendasi Urea, SP36, dan KCL berdasarkan prediksi nutrisi.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fert_info = fertilizer_info()
        fert_cols = st.columns(3, gap='large')
        for col, key in zip(fert_cols, ['Urea', 'SP36', 'KCL']):
            props = fert_info[key]
            col.markdown(fertilizer_card(key, fertilizer_result[key], props['content']), unsafe_allow_html=True)

        st.markdown(
            """
            <div class='success-box'>
                ✅ Prediksi berhasil. Gunakan hasil ini sebagai referensi awal dan sesuaikan dengan kondisi lapangan.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class='section-card'>
                <h2>Selamat datang di SmartFerti AI</h2>
                <p>Gunakan panel input di kiri untuk mengisi parameter dan klik tombol prediksi untuk melihat hasil rekomendasi pupuk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col2:
    st.markdown(
        """
        <div class='section-card'>
            <h2>Kenapa SmartFerti?</h2>
            <ul>
                <li>Desain dashboard modern bergaya agritech</li>
                <li>Input sidebar yang rapi</li>
                <li>Output metric cards untuk NPK</li>
                <li>Rekomendasi pupuk yang mudah dibaca</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='section-card'>
            <h2>Langkah Cepat</h2>
            <ol>
                <li>Pilih data tanaman, tanah, dan musim.</li>
                <li>Isi umur tanaman dan luas lahan.</li>
                <li>Klik Prediksi Sekarang.</li>
                <li>Gunakan hasil sebagai panduan awal.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style='text-align:center; color:#7a6245; margin-top:24px; font-size:0.9rem;'>
        SmartFerti AI © 2026 | Dashboard agritech untuk pertanian presisi.
    </div>
    """,
    unsafe_allow_html=True,
)
