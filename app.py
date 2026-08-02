import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

st.set_page_config(
    page_title="Eldorado SPK Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e8eaf0;
    font-size: 17px;
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
}
/* Teks isi dinaikkan agar tetap terbaca saat dipresentasikan lewat proyektor */
.stMarkdown, .stMarkdown p, .stMarkdown li { font-size: 1.02rem; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2a3550;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.main-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
}
.main-header p {
    color: #a0aab8;
    font-size: 1.12rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #1a1f2e;
    border: 1px solid #2a3550;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
}
.metric-label {
    font-size: 0.92rem;
    font-weight: 700;
    color: #6b7590;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-sub {
    font-size: 0.95rem;
    color: #6b7590;
    margin-top: 0.3rem;
    line-height: 1.4;
}
.metric-desc {
    font-size: 0.92rem;
    color: #9aa4b8;
    margin-top: 0.5rem;
    line-height: 1.5;
    border-top: 1px solid #2a3550;
    padding-top: 0.5rem;
}
.metric-accent-purple .metric-value { color: #a5b4fc; }
.metric-accent-green .metric-value  { color: #6ee7b7; }
.metric-accent-red .metric-value    { color: #fca5a5; }
.metric-accent-yellow .metric-value { color: #fde68a; }
/* Warna klaster, disamakan dengan Gambar 4.2 pada naskah */
.metric-accent-c0 .metric-value { color: #60a5fa; }  /* Dead Stock  - biru  */
.metric-accent-c1 .metric-value { color: #f87171; }  /* Low Tier    - merah */
.metric-accent-c2 .metric-value { color: #4ade80; }  /* Anomali     - hijau */
.metric-accent-c3 .metric-value { color: #c084fc; }  /* Star        - ungu  */

/* Section */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.42rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 0.5rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #2a3550;
}
.section-desc {
    font-size: 1.0rem;
    color: #9aa4b8;
    margin-bottom: 1rem;
    line-height: 1.6;
}

/* Info/alert boxes */
.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px;
    padding: 1.1rem 1.35rem;
    font-size: 1.0rem;
    color: #a5b4fc;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}
.success-box {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.9rem;
    color: #6ee7b7;
    margin-bottom: 1rem;
    line-height: 1.6;
}
.warning-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.9rem;
    color: #fde68a;
    margin-bottom: 1rem;
    line-height: 1.6;
}

/* SPK rule cards */
.spk-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.spk-card {
    background: #1a1f2e;
    border-radius: 12px;
    padding: 1.25rem;
}
.spk-card-restock { border: 2px solid rgba(239,68,68,0.4); }
.spk-card-nonrestock   { border: 2px solid rgba(107,114,128,0.4); }
.spk-card-aman    { border: 2px solid rgba(52,211,153,0.4); }
.spk-title {
    font-size: 0.98rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}
.spk-title-restock { color: #f87171; }
.spk-title-nonrestock   { color: #9ca3af; }
.spk-title-aman    { color: #6ee7b7; }
.spk-body {
    font-size: 0.98rem;
    color: #ccd4e4;
    line-height: 1.6;
}

/* Cluster legend */
.cluster-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.cluster-card {
    background: #1a1f2e;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.cluster-name {
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.cluster-desc {
    font-size: 0.96rem;
    color: #9aa4b8;
    line-height: 1.6;
}

/* Chart caption */
.chart-caption {
    font-size: 0.95rem;
    color: #8892a4;
    text-align: center;
    margin-top: 0.25rem;
    margin-bottom: 1rem;
    font-style: italic;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161f !important;
    border-right: 1px solid #2a3550;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    color: #8892a4;
    font-weight: 600;
    font-size: 1.08rem;
    padding: 0.9rem 1.5rem;
}
.stTabs [aria-selected="true"] { color: #a5b4fc !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #6366f1 !important; }

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Streamlit default font size override */
.stMarkdown p { font-size: 1.02rem; line-height: 1.75; }
label { font-size: 1.0rem !important; font-weight: 600 !important; }
/* Tabel data: perbesar agar terbaca saat demo */
div[data-testid="stDataFrame"] * { font-size: 0.98rem !important; }
/* Panel penjelasan istilah */
.stExpander summary { font-size: 1.02rem !important; font-weight: 600 !important; }
.glossary { font-size: 1.0rem; line-height: 1.8; color: #c8d0e0; }
.glossary b { color: #a5b4fc; }
.glossary .g-item { margin-bottom: 0.85rem; padding-left: 0.9rem; border-left: 2px solid #2a3550; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# BRAINROT LIST
# ─────────────────────────────────────────
BRAINROT_LIST = [
    "Jackorilla","Los Matteos","La Vacca Saturno Saturnita","Karkerkar Kurkur",
    "Bisonte Giuppitere","Sammyni Spyderini","Trenostruzzo Turbo 4000",
    "Torrtuginni Dragonfrutini","Dul Dul Dul","Blackhole Goat","Chachechi",
    "Agarrini la Palini","Los Spyderinis","Extinct Tralalero","La Cucaracha",
    "Los Tortus","Vulturino Skeletono","Los Tralaleritos","Zombie Tralala",
    "Boatito Auratito","Guerriro Digitale","Yess my examine","La Karkerkar Combinasion",
    "La Vacca Prese Presente","Reindeer Tralala","Extinct Matteo","Pumpkini Spyderini",
    "Rocco Disco","Las Tralaleritas","Frankentteo","Job Job Job Sahur","Los Trios",
    "Karker Sahur","Los Karkeritos","Las Vaquitas Saturnitas","Santteo","Fishboard",
    "La Vacca Jacko Linterino","Buntteo","Triplito Tralaleritos","Trickolino",
    "Paradiso Axolottino","GOAT","Giftini Spyderini","Graipuss Medussi",
    "Perrito Burrito","Bombardiro Vaccariro","Love Love Love Sahur","1x1x1x1",
    "La Vacca Lepre Lepreino","Los Cucarachas","Hippo Golazo","Easter Easter Easter Sahur",
    "Please my Present","Craburger","Cuadramat and Pakrahmatmamat","Los Jobcitos",
    "Bunnyman","Berryno","Nooo My Hotspot","Noo my examine","La Sahur Combinasion",
    "List List List Sahur","Telemorte","To to to Sahur","Bunny Bunny Bunny Sahur",
    "Glaciator","Pirulitoita Bicicleteira","Pot Hotspot","Santa Hotspot",
    "Ref Ref Ref Sahur","Horegini Boom","Buho De Volto","Naughty Naughty",
    "Pot Pumpkin","Quesadilla Crocodila","Rocketini Frostini","Bunito Bunito Spinito",
    "Cupid Cupid Sahur","Ho Ho Ho Sahur","Mi Gatito","Octoball","Eid Eid Eid Sahur",
    "Chicleteira Bicicleteira","Quesadillo Vampiro","Brunito Marsito","Cupid Hotspot",
    "Flancito","Luck Luck Luck Sahur","Chill Puppy","Burrito Bandito","Granny",
    "Chicleteirina Bicicleteirina","Aquarino","Los Bunitos","Futbolini Skatini",
    "Los Quesadillas","Noo my Candy","Arcadopus","Serafinna Medusella",
    "Los Nooo My Hotspotsitos","Noo my Present","Flipa Sandala","Rang Ring Bus",
    "Strawberrita","Ombrello Topolino","Los Mi Gatitos","Los Chicleteiras",
    "Noo my Eggs","Donkeyturbo Express","John Doe","Sushi Inu","Los Burritos",
    "La Grande Combinasion","Los 25","Tacorillo Crocodillo","Mariachi Corazoni",
    "Swag Soda","Noo my Heart","Noo my Gold","Chimnino","Bananito",
    "Nuclearo Dinossauro","Los Combinasionas","Chicleteira Noelteira",
    "Chicleteira Surfeiteira","Baskito","Tacorita Bicicleta","Los Sweethearts",
    "Spinny Hammy","Camera Ramena","Las Sis","Chicleteira Cupideira","DJ Panda",
    "Girafini Raftini","Los Planitos","Snailo Clovero","Cigno Fulgoro",
    "Frullato Framingo","Los Spooky Combinasionas","Los Jolly Combinasionas",
    "Los Hotspotsitos","Churrito Bunnito","Money Money Puggy","Los Mobilis",
    "Capitano Gullini","Los 67","Celularcini Viciosini","Los Fruits","Los Candies",
    "La Extinct Grande","Los Bros","Bacuru and Egguru","La Spooky Grande",
    "Chillin Chili","Chimnino","Chimino","Chipso and Queso","Money Money Reindeer","Mieteteira Bicicleteira",
    "Tuff Toucan","Tralaledon","Globa Steppa","Gobblino Uniciclino","Los Cupids",
    "W or L","Los Mariachis","Sand Sand Sand","Los Puggies","La Jolly Grande",
    "Esok Sekolah","Los Primos","Eviledon","Los Tacoritas","Fragola La La La",
    "Esok Goala","Lovin Rose","Abyssaloco","Tang Tang Keletang","Coco and Mango",
    "La Taco Combinasion","Dug dug dug","Ketupat Kepat","Tictac Sahur","Orcaledon",
    "Swaggy Bros","La Romantic Grande","La Lucky Grande","Gym Bros",
    "Ketchuru and Musturu","Tirilikalika Tirilikalako","Rico Dinero","Jolly Jolly Sahur",
    "Lavadorito Spinito","Gold Gold Gold","Fishino Clownino","Money Money Bros",
    "Nacho Spyder","Garama and Madundung","La Anniversary Grande","Rosetti Tualetti",
    "Hopilikalika Hopilikalako","Steakini Fattini","Caylusaurus","La Easter Grande",
    "Cloverat Clapat","Spaghetti Tualetti","La Summer Grande","Ventoliero Pavonero",
    "Quackini Snackini","Guest 666","Festive 67","Los Spaghettis","Rubrikiko",
    "Sammyni Fattini","Bearito Cabinito","La Ginger Sekolah","Ginger Gerat",
    "Los Chillis","Los Hackers","Spooky and Pumpky","Sammyni Cakini","Duggy Bros",
    "La Food Combinasion","La Casa Boo","Fragrama and Chocrama","Cash or Card",
    "Los Sekolahs","Foxini Lanternini","Kalika Bros","Pancake and Syrup","Antonio",
    "La Secret Combinasion","Los Amigos","Fortunu and Cashuru","Reinito Sleighito",
    "Ketupat Bros","Arcadragon","Burguro and Fryuro","Cooki and Milki",
    "Capitano Moby","Rosey and Teddy","Bunny and Eggy","Popcuru and Fizzuru",
    "Cerberus","Celestial Pegasus","Jelly Moby","Hydra Bunny","Elefanto Frigo",
    "La Supreme Combinasion","Kraken","Digi Narwhal","Love Love Bear",
    "Dragon Cannellloni","Dragon Cannelloni","Signore Carapace",
    "Hydra Dragon Cannelloni",
    "Dragon Gingerini","Dragon Aquanini","Griffin",
    "Shroombino","Mr Carrot","Tomatrio","King Limone","Carrot","Mango",
    "My Heart","My Gold","My Candy","My Eggs","My Present","67","25",
]
BRAINROT_SORTED = sorted(BRAINROT_LIST, key=len, reverse=True)

def _sil_label(v):
    """Interpretasi Silhouette Score menurut Kaufman & Rousseeuw (1990)."""
    if v >= 0.71: return "struktur kuat"
    if v >= 0.51: return "struktur memadai"
    if v >= 0.26: return "struktur lemah"
    return "tidak berstruktur"

def _sil_color(v):
    if v >= 0.51: return "#6ee7b7"
    if v >= 0.26: return "#fde68a"
    return "#fca5a5"

CLUSTER_META = {
    0: {"name": "Dead Stock",          "color": "#60a5fa", "emoji": "💀",
        "desc": "Produk hampir tidak pernah terjual. Frekuensi dan pendapatan sangat rendah. Pertimbangkan untuk tidak melanjutkan stok produk ini."},
    1: {"name": "Low Tier",            "color": "#f87171", "emoji": "📦",
        "desc": "Produk dengan performa penjualan menengah. Masih menghasilkan pendapatan tapi belum jadi prioritas utama."},
    2: {"name": "High Volume Anomaly", "color": "#4ade80", "emoji": "⚡",
        "desc": "Produk dibeli dalam jumlah unit sangat besar per transaksi — umumnya komoditas bernilai satuan rendah yang dijual borongan. Volume ekstrem namun pendapatan per produk relatif kecil."},
    3: {"name": "Star Products",       "color": "#c084fc", "emoji": "⭐",
        "desc": "Produk unggulan dengan frekuensi transaksi dan pendapatan tertinggi. Ini adalah tulang punggung lapak — prioritas utama untuk dijaga stoknya."},
}

# Normalisasi ejaan: satu produk yang ditulis dua ejaan berbeda oleh penjual
ALIAS_EJAAN = {
    "Chimnino": "Chimino",
    "Dragon Cannellloni": "Dragon Cannelloni",
}

# Listing dengan isi TIDAK PASTI tidak boleh dicocokkan ke satu nama produk,
# karena item yang diterima pembeli tidak deterministik.
POLA_TARGET_GENERIK = re.compile(
    r'\b(random|any)\s+(secret|brainrot|plant|item|mutation)\b', re.IGNORECASE)

def isi_tidak_pasti(nama):
    n = nama.lower()
    if 'random' in n and ' or ' in n:      # menawarkan dua kemungkinan item
        return True
    return bool(POLA_TARGET_GENERIK.search(n))   # target acak generik

def match_brainrot(name):
    if isi_tidak_pasti(name):
        return name
    name_lower = name.lower()
    for br in BRAINROT_SORTED:
        if br.lower() in name_lower:
            return br
    return name

def super_clean_title(text):
    text = str(text).lower()
    # Baris baru diubah jadi spasi, BUKAN dijadikan pemisah. Banyak penjual
    # menulis nama mutasi di baris 1 dan nama brainrot di baris 2.
    text = text.replace('\n', ' ')
    text = re.split(r'[-|,]', text)[0]
    text = re.sub(r'\d+\.?\d*\s*(m/s|sp/s|s/s|ms)', '', text)
    return " ".join(text.split()).title()

@st.cache_data(show_spinner=False)
def load_and_process(file_bytes_list):
    import io
    dfs = []
    for b in file_bytes_list:
        try:
            dfs.append(pd.read_csv(io.BytesIO(b)))
        except Exception:
            pass
    if not dfs:
        return None, None, None, None, None

    df_all = pd.concat(dfs, ignore_index=True)
    df_clean = df_all[df_all['Order State'] == 'Completed'].copy()
    df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'])
    df_clean['Title'] = df_clean['Title'].fillna(
        df_clean['Offer Type'].astype(str) + " - " + df_clean['Description'].astype(str))
    df_clean['Base_Name'] = df_clean['Title'].apply(super_clean_title)
    df_clean = df_clean[df_clean['Base_Name'] != 'Requestedboosting']
    df_clean['Base_Name'] = df_clean['Base_Name'].apply(match_brainrot)
    df_clean['Base_Name'] = df_clean['Base_Name'].replace(ALIAS_EJAAN)

    product_base = df_clean.groupby('Base_Name').agg(
        Frequency=('Order Id', 'count'),
        Total_Volume=('Purchase Quantity', 'sum'),
        Total_Revenue=('Total Order Amount', 'sum')
    ).reset_index()

    X = product_base[['Frequency','Total_Volume','Total_Revenue']].values
    X_log = np.log1p(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_log)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    product_base['Cluster'] = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, product_base['Cluster'])

    # Remap label agar penomoran klaster konsisten dengan naskah skripsi.
    # Urutan penetapan dipilih dari ciri paling tidak ambigu lebih dulu:
    #   3 = Star (pendapatan tertinggi), 2 = Anomali (volume tertinggi),
    #   0 = Dead Stock (paling jarang dipesan), 1 = Low Tier (sisanya)
    profil = product_base.groupby('Cluster')[
        ['Frequency', 'Total_Volume', 'Total_Revenue']].mean()
    c_star    = profil['Total_Revenue'].idxmax()
    sisa      = profil.drop(index=c_star)
    c_anomali = sisa['Total_Volume'].idxmax()
    sisa      = sisa.drop(index=c_anomali)
    c_dead    = sisa['Frequency'].idxmin()
    c_low     = [c for c in sisa.index if c != c_dead][0]
    remap = {c_dead: 0, c_low: 1, c_anomali: 2, c_star: 3}
    product_base['Cluster'] = product_base['Cluster'].map(remap)

    df_clean['Month_Year'] = df_clean['Order Date'].dt.to_period('M')
    trend_df = df_clean.groupby(['Base_Name','Month_Year'])['Order Id'].count().reset_index(name='Freq')
    pivot_trend = trend_df.pivot(index='Base_Name', columns='Month_Year', values='Freq').fillna(0)
    pivot_trend.columns = [str(c) for c in pivot_trend.columns]

    final_spk = pd.merge(product_base, pivot_trend, on='Base_Name', how='left')
    month_cols = list(pivot_trend.columns)
    if len(month_cols) >= 2:
        final_spk['Trend_Status'] = np.where(
            final_spk[month_cols[-1]] > final_spk[month_cols[-2]], 'Naik', 'Turun')
    else:
        final_spk['Trend_Status'] = 'Turun'

    def keputusan(row):
        if row['Cluster'] in [2, 3] and row['Trend_Status'] == 'Naik':
            return 'RESTOCK SEGERA'
        elif row['Cluster'] == 0:
            return 'NON-RESTOCK'
        return 'AMAN'

    final_spk['Rekomendasi'] = final_spk.apply(keputusan, axis=1)
    return product_base, final_spk, pivot_trend, sil, df_clean

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Space Grotesk, sans-serif; font-size: 1.1rem; font-weight: 700; color: #a5b4fc; margin-bottom: 1rem;'>
    🎮 ELDORADO SPK
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.85rem; font-weight:700; color:#6b7590; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;'>
    UPLOAD DATA CSV
    </div>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Pilih file CSV transaksi dari Eldorado.gg",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload satu atau lebih file CSV yang diunduh dari halaman Invoice Eldorado.gg"
    )

    # Data contoh (opsional). Bila folder 'data' berisi berkas CSV, pengguna
    # dapat mencoba aplikasi tanpa menyiapkan berkas sendiri terlebih dahulu.
    DATA_DIR = Path(__file__).parent / "data"
    berkas_contoh = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.is_dir() else []

    if berkas_contoh and not uploaded_files:
        st.caption("Belum punya berkas CSV?")
        if st.button("Muat data contoh", use_container_width=True):
            st.session_state["pakai_contoh"] = True
        if st.session_state.get("pakai_contoh"):
            st.success(f"Data contoh aktif ({len(berkas_contoh)} berkas)")
            if st.button("Kosongkan data contoh", use_container_width=True):
                st.session_state["pakai_contoh"] = False
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.85rem; color:#8892a4; line-height:1.8'>
    <b style='color:#e8eaf0'>Tentang Sistem Ini</b><br>
    Dashboard ini menganalisis data transaksi penjualan item virtual di platform
    Eldorado.gg, didominasi game <b style='color:#a5b4fc'>Steal a Brainrot</b>
    dengan sebagian item <b style='color:#a5b4fc'>Plants vs Brainrots</b>,
    menggunakan metode <b style='color:#a5b4fc'>K-Means Clustering</b>
    dan analisis tren waktu untuk menghasilkan rekomendasi manajemen produk.<br><br>
    <b style='color:#e8eaf0'>Metode:</b><br>
    CRISP-DM · Regex · Brainrot Matching<br>
    Log1p · MinMaxScaler · K-Means (k=4)<br>
    Elbow Method · Silhouette Score<br>
    Time-Series · SPK Rule-Based
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#3d4a63; text-align:center'>
    Eldorado.gg · Aset Virtual<br>
    Sep 2025 – Mar 2026
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">🎮 Virtual Asset · K-Means Clustering · SPK</div>
    <h1>Eldorado Product Intelligence</h1>
    <p>Sistem Pendukung Keputusan Manajemen Produk Virtual — Eldorado.gg<br>
    <span style='font-size:0.9rem; color:#6b7590'>Upload file CSV transaksi di sidebar kiri untuk memulai analisis otomatis</span></p>
</div>
""", unsafe_allow_html=True)

pakai_contoh = bool(st.session_state.get("pakai_contoh")) and bool(berkas_contoh)

if not uploaded_files and not pakai_contoh:
    st.markdown("""
    <div class="info-box">
        📂 <b>Cara Penggunaan:</b> Upload file CSV invoice dari Eldorado.gg di sidebar kiri.
        Sistem akan otomatis membersihkan data, mengelompokkan produk berdasarkan nama brainrot,
        menjalankan K-Means Clustering (k=4), menganalisis tren bulanan, dan menghasilkan
        rekomendasi SPK untuk setiap produk.
    </div>
    """, unsafe_allow_html=True)

    # Tampilkan panduan klaster dan logika SPK saat belum ada data
    st.markdown('<div class="section-title">📌 Panduan: 4 Kategori Produk</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Sistem mengelompokkan semua produk ke dalam 4 kategori berdasarkan frekuensi transaksi, volume penjualan, dan total pendapatan.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cluster-grid">
        <div class="cluster-card" style="border-left: 4px solid #c084fc">
            <div class="cluster-name" style="color:#c084fc">⭐ Cluster 3 — Star Products</div>
            <div class="cluster-desc">Produk unggulan dengan frekuensi dan pendapatan tertinggi. Prioritas utama untuk dijaga stoknya.</div>
        </div>
        <div class="cluster-card" style="border-left: 4px solid #4ade80">
            <div class="cluster-name" style="color:#4ade80">⚡ Cluster 2 — High Volume Anomaly</div>
            <div class="cluster-desc">Produk dibeli dalam jumlah unit sangat besar per transaksi — komoditas bernilai satuan rendah yang dijual borongan.</div>
        </div>
        <div class="cluster-card" style="border-left: 4px solid #f87171">
            <div class="cluster-name" style="color:#f87171">📦 Cluster 1 — Low Tier</div>
            <div class="cluster-desc">Produk dengan performa menengah. Masih menghasilkan tapi belum jadi prioritas utama.</div>
        </div>
        <div class="cluster-card" style="border-left: 4px solid #60a5fa">
            <div class="cluster-name" style="color:#60a5fa">💀 Cluster 0 — Dead Stock</div>
            <div class="cluster-desc">Produk hampir tidak pernah terjual. Frekuensi dan pendapatan sangat rendah.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.5rem">🎯 Logika Rekomendasi SPK</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Sistem menghasilkan rekomendasi berdasarkan kombinasi kategori klaster dan tren penjualan 2 bulan terakhir.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="spk-grid">
        <div class="spk-card spk-card-restock">
            <div class="spk-title spk-title-restock">🚨 RESTOCK SEGERA</div>
            <div class="spk-body">
                <b>Kondisi:</b> Produk berada di Cluster 2 atau 3 (produk laku)<br>
                <b>DAN</b> tren penjualan bulan terakhir sedang <b>Naik</b><br><br>
                <i>Artinya: Produk ini laku dan permintaannya sedang meningkat — tambah stok sekarang sebelum kehabisan.</i>
            </div>
        </div>
        <div class="spk-card spk-card-nonrestock">
            <div class="spk-title spk-title-nonrestock">⏸️ NON-RESTOCK</div>
            <div class="spk-body">
                <b>Kondisi:</b> Produk berada di Cluster 0 (Dead Stock)<br><br>
                <i>Artinya: Produk ini sudah lama tidak diminati pasar. Produk tetap tercatat dalam daftar, hanya tidak diprioritaskan untuk diadakan ulang (restock).</i>
            </div>
        </div>
        <div class="spk-card spk-card-aman">
            <div class="spk-title spk-title-aman">✅ AMAN</div>
            <div class="spk-body">
                <b>Kondisi:</b> Selain kondisi di atas<br><br>
                <i>Artinya: Produk dalam kondisi stabil atau tren sedang turun sementara — monitor berkala, tidak perlu tindakan mendesak.</i>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Process
if uploaded_files:
    file_bytes = [f.read() for f in uploaded_files]
else:
    file_bytes = [f.read_bytes() for f in berkas_contoh]
    st.info("Menampilkan **data contoh** dari penelitian ini "
            "(lapak Eldorado.gg, September 2025 – Maret 2026). "
            "Unggah berkas CSV Anda sendiri di sidebar untuk menganalisis data lain.")
with st.spinner("⚙️ Memproses data, menjalankan K-Means Clustering, dan menghitung rekomendasi SPK..."):
    product_base, final_spk, pivot_trend, sil, df_clean = load_and_process(file_bytes)

if product_base is None:
    st.error("❌ Gagal memproses file. Pastikan file CSV memiliki kolom: Order Id, Order Date, Title, Order State, Purchase Quantity, Total Order Amount.")
    st.stop()

# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────
total_produk  = len(product_base)
star_count    = len(product_base[product_base['Cluster'] == 3])
dead_count    = len(product_base[product_base['Cluster'] == 0])
restock_count = len(final_spk[final_spk['Rekomendasi'] == 'RESTOCK SEGERA'])
tgl_min = df_clean['Order Date'].min().strftime('%d %b %Y')
tgl_max = df_clean['Order Date'].max().strftime('%d %b %Y')
total_transaksi = len(df_clean)
total_revenue = df_clean['Total Order Amount'].sum()

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card metric-accent-purple">
        <div class="metric-label">📦 Total Produk Unik</div>
        <div class="metric-value">{total_produk:,}</div>
        <div class="metric-sub">{tgl_min} – {tgl_max}</div>
        <div class="metric-desc">Jumlah produk unik yang berhasil diidentifikasi dan dianalisis dari seluruh data transaksi.</div>
    </div>
    <div class="metric-card metric-accent-c3">
        <div class="metric-label">⭐ Star Products</div>
        <div class="metric-value">{star_count}</div>
        <div class="metric-sub">Cluster 3 · Produk unggulan</div>
        <div class="metric-desc">Produk dengan frekuensi transaksi dan pendapatan tertinggi — tulang punggung lapak yang wajib dijaga stoknya.</div>
    </div>
    <div class="metric-card metric-accent-c0">
        <div class="metric-label">💀 Dead Stock</div>
        <div class="metric-value">{dead_count}</div>
        <div class="metric-sub">Cluster 0 · Tidak perlu restock</div>
        <div class="metric-desc">Produk yang hampir tidak pernah dibeli dalam periode analisis — tidak perlu diadakan ulang, namun tetap tercatat pada daftar.</div>
    </div>
    <div class="metric-card metric-accent-green">
        <div class="metric-label">🚨 Restock Segera</div>
        <div class="metric-value">{restock_count}</div>
        <div class="metric-sub">Silhouette Score: {sil:.3f}</div>
        <div class="metric-desc">Produk laku yang trennya sedang naik — perlu segera ditambah stoknya. Silhouette Score {sil:.3f} ({_sil_label(sil)}).</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PANEL PENJELASAN ISTILAH
# ─────────────────────────────────────────
with st.expander("📖  Apa arti angka dan istilah di halaman ini?  (klik untuk membuka)"):
    st.markdown(f"""
    <div class="glossary">
    <div class="g-item"><b>Produk Unik</b> — jumlah jenis produk setelah nama-nama varian disatukan.
    Satu produk sering dijual dengan puluhan judul berbeda (misalnya "Galaxy Los 67", "Gold Los 67",
    "Radioactive Los 67"). Semua varian itu dihitung sebagai <b>satu</b> produk: "Los 67".
    Saat ini terdapat <b>{total_produk} produk unik</b> dari {total_transaksi:,} transaksi.</div>

    <div class="g-item"><b>Frequency (Frekuensi)</b> — berapa <b>kali</b> sebuah produk dipesan.
    Satu pesanan dihitung satu, tidak peduli pembeli membeli 1 unit atau 1.000 unit.</div>

    <div class="g-item"><b>Total Volume</b> — berapa <b>unit</b> barang yang berpindah tangan.
    Berbeda dengan Frequency: satu pesanan berisi 500 koin dihitung 1 frekuensi tetapi 500 volume.</div>

    <div class="g-item"><b>Total Revenue</b> — total uang masuk dari produk tersebut selama periode
    analisis, dalam dolar AS (USD).</div>

    <div class="g-item"><b>Cluster (Klaster)</b> — kelompok produk yang dibentuk otomatis oleh
    algoritma <b>K-Means</b>. Komputer mengelompokkan produk yang pola penjualannya mirip, tanpa
    diberi tahu sebelumnya kelompok apa saja yang ada. Terbentuk 4 kelompok: Dead Stock,
    Low Tier, High Volume Anomaly, dan Star Products.</div>

    <div class="g-item"><b>K-Means Clustering</b> — metode pengelompokan otomatis. Cara kerjanya:
    komputer menaruh 4 titik pusat secara acak, lalu berulang kali memindahkan tiap produk ke pusat
    terdekat dan menggeser pusatnya, sampai posisinya tidak berubah lagi.</div>

    <div class="g-item"><b>Status Tren</b> — perbandingan jumlah pesanan <b>bulan terakhir</b>
    dengan <b>bulan sebelumnya</b>. Kalau naik ditandai "Naik", kalau sama atau turun ditandai
    "Turun". Ini dipakai untuk melihat arah permintaan terkini, bukan meramalkan masa depan.</div>

    <div class="g-item"><b>Rekomendasi SPK</b> — keputusan otomatis hasil penggabungan dua hal di
    atas: kategori klaster + status tren. Ada tiga kemungkinan: RESTOCK SEGERA, NON-RESTOCK,
    atau AMAN.</div>

    <div class="g-item"><b>Silhouette Score</b> — nilai antara -1 sampai 1 yang mengukur seberapa
    rapi pengelompokan terbentuk. Nilainya tinggi bila tiap produk dekat dengan teman sekelompoknya
    dan jauh dari kelompok lain. Nilai saat ini <b>{sil:.3f}</b> ({_sil_label(sil)}).</div>

    <div class="g-item"><b>Log1p</b> — perhitungan matematis untuk meredam angka yang terlalu
    ekstrem. Diperlukan karena ada produk dengan volume jutaan unit, sementara mayoritas hanya
    puluhan; tanpa peredaman, produk ekstrem itu akan menguasai seluruh perhitungan.</div>

    <div class="g-item"><b>MinMaxScaler</b> — mengubah semua angka ke skala seragam 0 sampai 1,
    supaya frekuensi (satuan "kali"), volume (satuan "unit"), dan pendapatan (satuan "dolar")
    bisa dibandingkan secara adil.</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Segmentasi Klaster",
    "📈  Tren Penjualan",
    "🎯  Rekomendasi SPK",
    "🔬  Evaluasi Model"
])

# ══ TAB 1 ══
with tab1:
    st.markdown('<div class="section-title">Peta Segmentasi Produk (K-Means Clustering, k=4)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Setiap titik pada grafik mewakili satu produk. Posisi titik menunjukkan seberapa sering produk dipesan (sumbu X) dan berapa total pendapatannya (sumbu Y). Warna menunjukkan kategori klaster. <b>Hover pada titik</b> untuk melihat nama produk dan detailnya.</div>', unsafe_allow_html=True)

    product_base['Cluster_Name'] = product_base['Cluster'].map(
        lambda x: f"Cluster {x} — {CLUSTER_META[x]['name']}")
    color_map = {f"Cluster {k} — {v['name']}": v['color'] for k,v in CLUSTER_META.items()}

    fig_scatter = px.scatter(
        product_base, x='Frequency', y='Total_Revenue',
        color='Cluster_Name', color_discrete_map=color_map,
        hover_name='Base_Name',
        hover_data={'Frequency': True, 'Total_Volume': True, 'Total_Revenue': ':.1f', 'Cluster_Name': False},
        labels={'Frequency': 'Total Frekuensi Terjual (kali)', 'Total_Revenue': 'Total Pendapatan (USD)'},
        template='plotly_dark'
    )
    fig_scatter.update_traces(marker=dict(size=10, opacity=0.85))
    fig_scatter.update_layout(
        paper_bgcolor='#1a1f2e', plot_bgcolor='#1a1f2e',
        legend_title_text='Kategori Produk',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c8d0e0', size=14)),
        font=dict(color='#c8d0e0', size=14),
        xaxis=dict(gridcolor='#2a3550', title_font=dict(size=15)),
        yaxis=dict(gridcolor='#2a3550', title_font=dict(size=15)),
        margin=dict(l=0, r=0, t=20, b=0), height=480
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown(f'<div class="chart-caption">Peta segmentasi {total_produk} produk. Produk di pojok kanan atas (frekuensi tinggi, pendapatan tinggi) adalah Star Products. Produk yang menumpuk di pojok kiri bawah adalah Dead Stock.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">Jumlah Produk per Kategori</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Grafik ini menunjukkan berapa banyak produk yang masuk ke tiap kategori klaster. Semakin tinggi batang, semakin banyak produk di kategori tersebut.</div>', unsafe_allow_html=True)

        cluster_counts = product_base['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster','Jumlah']
        cluster_counts['Nama'] = cluster_counts['Cluster'].map(lambda x: f"{CLUSTER_META[x]['emoji']} {CLUSTER_META[x]['name']}")
        cluster_counts['Warna'] = cluster_counts['Cluster'].map(lambda x: CLUSTER_META[x]['color'])
        cluster_counts = cluster_counts.sort_values('Cluster')

        fig_bar = px.bar(
            cluster_counts, x='Nama', y='Jumlah', color='Nama',
            color_discrete_map={f"{v['emoji']} {v['name']}": v['color'] for v in CLUSTER_META.values()},
            text='Jumlah', template='plotly_dark'
        )
        fig_bar.update_traces(textposition='outside', textfont=dict(color='#e8eaf0', size=14))
        fig_bar.update_layout(
            paper_bgcolor='#1a1f2e', plot_bgcolor='#1a1f2e', showlegend=False,
            font=dict(color='#c8d0e0', size=14),
            margin=dict(l=0, r=0, t=20, b=0), height=320,
            xaxis=dict(title='', gridcolor='#2a3550'),
            yaxis=dict(title='Jumlah Produk', gridcolor='#2a3550')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(f'<div class="chart-caption">Terdapat {dead_count} produk Dead Stock dari total {total_produk} produk — porsi terbesar katalog justru sudah tidak diminati pasar.</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">Proporsi Komposisi Klaster</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-desc">Grafik donut ini menunjukkan persentase produk di tiap kategori dari total keseluruhan {total_produk} produk.</div>', unsafe_allow_html=True)

        fig_pie = px.pie(
            cluster_counts, values='Jumlah', names='Nama', color='Nama',
            color_discrete_map={f"{v['emoji']} {v['name']}": v['color'] for v in CLUSTER_META.values()},
            template='plotly_dark', hole=0.45
        )
        fig_pie.update_traces(
            textinfo='percent+label',
            textfont=dict(color='#e8eaf0', size=13)
        )
        fig_pie.update_layout(
            paper_bgcolor='#1a1f2e', showlegend=False,
            font=dict(color='#c8d0e0', size=14),
            margin=dict(l=0, r=0, t=20, b=0), height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        _star_pct = star_count / total_produk * 100 if total_produk else 0
        _star_rev = product_base.loc[product_base['Cluster']==3,'Total_Revenue'].sum() / product_base['Total_Revenue'].sum() * 100
        st.markdown(f'<div class="chart-caption">Star Products hanya {_star_pct:.0f}% dari jumlah produk, namun menyumbang {_star_rev:.0f}% dari total pendapatan lapak.</div>', unsafe_allow_html=True)

    # Legenda klaster
    st.markdown('<div class="section-title" style="margin-top:1.5rem">📖 Penjelasan Tiap Kategori Klaster</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="cluster-grid">
        <div class="cluster-card" style="border-left: 4px solid {CLUSTER_META[3]['color']}">
            <div class="cluster-name" style="color:{CLUSTER_META[3]['color']}">⭐ Cluster 3 — Star Products</div>
            <div class="cluster-desc">{CLUSTER_META[3]['desc']}</div>
        </div>
        <div class="cluster-card" style="border-left: 4px solid {CLUSTER_META[2]['color']}">
            <div class="cluster-name" style="color:{CLUSTER_META[2]['color']}">⚡ Cluster 2 — High Volume Anomaly</div>
            <div class="cluster-desc">{CLUSTER_META[2]['desc']}</div>
        </div>
        <div class="cluster-card" style="border-left: 4px solid {CLUSTER_META[1]['color']}">
            <div class="cluster-name" style="color:{CLUSTER_META[1]['color']}">📦 Cluster 1 — Low Tier</div>
            <div class="cluster-desc">{CLUSTER_META[1]['desc']}</div>
        </div>
        <div class="cluster-card" style="border-left: 4px solid {CLUSTER_META[0]['color']}">
            <div class="cluster-name" style="color:{CLUSTER_META[0]['color']}">💀 Cluster 0 — Dead Stock</div>
            <div class="cluster-desc">{CLUSTER_META[0]['desc']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Profil klaster
    st.markdown('<div class="section-title" style="margin-top:1.5rem">📊 Profil Rata-Rata per Kategori</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Tabel ini menunjukkan rata-rata performa produk di tiap kategori. <b>Rata-rata Frekuensi</b> = berapa kali satu produk dipesan; <b>Rata-rata Volume</b> = berapa unit barang berpindah; <b>Rata-rata Revenue</b> = uang masuk per produk. Perhatikan Cluster 2: volumenya jutaan unit tapi pendapatannya kecil — itulah ciri produk borongan bernilai satuan rendah.</div>', unsafe_allow_html=True)

    profile = product_base.groupby('Cluster').agg(
        Avg_Frequency=('Frequency','mean'),
        Avg_Volume=('Total_Volume','mean'),
        Avg_Revenue=('Total_Revenue','mean'),
        Jumlah_Produk=('Base_Name','count')
    ).round(2).reset_index()
    profile['Kategori'] = profile['Cluster'].map(lambda x: f"{CLUSTER_META[x]['emoji']} Cluster {x} — {CLUSTER_META[x]['name']}")
    profile['Avg_Revenue'] = profile['Avg_Revenue'].map(lambda x: f"${x:,.2f}")
    profile['Avg_Volume'] = profile['Avg_Volume'].map(lambda x: f"{x:,.2f}")
    profile = profile[['Kategori','Avg_Frequency','Avg_Volume','Avg_Revenue','Jumlah_Produk']]
    profile.columns = ['Kategori','Rata-rata Frekuensi (x)','Rata-rata Volume (unit)','Rata-rata Revenue (USD)','Jumlah Produk']
    st.dataframe(profile, use_container_width=True, hide_index=True)

    # Top 10 per cluster
    st.markdown('<div class="section-title" style="margin-top:1.5rem">🏆 Top 10 Produk per Kategori</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Pilih kategori untuk melihat 10 produk dengan pendapatan tertinggi di kategori tersebut. Daftar ini berguna untuk mengecek apakah pengelompokan masuk akal — produk di Cluster 3 seharusnya nama-nama yang memang laku, sedangkan Cluster 0 berisi produk yang hampir tak pernah terjual.</div>', unsafe_allow_html=True)
    sel_cluster = st.selectbox("Pilih Kategori Klaster", options=[3,2,1,0],
        format_func=lambda x: f"{CLUSTER_META[x]['emoji']} Cluster {x} — {CLUSTER_META[x]['name']}")
    top10 = (product_base[product_base['Cluster']==sel_cluster]
             .sort_values('Total_Revenue', ascending=False)
             .head(10)[['Base_Name','Frequency','Total_Volume','Total_Revenue']]
             .rename(columns={'Base_Name':'Nama Produk','Frequency':'Frekuensi (x)',
                              'Total_Volume':'Volume (unit)','Total_Revenue':'Revenue (USD)'}))
    top10['Revenue (USD)'] = top10['Revenue (USD)'].map(lambda x: f"${x:,.1f}")
    top10['Volume (unit)'] = top10['Volume (unit)'].map(lambda x: f"{x:,}")
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ══ TAB 2 ══
with tab2:
    st.markdown('<div class="section-title">Tren Penjualan Bulanan — Top 5 Star Products</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Grafik ini menunjukkan pergerakan jumlah pesanan dari bulan ke bulan untuk 5 produk unggulan terlaris. Sumbu tegak = berapa kali produk dipesan pada bulan tersebut. Garis naik berarti permintaan meningkat, garis turun berarti menurun. <b>Catatan:</b> ini pembacaan arah tren dari data yang sudah terjadi, bukan peramalan (<i>forecasting</i>) penjualan ke depan.</div>', unsafe_allow_html=True)

    star_products = (final_spk[final_spk['Cluster']==3]
                     .sort_values('Frequency', ascending=False)
                     .head(5)['Base_Name'].tolist())
    month_cols = list(pivot_trend.columns)

    if star_products and month_cols:
        fig_trend = go.Figure()
        colors_trend = ['#38bdf8','#fb923c','#a3e635','#f472b6','#facc15']
        for i, item in enumerate(star_products):
            if item in pivot_trend.index:
                tren = pivot_trend.loc[item]
                fig_trend.add_trace(go.Scatter(
                    x=list(tren.index), y=tren.values,
                    mode='lines+markers+text',
                    name=item,
                    line=dict(color=colors_trend[i%len(colors_trend)], width=3),
                    marker=dict(size=9),
                ))
        fig_trend.update_layout(
            paper_bgcolor='#1a1f2e', plot_bgcolor='#1a1f2e',
            font=dict(color='#c8d0e0', size=14),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c8d0e0', size=14),
                       title_text='Nama Produk', title_font=dict(size=13)),
            xaxis=dict(title='Periode Bulan', gridcolor='#2a3550', title_font=dict(size=15)),
            yaxis=dict(title='Frekuensi Transaksi (kali)', gridcolor='#2a3550', title_font=dict(size=15)),
            margin=dict(l=0, r=0, t=20, b=0), height=450
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('<div class="chart-caption">Grafik tren bulanan 5 Star Products terlaris. Puncak transaksi umumnya terjadi saat game sedang ramai dimainkan. Produk yang trennya naik di bulan terakhir akan mendapat rekomendasi RESTOCK SEGERA.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.5rem">🗺️ Heatmap Tren Penjualan — Top 20 Produk</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Heatmap ini menunjukkan intensitas penjualan 20 produk teratas dari bulan ke bulan. Warna lebih gelap berarti frekuensi transaksi lebih rendah, warna lebih terang berarti lebih tinggi. Berguna untuk melihat pola musiman secara sekaligus.</div>', unsafe_allow_html=True)

    top20_names = product_base.sort_values('Total_Revenue', ascending=False).head(20)['Base_Name'].tolist()
    heat_data = pivot_trend.loc[pivot_trend.index.isin(top20_names)].copy()
    heat_data.columns = [str(c) for c in heat_data.columns]
    if not heat_data.empty:
        fig_heat = px.imshow(
            heat_data,
            labels=dict(x='Bulan', y='Nama Produk', color='Frekuensi Transaksi'),
            color_continuous_scale='Blues',
            template='plotly_dark', aspect='auto'
        )
        fig_heat.update_layout(
            paper_bgcolor='#1a1f2e', plot_bgcolor='#1a1f2e',
            font=dict(color='#c8d0e0', size=14),
            margin=dict(l=0, r=0, t=20, b=0),
            height=550, xaxis=dict(tickangle=45, tickfont=dict(size=13)),
            yaxis=dict(tickfont=dict(size=13)),
            coloraxis_colorbar=dict(title='Frekuensi', tickfont=dict(size=11))
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('<div class="chart-caption">Cara membaca: tiap baris satu produk, tiap kolom satu bulan. Semakin <b>terang/pekat</b> warna sel, semakin banyak transaksi di bulan itu. Sel gelap berarti tidak ada transaksi sama sekali.</div>', unsafe_allow_html=True)

# ══ TAB 3 ══
with tab3:
    st.markdown('<div class="section-title">Sistem Pendukung Keputusan (SPK) Manajemen Produk</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Sistem ini menggabungkan hasil segmentasi K-Means dengan analisis tren penjualan bulanan untuk menghasilkan rekomendasi tindakan preskriptif bagi setiap produk. Gunakan filter di bawah untuk menyaring hasil sesuai kebutuhan.</div>', unsafe_allow_html=True)

    # Logika SPK
    st.markdown("""
    <div class="spk-grid">
        <div class="spk-card spk-card-restock">
            <div class="spk-title spk-title-restock">🚨 RESTOCK SEGERA</div>
            <div class="spk-body">
                <b>Syarat:</b> Produk di Cluster 2/3 (laku) <b>DAN</b> tren bulan terakhir <b>Naik</b><br><br>
                Produk ini sedang diminati dan stoknya berisiko habis. Segera tambah stok untuk memaksimalkan pendapatan.
            </div>
        </div>
        <div class="spk-card spk-card-nonrestock">
            <div class="spk-title spk-title-nonrestock">⏸️ NON-RESTOCK</div>
            <div class="spk-body">
                <b>Syarat:</b> Produk berada di Cluster 0 (Dead Stock)<br><br>
                Produk sudah tidak diminati pasar dalam periode analisis. Tidak perlu diadakan ulang, namun tetap tercatat pada daftar dan dapat dipantau sewaktu-waktu.
            </div>
        </div>
        <div class="spk-card spk-card-aman">
            <div class="spk-title spk-title-aman">✅ AMAN</div>
            <div class="spk-body">
                <b>Syarat:</b> Selain kondisi di atas<br><br>
                Produk dalam kondisi stabil atau trennya sedang menurun sementara. Pantau secara berkala — tidak perlu tindakan mendesak saat ini.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ringkasan
    restock_count2 = len(final_spk[final_spk['Rekomendasi'] == 'RESTOCK SEGERA'])
    nonrestock_count = len(final_spk[final_spk['Rekomendasi'] == 'NON-RESTOCK'])
    aman_count     = len(final_spk[final_spk['Rekomendasi'] == 'AMAN'])

    st.markdown(f"""
    <div style='display:grid; grid-template-columns: repeat(3,1fr); gap:0.75rem; margin-bottom:1.5rem'>
        <div style='background:#1a1f2e; border:1px solid rgba(239,68,68,0.3); border-radius:10px; padding:1rem; text-align:center'>
            <div style='font-size:1.8rem; font-weight:700; color:#f87171'>{restock_count2}</div>
            <div style='font-size:0.85rem; color:#6b7590; margin-top:0.25rem'>Produk perlu RESTOCK</div>
        </div>
        <div style='background:#1a1f2e; border:1px solid rgba(107,114,128,0.3); border-radius:10px; padding:1rem; text-align:center'>
            <div style='font-size:1.8rem; font-weight:700; color:#9ca3af'>{nonrestock_count}</div>
            <div style='font-size:0.85rem; color:#6b7590; margin-top:0.25rem'>Produk NON-RESTOCK</div>
        </div>
        <div style='background:#1a1f2e; border:1px solid rgba(52,211,153,0.3); border-radius:10px; padding:1rem; text-align:center'>
            <div style='font-size:1.8rem; font-weight:700; color:#6ee7b7'>{aman_count}</div>
            <div style='font-size:0.85rem; color:#6b7590; margin-top:0.25rem'>Produk status AMAN</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_cluster = st.multiselect("Filter Kategori Klaster", options=[3,2,1,0],
            default=[3,2,1,0],
            format_func=lambda x: f"{CLUSTER_META[x]['emoji']} Cluster {x} — {CLUSTER_META[x]['name']}")
    with col_f2:
        filter_rec = st.multiselect("Filter Rekomendasi",
            options=['RESTOCK SEGERA','NON-RESTOCK','AMAN'],
            default=['RESTOCK SEGERA','NON-RESTOCK','AMAN'])
    with col_f3:
        filter_tren = st.multiselect("Filter Status Tren",
            options=['Naik','Turun'], default=['Naik','Turun'],
            help="Tren dihitung dari perbandingan frekuensi 2 bulan terakhir data")

    spk_display = final_spk[
        (final_spk['Cluster'].isin(filter_cluster)) &
        (final_spk['Rekomendasi'].isin(filter_rec)) &
        (final_spk['Trend_Status'].isin(filter_tren))
    ][['Base_Name','Cluster','Frequency','Total_Revenue','Trend_Status','Rekomendasi']].copy()

    spk_display['Cluster'] = spk_display['Cluster'].map(
        lambda x: f"{CLUSTER_META[x]['emoji']} C{x} — {CLUSTER_META[x]['name']}")
    spk_display['Total_Revenue'] = spk_display['Total_Revenue'].map(lambda x: f"${x:,.1f}")
    spk_display['Frequency'] = spk_display['Frequency'].map(lambda x: f"{x:,}x")
    spk_display = spk_display.rename(columns={
        'Base_Name':'Nama Produk',
        'Frequency':'Frekuensi Transaksi',
        'Total_Revenue':'Total Revenue (USD)',
        'Trend_Status':'Status Tren Terakhir',
        'Rekomendasi':'Rekomendasi SPK'
    })

    st.dataframe(spk_display, use_container_width=True, hide_index=True, height=450)
    st.caption(f"Menampilkan {len(spk_display):,} dari {len(final_spk):,} produk · "
               f"Kolom Frekuensi = berapa kali dipesan · Revenue = total uang masuk · "
               f"Status Tren = perbandingan bulan terakhir vs bulan sebelumnya")

    csv_out = spk_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Hasil SPK (.csv)",
        csv_out, "hasil_spk_eldorado.csv", "text/csv",
        help="Download seluruh hasil rekomendasi SPK dalam format CSV"
    )

# ══ TAB 4 ══
with tab4:
    st.markdown('<div class="section-title">Evaluasi Kualitas Model K-Means Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Bagian ini menampilkan metrik evaluasi yang digunakan untuk mengukur seberapa baik model K-Means berhasil mengelompokkan produk. Dua metode utama yang digunakan adalah Elbow Method dan Silhouette Score.</div>', unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 2])

    with col_e1:
        sil_status = _sil_label(sil)
        sil_color  = _sil_color(sil)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:1rem">
            <div class="metric-label">Silhouette Score (k=4)</div>
            <div class="metric-value" style="color:{sil_color}; font-size:2.8rem">{sil:.3f}</div>
            <div class="metric-sub">{sil_status}</div>
            <div class="metric-desc">
                Silhouette Score mengukur seberapa baik setiap produk cocok dengan klasternya sendiri dibanding klaster lain.
                Interpretasi mengacu skala Kaufman &amp; Rousseeuw (1990):
                <b>0,71–1,00</b> struktur kuat · <b>0,51–0,70</b> memadai ·
                <b>0,26–0,50</b> lemah · <b>&lt;0,25</b> tidak berstruktur.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Profil Hasil Klaster</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Ringkasan karakteristik rata-rata tiap klaster. <b>Avg Frekuensi</b> = rata-rata berapa kali produk di kelompok itu dipesan; <b>Avg Revenue</b> = rata-rata pendapatan per produk; <b>Jumlah Produk</b> = banyaknya anggota kelompok.</div>', unsafe_allow_html=True)

        profile2 = product_base.groupby('Cluster').agg(
            Avg_Freq=('Frequency','mean'),
            Avg_Rev=('Total_Revenue','mean'),
            Jumlah=('Base_Name','count')
        ).round(2).reset_index()
        profile2['Klaster'] = profile2['Cluster'].map(lambda x: f"{CLUSTER_META[x]['emoji']} C{x} {CLUSTER_META[x]['name']}")
        profile2['Avg_Rev'] = profile2['Avg_Rev'].map(lambda x: f"${x:,.1f}")
        profile2 = profile2[['Klaster','Avg_Freq','Avg_Rev','Jumlah']]
        profile2.columns = ['Klaster','Avg Frekuensi','Avg Revenue','Jumlah Produk']
        st.dataframe(profile2, use_container_width=True, hide_index=True)

    with col_e2:
        st.markdown('<div class="section-title">Elbow Method — Penentuan Jumlah Klaster Optimal</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Elbow Method digunakan untuk menemukan jumlah klaster (k) yang paling optimal. Grafik menunjukkan nilai WCSS (Within-Cluster Sum of Squares) — semakin kecil WCSS, semakin padat klaster yang terbentuk. Titik "siku" atau perubahan drastis pada grafik menunjukkan nilai k yang optimal. Pada data ini, <b>k=4 dipilih</b> karena di situlah titik siku paling jelas terlihat.</div>', unsafe_allow_html=True)

        X2 = np.log1p(product_base[['Frequency','Total_Volume','Total_Revenue']].values)
        X2 = MinMaxScaler().fit_transform(X2)
        with st.spinner("Menghitung nilai WCSS untuk k=1 sampai k=10..."):
            wcss = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X2).inertia_
                    for k in range(1,11)]

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(1,11)), y=wcss,
            mode='lines+markers',
            line=dict(color='#6366f1', width=3, dash='dash'),
            marker=dict(size=10, color='#a5b4fc'),
            name='WCSS'
        ))
        fig_elbow.add_vline(
            x=4, line_dash="dot", line_color="#fde68a", line_width=2,
            annotation_text="k=4 dipilih sebagai optimal",
            annotation_font_color="#fde68a",
            annotation_font_size=13
        )
        fig_elbow.update_layout(
            paper_bgcolor='#1a1f2e', plot_bgcolor='#1a1f2e',
            font=dict(color='#c8d0e0', size=14),
            xaxis=dict(
                title='Jumlah Kelompok (K)',
                gridcolor='#2a3550', tickmode='linear',
                title_font=dict(size=15)
            ),
            yaxis=dict(
                title='WCSS — total jarak produk ke pusat kelompoknya',
                gridcolor='#2a3550',
                title_font=dict(size=15)
            ),
            margin=dict(l=0, r=0, t=20, b=0),
            height=400, showlegend=False
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.markdown('<div class="chart-caption">Cara membaca: sumbu tegak (WCSS) mengukur seberapa jauh rata-rata produk dari pusat kelompoknya — makin kecil makin padat. Menambah jumlah kelompok selalu menurunkan nilai ini, jadi yang dicari adalah titik di mana penurunannya mulai melandai. Setelah k=4, penurunannya sudah tidak berarti lagi.</div>', unsafe_allow_html=True)
