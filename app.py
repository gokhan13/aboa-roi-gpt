
import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO

st.set_page_config(page_title="Ajans ROI Hesaplayıcı", layout="wide")

# -------------------------
# Helpers
# -------------------------
def currency_fmt(x):
    try:
        return f"{x:,.2f}"
    except:
        return x

def pct_fmt(x):
    try:
        return f"{x*100:.1f}%"
    except:
        return x

def safe_pct(x):
    # Accept 0-100 or 0-1; return 0-1
    if x is None or (isinstance(x, str) and x.strip()== ""):
        return None
    try:
        v = float(x)
        return v/100 if v > 1.0 else v
    except:
        return None

def break_even_roas(margin):
    m = safe_pct(margin)
    if not m or m == 0:
        return np.nan
    return 1.0/m

def calc_channels(df_channels, default_margin, default_returns, default_other_costs):
    if df_channels.empty:
        return df_channels, {}
    df = df_channels.copy()
    # Coerce types
    for col in ["Harcama","Gelir_Platform","IF","Kanal_Marj","Iade","Diger_Gider"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Compute
    def _if(v, fallback):
        return fallback if (pd.isna(v) or v=="" ) else v

    def _pct(v):
        x = safe_pct(v)
        return 0 if x is None else x

    df["Artimsal_Gelir"] = df.apply(lambda r: (r.get("Gelir_Platform",0) or 0) * (_if(r.get("IF"),1)), axis=1)
    df["Net_Gelir"] = df.apply(lambda r: (r["Artimsal_Gelir"]) * (1 - _pct(_if(r.get("Iade"), default_returns))), axis=1)
    df["Brut_Kar"] = df.apply(lambda r: (r["Net_Gelir"]) * (_pct(_if(r.get("Kanal_Marj"), default_margin))), axis=1)
    df["Toplam_Paz_Gideri"] = df.apply(lambda r: (r.get("Harcama",0) or 0) + _if(r.get("Diger_Gider"), default_other_costs), axis=1)
    df["Net_Kar"] = df["Brut_Kar"] - df["Toplam_Paz_Gideri"]
    df["ROAS"] = df.apply(lambda r: (r["Artimsal_Gelir"] / r["Harcama"]) if (r.get("Harcama",0) not in [0, None, np.nan]) else np.nan, axis=1)
    df["BreakEven_ROAS"] = df.apply(lambda r: break_even_roas(_if(r.get("Kanal_Marj"), default_margin)), axis=1)
    df["ROI"] = df.apply(lambda r: (r["Net_Kar"] / r["Toplam_Paz_Gideri"]) if (r["Toplam_Paz_Gideri"] not in [0, None, np.nan]) else np.nan, axis=1)

    totals = {
        "Harcama": df["Harcama"].sum(skipna=True),
        "Artimsal_Gelir": df["Artimsal_Gelir"].sum(skipna=True),
        "Net_Gelir": df["Net_Gelir"].sum(skipna=True),
        "Brut_Kar": df["Brut_Kar"].sum(skipna=True),
        "Toplam_Paz_Gideri": df["Toplam_Paz_Gideri"].sum(skipna=True),
        "Net_Kar": df["Net_Kar"].sum(skipna=True),
    }
    totals["MER"] = (totals["Artimsal_Gelir"] / totals["Harcama"]) if totals["Harcama"] else np.nan
    totals["ROI"] = (totals["Net_Kar"] / totals["Toplam_Paz_Gideri"]) if totals["Toplam_Paz_Gideri"] else np.nan
    totals["BreakEven_ROAS"] = break_even_roas(default_margin)
    return df, totals

def ecom_metrics(inp):
    # inp keys: siparis, aov, marj, iade, paz, diger
    aov = float(inp.get("aov") or 0)
    sip = float(inp.get("siparis") or 0)
    marj = safe_pct(inp.get("marj"))
    iade = safe_pct(inp.get("iade"))
    paz = float(inp.get("paz") or 0)
    diger = float(inp.get("diger") or 0)
    if marj is None: marj = safe_pct(inp.get("def_marj"))
    if iade is None: iade = safe_pct(inp.get("def_iade"))
    if np.isnan(diger) or diger==0:
        if (inp.get("def_diger") not in [None,""]):
            diger = float(inp.get("def_diger"))
    toplam_gelir = sip * aov
    net_gelir = toplam_gelir * (1 - (iade or 0))
    brut_kar = net_gelir * (marj or 0)
    be_roas = break_even_roas(marj)
    mer = (toplam_gelir / (paz + diger)) if (paz + diger) else np.nan
    roi = ((brut_kar - (paz + diger)) / (paz + diger)) if (paz + diger) else np.nan
    return dict(toplam_gelir=toplam_gelir, net_gelir=net_gelir, brut_kar=brut_kar, be_roas=be_roas, mer=mer, roi=roi)

def leadgen_metrics(inp):
    # inp keys: trafik, cpc, lead_pct, mql_pct, sql_pct, close_pct, deal, marj, paz
    trafik = float(inp.get("trafik") or 0)
    cpc = float(inp.get("cpc") or 0)
    lead_pct = safe_pct(inp.get("lead_pct")) or 0
    mql_pct = safe_pct(inp.get("mql_pct")) or 0
    sql_pct = safe_pct(inp.get("sql_pct")) or 0
    close_pct = safe_pct(inp.get("close_pct")) or 0
    deal = float(inp.get("deal") or 0)
    marj = safe_pct(inp.get("marj"))
    if marj is None: marj = safe_pct(inp.get("def_marj"))
    paz = trafik * cpc
    leads = trafik * lead_pct
    mql = leads * mql_pct
    sql = mql * sql_pct
    sales = sql * close_pct
    gelir = sales * deal
    brut_kar = gelir * (marj or 0)
    cac = (paz / sales) if sales else np.nan
    roi = ((brut_kar - paz) / paz) if paz else np.nan
    mer = (gelir / paz) if paz else np.nan
    return dict(paz=paz, leads=leads, mql=mql, sql=sql, sales=sales, gelir=gelir, brut_kar=brut_kar, cac=cac, roi=roi, mer=mer)

def saas_metrics(inp):
    # inp keys: arpu, churn, marj, cac, active
    arpu = float(inp.get("arpu") or 0)
    churn = safe_pct(inp.get("churn")) or 0
    marj = safe_pct(inp.get("marj"))
    if marj is None: marj = safe_pct(inp.get("def_marj"))
    cac = float(inp.get("cac") or 0)
    active = float(inp.get("active") or 0)
    gelir = arpu * active
    ltv = (arpu * (marj or 0) / churn) if churn else np.nan
    ltv_cac = (ltv / cac) if cac else np.nan
    aylik_kar = arpu * (marj or 0)
    payback = (cac / aylik_kar) if aylik_kar else np.nan
    return dict(gelir=gelir, ltv=ltv, ltv_cac=ltv_cac, aylik_kar=aylik_kar, payback=payback)

# -------------------------
# State Defaults
# -------------------------
if "settings" not in st.session_state:
    st.session_state["settings"] = dict(
        currency="TRY",
        period_start="",
        period_end="",
        def_margin=0.30,
        def_returns=0.05,
        def_other_costs=0.0,
        consent_flag=False,
        model_choice="Platform Atfı",
        target_mer=4.0,
        target_roi=0.30
    )
if "channels" not in st.session_state:
    st.session_state["channels"] = pd.DataFrame([
        {"Kanal":"Meta","Harcama":0,"Gelir_Platform":0,"IF":1.0,"Kanal_Marj":"","Iade":"","Diger_Gider":""},
        {"Kanal":"Google","Harcama":0,"Gelir_Platform":0,"IF":1.0,"Kanal_Marj":"","Iade":"","Diger_Gider":""},
        {"Kanal":"TikTok","Harcama":0,"Gelir_Platform":0,"IF":1.0,"Kanal_Marj":"","Iade":"","Diger_Gider":""}
    ])

settings = st.session_state["settings"]

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Ayarlar")
    settings["currency"] = st.text_input("Para Birimi", settings["currency"], key="sb_currency4")
    settings["period_start"] = st.text_input("Dönem Başlangıç (YYYY-MM-DD)", settings["period_start"], key="sb_start4")
    settings["period_end"] = st.text_input("Dönem Bitiş (YYYY-MM-DD)", settings["period_end"], key="sb_end4")
    settings["def_margin"] = st.number_input("Varsayılan Brüt Marj %", min_value=0.0, max_value=100.0, value=float(settings["def_margin"]*100), step=1.0, key="sb_def_margin4")/100.0
    settings["def_returns"] = st.number_input("Varsayılan İade %", min_value=0.0, max_value=100.0, value=float(settings["def_returns"]*100), step=0.5, key="sb_def_returns4")/100.0
    settings["def_other_costs"] = st.number_input("Varsayılan Diğer Pazarlama Giderleri", min_value=0.0, value=float(settings["def_other_costs"]), step=100.0, key="sb_def_other4")
    settings["consent_flag"] = st.checkbox("Modelleme/Ölçüm Bayrağı (Consent/iOS etkisi)", value=settings["consent_flag"], key="sb_consent4")
    settings["model_choice"] = st.selectbox("Atıf/Model Seçimi", ["Platform Atfı","Artımsal","MMM"], index=["Platform Atfı","Artımsal","MMM"].index(settings["model_choice"]), key="sb_model4")
    settings["target_mer"] = st.number_input("Hedef MER", min_value=0.0, value=float(settings["target_mer"]), step=0.1, key="sb_target_mer4")
    settings["target_roi"] = st.number_input("Hedef ROI %", min_value=0.0, max_value=100.0, value=float(settings["target_roi"]*100), step=5.0, key="sb_target_roi4")/100.0
    st.divider()
    st.caption("Veri Dışa/İçe Aktarma")
    export = dict(settings=settings, channels=st.session_state["channels"].to_dict(orient="list"))
    st.download_button("Yapılandırmayı İndir (JSON)", data=json.dumps(export, ensure_ascii=False, indent=2), file_name="roi_config.json", key="sb_download4")
    upl = st.file_uploader("Yapılandırma İçeri Aktar (JSON)", type=["json"], key="sb_upload4")
    if upl:
        try:
            data = json.load(upl)
            st.session_state["settings"] = data.get("settings", settings)
            ch = pd.DataFrame(data.get("channels", {}))
            if not ch.empty:
                st.session_state["channels"] = ch
            st.success("Yapılandırma içe aktarıldı.")
        except Exception as e:
            st.error(f"İçe aktarma hatası: {e}")

st.title("Ajans ROI Hesaplayıcı")
st.write("**Kapsam:** Kanal bazlı ROI, e-ticaret / lead-gen / SaaS metrikleri ve senaryo simülasyonu.")

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_channels, tab_ecom, tab_lead, tab_saas, tab_scenario = st.tabs(
    ["Genel Bakış", "Kanallar", "E-Ticaret", "Lead-Gen", "SaaS", "Senaryo Simülatörü"]
)

with tab_channels:
    st.subheader("Kanal Bazlı Girdiler")
    st.caption("Not: Boş marj/iade alanları Ayarlar’daki varsayılanlarla doldurulur.")
    edited = st.data_editor(
        st.session_state["channels"],
        num_rows="dynamic",
        use_container_width=True,
        key="ch_editor4",
        column_config={
            "Kanal": st.column_config.TextColumn("Kanal"),
            "Harcama": st.column_config.NumberColumn("Harcama", step=100.0, format="%.2f"),
            "Gelir_Platform": st.column_config.NumberColumn("Gelir_Platform", step=100.0, format="%.2f"),
            "IF": st.column_config.NumberColumn("IF (Artımsal Katsayı)", help="Platform gelirini artımsala kalibre edin; deney/MMM sonuçlarına göre", step=0.05, format="%.2f"),
            "Kanal_Marj": st.column_config.NumberColumn("Kanal Brüt Marj % (ops.)", help="Boş kalırsa Ayarlar’daki marj kullanılır", step=1.0, format="%.2f"),
            "Iade": st.column_config.NumberColumn("İade % (ops.)", step=0.5, format="%.2f"),
            "Diger_Gider": st.column_config.NumberColumn("Diğer Paz. Gideri (ops.)", step=100.0, format="%.2f"),
        }
    )
    st.session_state["channels"] = edited

    df_calc, totals = calc_channels(
        st.session_state["channels"],
        settings["def_margin"],
        settings["def_returns"],
        settings["def_other_costs"]
    )

    if not df_calc.empty:
        st.markdown("**Hesaplanan Metrikler (Kanal Tablosu)**")
        show = df_calc.copy()
        show["ROAS"] = show["ROAS"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}x")
        show["BreakEven_ROAS"] = show["BreakEven_ROAS"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}x")
        show["ROI"] = show["ROI"].apply(lambda v: "" if pd.isna(v) else pct_fmt(v))
        show["Brut_Kar"] = show["Brut_Kar"].apply(lambda v: currency_fmt(v))
        show["Net_Kar"] = show["Net_Kar"].apply(lambda v: currency_fmt(v))
        show["Toplam_Paz_Gideri"] = show["Toplam_Paz_Gideri"].apply(lambda v: currency_fmt(v))
        show["Net_Gelir"] = show["Net_Gelir"].apply(lambda v: currency_fmt(v))
        show["Artimsal_Gelir"] = show["Artimsal_Gelir"].apply(lambda v: currency_fmt(v))
        st.dataframe(show, use_container_width=True)

        st.markdown("**Toplamlar**")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Toplam Harcama", currency_fmt(totals["Harcama"]))
        c2.metric("Artımsal Gelir", currency_fmt(totals["Artimsal_Gelir"]))
        c3.metric("Net Gelir", currency_fmt(totals["Net_Gelir"]))
        c4.metric("Brüt Kâr", currency_fmt(totals["Brut_Kar"]))
        c5.metric("Toplam Paz. Gideri", currency_fmt(totals["Toplam_Paz_Gideri"]))
        c6.metric("Net Kâr", currency_fmt(totals["Net_Kar"]))
        c7.metric("MER", "" if pd.isna(totals["MER"]) else f"{totals['MER']:.2f}x")
        c1, c2 = st.columns(2)
        c1.metric("ROI %", "" if pd.isna(totals["ROI"]) else pct_fmt(totals["ROI"]))
        c2.metric("Break-even ROAS", "" if pd.isna(totals["BreakEven_ROAS"]) else f"{totals['BreakEven_ROAS']:.2f}x")

with tab_ecom:
    st.subheader("E-Ticaret")
    siparis = st.number_input("Sipariş Sayısı", min_value=0, value=1000, step=10, key="ecom_siparis4")
    aov = st.number_input("Ortalama Sepet (AOV)", min_value=0.0, value=800.0, step=10.0, key="ecom_aov4")
    marj = st.number_input("Brüt Marj %", min_value=0.0, max_value=100.0, value=float(settings["def_margin"]*100), step=1.0, key="ecom_marj4")
    iade = st.number_input("İade Oranı %", min_value=0.0, max_value=100.0, value=float(settings["def_returns"]*100), step=0.5, key="ecom_iade4")
    paz = st.number_input("Pazarlama Harcaması", min_value=0.0, value=100000.0, step=1000.0, key="ecom_paz4")
    diger = st.number_input("Diğer Paz. Giderleri", min_value=0.0, value=float(settings["def_other_costs"]), step=1000.0, key="ecom_diger4")
    res = ecom_metrics(dict(
        siparis=siparis, aov=aov, marj=marj, iade=iade, paz=paz, diger=diger,
        def_marj=settings["def_margin"], def_iade=settings["def_returns"], def_diger=settings["def_other_costs"]
    ))
    st.markdown("**Sonuçlar**")
    c1,c2,c3 = st.columns(3)
    c1.metric("Toplam Gelir", currency_fmt(res["toplam_gelir"]))
    c2.metric("Net Gelir", currency_fmt(res["net_gelir"]))
    c3.metric("Brüt Kâr", currency_fmt(res["brut_kar"]))
    c1,c2,c3 = st.columns(3)
    c1.metric("Break-even ROAS", "" if np.isnan(res["be_roas"]) else f"{res['be_roas']:.2f}x")
    c2.metric("MER", "" if np.isnan(res["mer"]) else f"{res['mer']:.2f}x")
    c3.metric("ROI %", "" if np.isnan(res["roi"]) else pct_fmt(res["roi"]))

with tab_lead:
    st.subheader("Lead-Gen")
    trafik = st.number_input("Trafik (tıklama)", min_value=0, value=20000, step=100, key="lead_trafik4")
    cpc = st.number_input("CPC", min_value=0.0, value=2.50, step=0.1, key="lead_cpc4")
    lead_pct = st.number_input("Lead Oranı %", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="lead_lead_pct4")
    mql_pct = st.number_input("MQL Oranı %", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key="lead_mql_pct4")
    sql_pct = st.number_input("SQL Oranı %", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key="lead_sql_pct4")
    close_pct = st.number_input("Kapanış Oranı %", min_value=0.0, max_value=100.0, value=30.0, step=0.5, key="lead_close_pct4")
    deal = st.number_input("Ortalama İş Hacmi (TL)", min_value=0.0, value=15000.0, step=500.0, key="lead_deal4")
    marj = st.number_input("Brüt Marj %", min_value=0.0, max_value=100.0, value=float(settings["def_margin"]*100), step=1.0, key="lead_marj4")
    res = leadgen_metrics(dict(
        trafik=trafik, cpc=cpc, lead_pct=lead_pct, mql_pct=mql_pct, sql_pct=sql_pct, close_pct=close_pct,
        deal=deal, marj=marj, def_marj=settings["def_margin"]
    ))
    st.markdown("**Sonuçlar**")
    c1,c2,c3 = st.columns(3)
    c1.metric("Pazarlama Harcaması", currency_fmt(res["paz"]))
    c2.metric("Satış", f"{res['sales']:.1f}")
    c3.metric("Gelir", currency_fmt(res["gelir"]))
    c1,c2,c3 = st.columns(3)
    c1.metric("Brüt Kâr", currency_fmt(res["brut_kar"]))
    c2.metric("CAC", "" if np.isnan(res["cac"]) else currency_fmt(res["cac"]))
    c3.metric("ROI %", "" if np.isnan(res["roi"]) else pct_fmt(res["roi"]))
    c4,c5 = st.columns(2)
    c4.metric("MER", "" if np.isnan(res["mer"]) else f"{res['mer']:.2f}x")

with tab_saas:
    st.subheader("SaaS / Abonelik")
    arpu = st.number_input("ARPU (Aylık)", min_value=0.0, value=250.0, step=10.0, key="saas_arpu4")
    churn = st.number_input("Aylık Churn %", min_value=0.0, max_value=100.0, value=4.0, step=0.5, key="saas_churn4")
    marj = st.number_input("Brüt Marj %", min_value=0.0, max_value=100.0, value=float(settings["def_margin"]*100), step=1.0, key="saas_marj4")
    cac = st.number_input("CAC", min_value=0.0, value=600.0, step=10.0, key="saas_cac4")
    active = st.number_input("Aktif Müşteri Sayısı", min_value=0, value=1200, step=10, key="saas_active4")
    res = saas_metrics(dict(arpu=arpu, churn=churn, marj=marj, cac=cac, active=active, def_marj=settings["def_margin"]))
    st.markdown("**Sonuçlar**")
    c1,c2,c3 = st.columns(3)
    c1.metric("Toplam Gelir (Aylık)", currency_fmt(res["gelir"]))
    c2.metric("LTV", "" if np.isnan(res["ltv"]) else currency_fmt(res["ltv"]))
    c3.metric("LTV:CAC", "" if np.isnan(res["ltv_cac"]) else f"{res['ltv_cac']:.2f}x")
    c1,c2 = st.columns(2)
    c1.metric("Aylık Müşteri Kârı", "" if np.isnan(res["aylik_kar"]) else currency_fmt(res["aylik_kar"]))
    c2.metric("CAC Payback (Ay)", "" if np.isnan(res["payback"]) else f"{res['payback']:.1f}")

with tab_overview:
    st.subheader("Genel Bakış & KPI Kartları")
    df_calc, totals = calc_channels(
        st.session_state["channels"],
        settings["def_margin"],
        settings["def_returns"],
        settings["def_other_costs"]
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Harcama", currency_fmt(totals.get("Harcama",0)))
    c2.metric("Artımsal Gelir", currency_fmt(totals.get("Artimsal_Gelir",0)))
    c3.metric("Net Kâr", currency_fmt(totals.get("Net_Kar",0)))
    c4, c5, c6 = st.columns(3)
    c4.metric("MER", "" if np.isnan(totals.get("MER",np.nan)) else f"{totals['MER']:.2f}x",
              delta=None, delta_color="normal")
    roi_val = totals.get("ROI", np.nan)
    target_roi = settings.get("target_roi", 0.3)
    mer_val = totals.get("MER", np.nan)
    target_mer = settings.get("target_mer", 4.0)
    c5.metric("ROI %", "" if np.isnan(roi_val) else pct_fmt(roi_val),
              delta=None if np.isnan(roi_val) else ("✓" if roi_val >= target_roi else "▼"))
    c6.metric("Break-even ROAS", "" if np.isnan(totals.get("BreakEven_ROAS",np.nan)) else f"{totals['BreakEven_ROAS']:.2f}x",
              delta=None if np.isnan(mer_val) else ("Hedef Üstü" if mer_val >= target_mer else "Hedef Altı"))

    if not df_calc.empty:
        st.markdown("**Kanal Karşılaştırması** (ROAS vs Break-even ROAS)")
        comp = df_calc[["Kanal","ROAS","BreakEven_ROAS","ROI","Net_Kar"]].copy()
        comp["ROAS"] = comp["ROAS"].apply(lambda v: np.nan if pd.isna(v) else float(v))
        comp["BreakEven_ROAS"] = comp["BreakEven_ROAS"].apply(lambda v: np.nan if pd.isna(v) else float(v))
        comp["ROI %"] = comp["ROI"].apply(lambda v: np.nan if pd.isna(v) else float(v)*100)
        comp["Net_Kar"] = comp["Net_Kar"].astype(float)
        st.dataframe(comp, use_container_width=True)

with tab_scenario:
    st.subheader("Senaryo Simülatörü")
    st.caption("Toplam rakamlar üzerinde ETKİ analizi: Bütçe, AOV, Marj kaydırıcıları artımsal gelir/kâr beklentisini ne yönde değiştirir?")
    budge_adj = st.slider("Bütçe Değişimi %", -50, 100, 0, step=5, key="sc_budge4")
    aov_adj = st.slider("AOV Değişimi % (E-ticaret)", -50, 100, 0, step=5, key="sc_aov4")
    margin_adj = st.slider("Marj Değişimi (puan)", -20, 20, 0, step=1, key="sc_margin4")
    # Apply
    df = st.session_state["channels"].copy()
    if not df.empty:
        df["Harcama"] = pd.to_numeric(df["Harcama"], errors="coerce").fillna(0) * (1 + budge_adj/100.0)
        df["Gelir_Platform"] = pd.to_numeric(df["Gelir_Platform"], errors="coerce").fillna(0) * (1 + aov_adj/100.0)
        # Shift margins
        adj_margin = max(0.0, min(1.0, settings["def_margin"] + margin_adj/100.0))
        df_calc, totals = calc_channels(df, adj_margin, settings["def_returns"], settings["def_other_costs"])
        st.markdown("**Yeni Toplamlar**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam Harcama", currency_fmt(totals["Harcama"]))
        c2.metric("Artımsal Gelir", currency_fmt(totals["Artimsal_Gelir"]))
        c3.metric("Net Kâr", currency_fmt(totals["Net_Kar"]))
        c4, c5, c6 = st.columns(3)
        c4.metric("MER", "" if np.isnan(totals["MER"]) else f"{totals['MER']:.2f}x")
        c5.metric("ROI %", "" if np.isnan(totals["ROI"]) else pct_fmt(totals["ROI"]))
        c6.metric("Break-even ROAS", "" if np.isnan(totals["BreakEven_ROAS"]) else f"{totals['BreakEven_ROAS']:.2f}x")
    else:
        st.info("Önce Kanallar tablosuna veri giriniz.")
