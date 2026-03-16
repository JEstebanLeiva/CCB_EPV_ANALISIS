import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import chi2_contingency, t as student_t
from scipy.stats import zscore


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="EPV 2024 — Región Metropolitana de Bogotá",
    layout="wide"
)

# =========================================================
# CONSTANTES
# =========================================================
MUN_COORDS = pd.DataFrame({
    "municipio": [
        "Bogotá","Chía","Cota","La Calera","Cajicá","Zipaquirá",
        "Tocancipá","Sopó","Tenjo","Sibaté","Fusagasugá","Silvania",
        "Chocontá","Guasca","Cáqueza","Fómeque","Ubaté"
    ],
    "lat": [
        4.711,4.865,4.805,4.919,4.919,5.022,4.970,4.905,4.873,
        4.487,4.337,4.398,5.143,4.863,4.411,4.484,5.315
    ],
    "lon": [
        -74.072,-73.926,-74.104,-73.970,-74.026,-74.000,-73.913,
        -73.948,-74.140,-74.259,-74.364,-74.490,-73.685,-73.877,
        -73.948,-73.898,-73.818
    ]
})

LIMITROFES = ["Chía", "Cota", "La Calera"]
SAB_CENTRO = ["Cajicá", "Zipaquirá", "Tocancipá", "Sopó", "Tenjo"]
OTROS_CUN = ["Sibaté", "Fusagasugá", "Silvania", "Chocontá", "Guasca", "Cáqueza", "Fómeque", "Ubaté"]

LABEL_MAP = {
    "VIC": "Victimización",
    "TESTIGO": "Testigo de delito",
    "HOGAR_VIC": "Víctima en hogar",
    "DENUNCIA_BIN": "Denunció el delito",
    "BARRIO_INSEG": "Barrio inseguro",
    "PERCEP_BARRIO": "Seguridad barrio (1-5)",
    "PERCEP_BOGOTA": "Seguridad Bogotá (1-5)",
    "PERCEP_NEG": "Seguridad negocios (1-5)",
    "PERCEP_NUM": "Seguridad TransMilenio (1-5)",
    "C_HURTO_PER": "Hurto a persona",
    "C_HURTO_RES": "Hurto a residencia",
    "C_LESIONES": "Lesiones personales",
    "C_HURTO_VEH": "Hurto de vehículo",
    "C_VIF_DELITO": "Violencia intrafamiliar",
    "C_VANDAL": "Vandalismo",
    "C_VIO_SEX": "Violencia sexual",
    "C_HURTO_BIC": "Hurto de bicicleta",
    "C_CIBERNETICO": "Delito cibernético",
    "C_EXTORSION": "Extorsión",
    "C_HURTO_EST": "Hurto a establecimiento",
    "C_AMENAZAS": "Amenazas",
    "CONF_POLICIA": "Calidad Policía (1-5)",
    "ATEN_POLICIA": "Atención Policía (1-5)",
    "CONF_123": "Atención línea 123 (1-5)",
    "CONF_DISTRI": "Relacionamiento Adm. Distrital (1-5)",
    "MEDIO_REDES": "Info redes sociales",
    "MEDIO_TV": "Info TV (noticieros)",
    "MEDIO_RADIO": "Info radio",
    "MEDIO_PRENSA": "Info periódicos",
    "MEDIO_FAMILIA": "Info familiares/conocidos",
    "MEDIO_VOZ": "Info voz a voz"
}

SECTIONS = {
    "Percepción seguridad": ["PERCEP_BARRIO", "PERCEP_BOGOTA", "PERCEP_NEG", "BARRIO_INSEG"],
    "Victimización y denuncia": ["VIC", "TESTIGO", "HOGAR_VIC", "DENUNCIA_BIN"],
    "Tipos de delito": [
        "C_HURTO_PER","C_HURTO_RES","C_LESIONES","C_HURTO_VEH",
        "C_VIF_DELITO","C_VANDAL","C_VIO_SEX","C_HURTO_BIC",
        "C_CIBERNETICO","C_EXTORSION","C_HURTO_EST","C_AMENAZAS"
    ],
    "Instituciones y Policía": ["CONF_POLICIA", "ATEN_POLICIA", "CONF_123", "CONF_DISTRI"],
    "Redes y medios": ["MEDIO_REDES", "MEDIO_TV", "MEDIO_RADIO", "MEDIO_PRENSA", "MEDIO_FAMILIA", "MEDIO_VOZ"]
}

GROUP_CHOICES = {
    "Género (Hombre/Mujer)": "GENERO",
    "Territorio": "TERRITORIO",
    "Rango de edad": "RANGO_EDAD",
    "Municipio": "MUNICIPIO"
}

GROUP_LABELS = {v: k for k, v in GROUP_CHOICES.items()}

CRIME_COLS = [
    "C_HURTO_PER","C_HURTO_RES","C_LESIONES","C_HURTO_VEH",
    "C_VIF_DELITO","C_VANDAL","C_VIO_SEX","C_HURTO_BIC",
    "C_CIBERNETICO","C_EXTORSION","C_HURTO_EST","C_AMENAZAS"
]

QUAD_COL = {
    "HH": "#c0392b",
    "LL": "#2980b9",
    "HL": "#f39c12",
    "LH": "#27ae60",
    "NS": "#aaaaaa"
}


# =========================================================
# HELPERS
# =========================================================
def lbl(v):
    return LABEL_MAP.get(v, v)


def to_numeric_safe(s):
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def bin_col(col):
    return (col.astype(str).str.strip() == "Sí").astype("Int64")


def wm(x, w):
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    ok = (~x.isna()) & (~w.isna()) & (w > 0)
    if ok.sum() < 3:
        return np.nan
    return np.average(x[ok], weights=w[ok])


def wm_safe(x, w):
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    ok = (~x.isna()) & (~w.isna()) & (w > 0)
    if ok.sum() < 5:
        return np.nan
    return np.average(x[ok], weights=w[ok])


def add_num_by_prefix(df, prefix, new_name):
    cols = [c for c in df.columns if str(c).strip().startswith(prefix)]
    if cols:
        col = cols[0]
        df = df.rename(columns={col: new_name})
        df[new_name] = to_numeric_safe(df[new_name])
    return df


# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def load_data(path="DF_EPV_version4.xlsx"):
    df = pd.read_excel(path, dtype=str)

    # Renombrar por posición (R 1-based -> Python 0-based)
    imap = {
        6: "SEXO",
        7: "ESTRATO",
        8: "RANGO_EDAD",
        21: "P102_BARRIO",
        23: "P106_CAMBIO",
        34: "P111_SEGURIDAD",
        70: "P121_TESTIGO",
        72: "PERCEP_NEG",
        75: "MEDIO_FAMILIA",
        76: "MEDIO_VOZ",
        77: "MEDIO_REDES",
        78: "MEDIO_TV",
        80: "MEDIO_RADIO",
        81: "MEDIO_PRENSA",
        83: "P203_VICTIMA",
        84: "P2141_DENUNCIA",
        86: "C_HURTO_PER",
        87: "C_HURTO_RES",
        88: "C_LESIONES",
        89: "C_HURTO_VEH",
        90: "C_VIF_DELITO",
        91: "C_VANDAL",
        92: "C_VIO_SEX",
        93: "C_HURTO_BIC",
        94: "C_CIBERNETICO",
        96: "C_EXTORSION",
        97: "C_HURTO_EST",
        103: "C_AMENAZAS",
        233: "FACTOR"
    }

    cols = list(df.columns)
    for pos_1b, new_name in imap.items():
        idx = pos_1b - 1
        if idx < len(cols):
            cols[idx] = new_name
    df.columns = cols

    # MUNICIPIO_*
    mun_cols = [c for c in df.columns if str(c).startswith("MUNICIPIO_")]
    if mun_cols:
        df = df.rename(columns={mun_cols[0]: "MUNICIPIO"})

    num_cols = [
        "SEXO","ESTRATO","P102_BARRIO","P106_CAMBIO","P111_SEGURIDAD","FACTOR",
        "PERCEP_NEG","MEDIO_FAMILIA","MEDIO_VOZ","MEDIO_REDES",
        "MEDIO_TV","MEDIO_RADIO","MEDIO_PRENSA"
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    if "P203_VICTIMA" in df.columns:
        df["VIC"] = bin_col(df["P203_VICTIMA"])

    if "P121_TESTIGO" in df.columns:
        df["TESTIGO"] = bin_col(df["P121_TESTIGO"])

    if "SEXO" in df.columns:
    sexo_num = pd.to_numeric(df["SEXO"], errors="coerce")
    df["GENERO"] = sexo_num.map({0: "Mujer", 1: "Hombre"})

    if "P111_SEGURIDAD" in df.columns:
    s = to_numeric_safe(df["P111_SEGURIDAD"])
    df["PERCEP_NUM"] = s
    df["PERCEP_CAT"] = np.select(
        [s <= 2, s == 3, s >= 4],
        ["Inseguro", "Neutro", "Seguro"],
        default=None
    )

    if "P2141_DENUNCIA" in df.columns:
    d = df["P2141_DENUNCIA"].astype(str).str.strip()
    df["DENUNCIA_BIN"] = np.select(
        [d == "Sí", d == "No"],
        [1, 0],
        default=np.nan
    )

    if "P102_BARRIO" in df.columns:
        df["BARRIO_INSEG"] = 1 - pd.to_numeric(df["P102_BARRIO"], errors="coerce")

    if "P106_CAMBIO" in df.columns:
    cambio = pd.to_numeric(df["P106_CAMBIO"], errors="coerce")
    df["CAMBIO_CAT"] = np.select(
        [cambio == 0, cambio == 1, cambio == 2],
        ["Empeoró", "Igual", "Mejoró"],
        default=None
    )

    p233_cols = [c for c in df.columns if str(c).startswith("P233_")]
    if p233_cols:
        df["CONV_CAT"] = df[p233_cols[0]].astype(str).str.strip()

    p230_cols = [c for c in df.columns if str(c).startswith("P230_")]
    if p230_cols:
        df["HOGAR_VIC"] = bin_col(df[p230_cols[0]])

    df = add_num_by_prefix(df, "P1021_", "PERCEP_BARRIO")
    df = add_num_by_prefix(df, "P1031_", "PERCEP_BOGOTA")
    df = add_num_by_prefix(df, "P449_", "CONF_DISTRI")
    df = add_num_by_prefix(df, "P308_", "CONF_123")
    df = add_num_by_prefix(df, "P4011_", "CONF_POLICIA")
    df = add_num_by_prefix(df, "P421_", "ATEN_POLICIA")

    for col in CRIME_COLS:
        if col in df.columns:
            v = df[col].astype(str).str.strip()
            df[col] = np.select(
                [v == "Sí", v == "No"],
                [1, 0],
                default=np.nan
            )

    if "MUNICIPIO" in df.columns:
        df["TERRITORIO"] = np.select(
            [
                df["MUNICIPIO"] == "Bogotá",
                df["MUNICIPIO"].isin(LIMITROFES),
                df["MUNICIPIO"].isin(SAB_CENTRO),
                df["MUNICIPIO"].isin(OTROS_CUN)
            ],
            ["Bogotá", "Limítrofes", "Sabana Centro", "Otros Cund."],
            default="Otro"
        )

    return df


# =========================================================
# ESTADÍSTICAS
# =========================================================
def w_chisq(df, v1, v2):
    if v1 not in df.columns or v2 not in df.columns:
        return None

    sub = df[[v1, v2]].dropna()
    if len(sub) < 20:
        return None

    tab = pd.crosstab(sub[v1].astype(str), sub[v2].astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None

    try:
        chi2, p, _, _ = chi2_contingency(tab)
        n = tab.values.sum()
        v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        return {"chi2": chi2, "p": p, "V": v, "n": n, "tab": tab}
    except Exception:
        return None


def w_pcorr(df, v1, v2):
    if not all(c in df.columns for c in [v1, v2, "FACTOR"]):
        return None

    sub = df[[v1, v2, "FACTOR"]].dropna()
    if len(sub) < 20:
        return None

    x = pd.to_numeric(sub[v1], errors="coerce")
    y = pd.to_numeric(sub[v2], errors="coerce")
    w = pd.to_numeric(sub["FACTOR"], errors="coerce")

    ok = (~x.isna()) & (~y.isna()) & (~w.isna())
    x, y, w = x[ok], y[ok], w[ok]

    if len(x) < 20 or w.sum() == 0:
        return None

    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cxy = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)

    if vx <= 0 or vy <= 0:
        return None

    r = cxy / np.sqrt(vx * vy)
    n = len(x)
    tstat = r * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - r**2))
    p = 2 * student_t.sf(abs(tstat), df=n - 2)

    return {"r": r, "p": p, "n": n}


def section_table(data, vars_, group_var):
    if group_var not in data.columns:
        return pd.DataFrame({"Mensaje": ["Variable de grupo no disponible en los datos"]})

    d = data.copy()
    d["FACTOR"] = pd.to_numeric(d["FACTOR"], errors="coerce")
    d[group_var] = d[group_var].astype(str)

    levels_g = sorted(d[group_var].dropna().unique().tolist())
    if len(levels_g) > 15:
        levels_g = levels_g[:15]

    rows = []

    for v in vars_:
        if v not in d.columns:
            continue

        xn = pd.to_numeric(d[v], errors="coerce")
        valid_x = xn.dropna()

        if valid_x.empty:
            continue

        is_bin = valid_x.min() >= 0 and valid_x.max() <= 1
        scale_v = 100 if is_bin else 1
        unit_lbl = "%" if is_bin else "media (1-5)"

        row = {"Variable": lbl(v), "Unidad": unit_lbl}
        row["Total"] = round(wm_safe(xn, d["FACTOR"]) * scale_v, 1)

        for g in levels_g:
            mask = d[group_var] == g
            row[g] = round(wm_safe(xn[mask], d.loc[mask, "FACTOR"]) * scale_v, 1)

        rows.append(row)

    if not rows:
        return pd.DataFrame({"Mensaje": ["Sin variables disponibles para esta sección"]})

    return pd.DataFrame(rows)


def make_ctable(data, row_var, grp_var):
    if row_var not in data.columns or grp_var not in data.columns or "FACTOR" not in data.columns:
        return None

    sub = data[[row_var, grp_var, "FACTOR"]].dropna().copy()
    if len(sub) < 20:
        return None

    xn = pd.to_numeric(sub[row_var], errors="coerce")
    is_bin = xn.notna().any() and (xn.dropna().min() >= 0) and (xn.dropna().max() <= 1) and (xn.dropna().nunique() <= 2)

    if is_bin:
        sub[row_var] = np.where(xn == 1, "Si", "No")
    else:
        sub[row_var] = sub[row_var].astype(str)

    sub[grp_var] = sub[grp_var].astype(str)
    sub["FACTOR"] = pd.to_numeric(sub["FACTOR"], errors="coerce")

    try:
        tab = pd.pivot_table(
            sub,
            index=row_var,
            columns=grp_var,
            values="FACTOR",
            aggfunc="sum",
            fill_value=0
        )

        if tab.shape[0] == 0 or tab.shape[1] == 0:
            return None

        pct = tab.div(tab.sum(axis=0), axis=1) * 100
        pct = pct.round(1)

        row_totals = (tab.sum(axis=1) / tab.values.sum() * 100).round(1)

        df_out = pct.copy()
        if df_out.shape[1] > 0:
            df_out = df_out[sorted(df_out.columns)]
        df_out["Total"] = row_totals
        df_out = df_out.reset_index().rename(columns={row_var: "Categoria"})

        r = w_chisq(data, grp_var, row_var)
        if r is not None:
            p_str = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            if r["V"] < 0.10:
                eff = "pequeño"
            elif r["V"] < 0.30:
                eff = "moderado"
            else:
                eff = "fuerte"
            chi_text = f"Chi2={r['chi2']:.1f} | p={p_str} | V={r['V']:.2f} (efecto {eff})"
        else:
            chi_text = "N insuficiente"

        return {"df": df_out, "chi": chi_text, "n": len(sub)}

    except Exception:
        return None


def all_corrs(df):
    cat_pairs = [
        ("GENERO","VIC"),("GENERO","TESTIGO"),("GENERO","BARRIO_INSEG"),
        ("GENERO","PERCEP_CAT"),("GENERO","DENUNCIA_BIN"),("GENERO","HOGAR_VIC"),
        ("RANGO_EDAD","VIC"),("RANGO_EDAD","PERCEP_CAT"),("RANGO_EDAD","BARRIO_INSEG"),
        ("TERRITORIO","VIC"),("TERRITORIO","PERCEP_CAT"),("TERRITORIO","TESTIGO"),
        ("TERRITORIO","HOGAR_VIC"),("TERRITORIO","BARRIO_INSEG"),("TERRITORIO","DENUNCIA_BIN"),
        ("PERCEP_CAT","VIC"),("PERCEP_CAT","TESTIGO"),("PERCEP_CAT","HOGAR_VIC"),
        ("VIC","DENUNCIA_BIN"),("CAMBIO_CAT","PERCEP_CAT"),("CAMBIO_CAT","VIC"),
        ("VIC","CONV_CAT"),("MUNICIPIO","VIC"),("MUNICIPIO","PERCEP_CAT")
    ]

    for crime in CRIME_COLS:
        cat_pairs += [
            ("GENERO", crime),
            ("TERRITORIO", crime),
            ("PERCEP_CAT", crime),
            ("RANGO_EDAD", crime)
        ]

    for mv in ["MEDIO_REDES", "MEDIO_TV", "MEDIO_RADIO", "MEDIO_PRENSA"]:
        cat_pairs += [
            ("GENERO", mv),
            ("TERRITORIO", mv),
            ("PERCEP_CAT", mv)
        ]

    for cv in ["CONF_POLICIA", "ATEN_POLICIA", "CONF_123", "CONF_DISTRI"]:
        cat_pairs += [
            ("TERRITORIO", cv),
            ("GENERO", cv)
        ]

    num_pairs = [
        ("PERCEP_BARRIO","VIC"),("PERCEP_BARRIO","BARRIO_INSEG"),("PERCEP_BARRIO","TESTIGO"),
        ("PERCEP_BOGOTA","VIC"),("PERCEP_BOGOTA","BARRIO_INSEG"),("PERCEP_BOGOTA","PERCEP_BARRIO"),
        ("PERCEP_NEG","VIC"),("PERCEP_NEG","PERCEP_BARRIO"),("PERCEP_NEG","PERCEP_BOGOTA"),
        ("ESTRATO","PERCEP_BARRIO"),("ESTRATO","PERCEP_BOGOTA"),("ESTRATO","VIC"),
        ("ESTRATO","BARRIO_INSEG"),("ESTRATO","DENUNCIA_BIN"),("ESTRATO","PERCEP_NEG"),
        ("CONF_POLICIA","VIC"),("CONF_POLICIA","PERCEP_BARRIO"),("CONF_POLICIA","DENUNCIA_BIN"),
        ("ATEN_POLICIA","VIC"),("ATEN_POLICIA","PERCEP_BARRIO"),
        ("CONF_123","VIC"),("CONF_DISTRI","VIC"),("CONF_DISTRI","PERCEP_BARRIO")
    ]

    rows = []

    for v1, v2 in cat_pairs:
        if v1 not in df.columns or v2 not in df.columns:
            continue
        r = w_chisq(df, v1, v2)
        if r is None:
            continue
        rows.append({
            "Variable1": v1,
            "Variable2": v2,
            "Tipo": "Chi²",
            "Estadístico": round(r["chi2"], 1),
            "Efecto": round(r["V"], 3),
            "P_valor": float(r["p"]),
            "N": int(r["n"])
        })

    for v1, v2 in num_pairs:
        if v1 not in df.columns or v2 not in df.columns:
            continue
        r = w_pcorr(df, v1, v2)
        if r is None:
            continue
        rows.append({
            "Variable1": v1,
            "Variable2": v2,
            "Tipo": "Pearson",
            "Estadístico": round(r["r"], 3),
            "Efecto": round(abs(r["r"]), 3),
            "P_valor": float(r["p"]),
            "N": int(r["n"])
        })

    if not rows:
        return pd.DataFrame(columns=["Variable1","Variable2","Tipo","Estadístico","Efecto","P_valor","N"])

    return pd.DataFrame(rows).sort_values("P_valor").reset_index(drop=True)


def mun_profile(data):
    all_vars = [
        "VIC","TESTIGO","BARRIO_INSEG","PERCEP_BARRIO","PERCEP_BOGOTA",
        "PERCEP_NEG","DENUNCIA_BIN","HOGAR_VIC",
        "C_HURTO_PER","C_HURTO_RES","C_LESIONES","C_HURTO_VEH","C_VIF_DELITO",
        "C_VANDAL","C_VIO_SEX","C_HURTO_BIC","C_CIBERNETICO","C_EXTORSION",
        "C_HURTO_EST","C_AMENAZAS","CONF_POLICIA","CONF_DISTRI","PERCEP_NUM",
        "MEDIO_REDES","MEDIO_TV","MEDIO_PRENSA"
    ]

    vars_ = [v for v in all_vars if v in data.columns]
    d = data.copy()
    d["FACTOR"] = pd.to_numeric(d["FACTOR"], errors="coerce")

    for v in vars_:
        d[v] = pd.to_numeric(d[v], errors="coerce")

    if "MUNICIPIO" not in d.columns:
        return pd.DataFrame()

    muns = sorted(d["MUNICIPIO"].dropna().unique().tolist())
    rows = []

    for m in muns:
        sub = d[d["MUNICIPIO"] == m].copy()
        w = sub["FACTOR"]
        row = {"MUNICIPIO": m, "N": w.sum(skipna=True)}
        for v in vars_:
            row[v] = wm(sub[v], w)
        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# MORAN / LISA
# =========================================================
def build_W(lats, lons):
    n = len(lats)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.sqrt((lats[i] - lats[j])**2 + (lons[i] - lons[j])**2)
                if d > 0:
                    W[i, j] = 1 / d
    rs = W.sum(axis=1)
    rs[rs == 0] = 1
    W = W / rs[:, None]
    return W


def moran_global(vals, W, nperm=499):
    vals = np.asarray(vals, dtype=float)
    ok = ~np.isnan(vals)
    if ok.sum() < 4:
        return {"I": np.nan, "p": np.nan}

    vals = vals[ok]
    W2 = W[np.ix_(ok, ok)]
    rs = W2.sum(axis=1)
    rs[rs == 0] = 1
    W2 = W2 / rs[:, None]

    z = zscore(vals, nan_policy="omit")
    z = np.nan_to_num(z)

    try:
        I = float((z.T @ W2 @ z) / len(vals))
    except Exception:
        return {"I": np.nan, "p": np.nan}

    perms = []
    for _ in range(nperm):
        zp = np.random.permutation(z)
        try:
            perms.append(float((zp.T @ W2 @ zp) / len(vals)))
        except Exception:
            pass

    perms = np.array(perms, dtype=float)
    perms = perms[np.isfinite(perms)]
    p = np.mean(np.abs(perms) >= np.abs(I)) if len(perms) > 0 else np.nan

    return {"I": I, "p": p}


def moran_lisa(vals, W):
    vals = np.asarray(vals, dtype=float)
    z = zscore(vals, nan_policy="omit")
    z = np.nan_to_num(z)
    lag_z = W @ z
    quad = np.select(
        [
            (z > 0) & (lag_z > 0),
            (z < 0) & (lag_z < 0),
            (z > 0) & (lag_z < 0),
            (z < 0) & (lag_z > 0),
        ],
        ["HH", "LL", "HL", "LH"],
        default="NS"
    )
    return pd.DataFrame({
        "z": z,
        "lag_z": lag_z,
        "Ii": z * lag_z,
        "quad": quad
    })


# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <div style="background:#1e3a5f;padding:14px 20px;border-radius:8px;margin-bottom:12px">
        <h2 style="color:white;margin:0">EPV 2024 — Región Metropolitana de Bogotá</h2>
        <p style="color:#d9e2ef;margin:6px 0 0 0">
            Tablas ponderadas · Correlaciones · Comparación territorial · Autocorrelación espacial
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Puedes dejar fijo el archivo o permitir carga manual
uploaded_file = st.sidebar.file_uploader("Carga el Excel EPV", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data("DF_EPV_version4.xlsx")

prof = mun_profile(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "Tablas por sección",
    "Correlaciones Chi²/Pearson",
    "Bogotá vs Municipios",
    "Moran & LISA"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    c1, c2 = st.columns([1, 3])

    with c1:
        seccion = st.selectbox("Sección", list(SECTIONS.keys()), index=0)
        grupo_label = st.selectbox("Agrupar por", list(GROUP_CHOICES.keys()), index=0)
        grupo = GROUP_CHOICES[grupo_label]
        territorio_tbl = st.selectbox(
            "Filtrar territorio",
            ["Todos", "Bogotá", "Limítrofes", "Sabana Centro", "Otros Cund."],
            index=0
        )

        st.caption(
            "Proporciones ponderadas con Factor de Expansión. % para variables Sí/No. "
            "Media (1-5) para escalas. p-valor y V Cramér: conteos sin ponderar."
        )

    with c2:
        d = df.copy()
        if territorio_tbl != "Todos" and "TERRITORIO" in d.columns:
            d = d[d["TERRITORIO"] == territorio_tbl].copy()

        st.subheader(f"{seccion} — {GROUP_LABELS.get(grupo, grupo)}")

        sec_tbl = section_table(d, SECTIONS[seccion], grupo)
        st.dataframe(sec_tbl, use_container_width=True)

        st.markdown("### Tablas individuales")

        for v in SECTIONS[seccion]:
            res = make_ctable(d, v, grupo)
            if res is None:
                continue

            st.markdown(f"**{lbl(v)} (n={res['n']})**")
            df_show = res["df"].copy()

            for col in df_show.columns:
                if col != "Categoria":
                    df_show[col] = df_show[col].map(lambda x: f"{x}%" if pd.notna(x) else "")

            st.dataframe(df_show, use_container_width=True)
            st.caption(res["chi"])


# =========================================================
# TAB 2
# =========================================================
with tab2:
    c1, c2 = st.columns([1, 3])

    with c1:
        territorio_corr = st.selectbox(
            "Territorio",
            ["Todos", "Bogotá", "Limítrofes", "Sabana Centro", "Otros Cund."],
            index=0,
            key="territorio_corr"
        )
        alpha = st.slider("α máximo", 0.001, 0.20, 0.05, step=0.005)
        efecto_min = st.slider("Efecto mínimo", 0.0, 0.5, 0.05, step=0.01)
        tipo = st.multiselect("Tipo", ["Chi²", "Pearson"], default=["Chi²", "Pearson"])

    with c2:
        d_corr = df.copy()
        if territorio_corr != "Todos" and "TERRITORIO" in d_corr.columns:
            d_corr = d_corr[d_corr["TERRITORIO"] == territorio_corr].copy()

        cors = all_corrs(d_corr)
        cors_filt = cors[
            (cors["P_valor"] <= alpha) &
            (cors["Efecto"] >= efecto_min) &
            (cors["Tipo"].isin(tipo))
        ].copy()

        st.dataframe(cors_filt, use_container_width=True)

        if not cors_filt.empty:
            idx = st.selectbox("Selecciona una fila para ver detalle", cors_filt.index)
            r = cors_filt.loc[idx]

            st.markdown(f"### Detalle: {r['Variable1']} × {r['Variable2']} ({r['Tipo']})")

            v1, v2 = r["Variable1"], r["Variable2"]

            if r["Tipo"] == "Chi²":
                sub = d_corr[[v1, v2, "FACTOR"]].dropna().copy()
                tab = pd.pivot_table(
                    sub,
                    index=v1,
                    columns=v2,
                    values="FACTOR",
                    aggfunc="sum",
                    fill_value=0
                )
                pct = tab.div(tab.sum(axis=1), axis=0) * 100
                pct = pct.reset_index().melt(id_vars=v1, var_name=v2, value_name="Pct")

                fig = px.density_heatmap(
                    pct,
                    x=v2,
                    y=v1,
                    z="Pct",
                    text_auto=".1f",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(
                    title=f"{v1} × {v2} — % por fila (ponderado FACTOR)"
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                sub = d_corr[[v1, v2]].dropna().copy()
                sub[v1] = pd.to_numeric(sub[v1], errors="coerce")
                sub[v2] = pd.to_numeric(sub[v2], errors="coerce")
                sub = sub.dropna()

                fig = px.scatter(
                    sub,
                    x=v1,
                    y=v2,
                    opacity=0.35,
                    title=f"r={r['Estadístico']:.3f} | p={r['P_valor']:.3g}"
                )
                st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 3
# =========================================================
with tab3:
    c1, c2 = st.columns([1, 3])

    with c1:
        avail_inds = [v for v in LABEL_MAP.keys() if v in prof.columns]
        ind_bar = st.selectbox(
            "Indicador",
            avail_inds,
            index=avail_inds.index("VIC") if "VIC" in avail_inds else 0,
            format_func=lbl
        )

        municipios_compare = [m for m in MUN_COORDS["municipio"].tolist() if m != "Bogotá"]
        mun2 = st.selectbox("Comparar vs Bogotá", municipios_compare, index=0)

    with c2:
        if ind_bar not in prof.columns:
            st.warning("Variable no disponible.")
        else:
            pd_bar = prof[["MUNICIPIO", ind_bar]].dropna().copy()
            pd_bar["clr"] = np.where(pd_bar["MUNICIPIO"] == "Bogotá", "Bogotá", "Municipio")
            pd_bar = pd_bar.sort_values(ind_bar)

            bv = pd_bar.loc[pd_bar["MUNICIPIO"] == "Bogotá", ind_bar]
            bv = bv.iloc[0] if not bv.empty else None

            fig_bar = px.bar(
                pd_bar,
                x=ind_bar,
                y="MUNICIPIO",
                orientation="h",
                color="clr",
                color_discrete_map={"Bogotá": "#c0392b", "Municipio": "#1e3a5f"},
                title=lbl(ind_bar)
            )

            if bv is not None:
                fig_bar.add_vline(x=bv, line_dash="dash", line_color="#c0392b")

            fig_bar.update_layout(yaxis_title="", xaxis_title="Media ponderada")
            st.plotly_chart(fig_bar, use_container_width=True)

            c21, c22 = st.columns([7, 5])

            with c21:
                ind_cols = [
                    v for v in [
                        "VIC","TESTIGO","BARRIO_INSEG","PERCEP_BARRIO","PERCEP_BOGOTA",
                        "DENUNCIA_BIN","HOGAR_VIC","C_HURTO_PER","C_HURTO_RES","C_LESIONES",
                        "C_HURTO_VEH","C_VIF_DELITO","C_VANDAL","C_VIO_SEX",
                        "CONF_POLICIA","CONF_DISTRI"
                    ] if v in prof.columns
                ]

                mat = prof[["MUNICIPIO"] + ind_cols].copy()
                for col in ind_cols:
                    mat[col] = zscore(mat[col], nan_policy="omit")

                long = mat.melt(id_vars="MUNICIPIO", var_name="Ind", value_name="Z")
                long["Label"] = long["Ind"].map(lbl)

                fig_heat = px.density_heatmap(
                    long,
                    x="Label",
                    y="MUNICIPIO",
                    z="Z",
                    color_continuous_scale=["#2980b9", "#f8f9fa", "#c0392b"],
                    title="Heatmap municipal — Z-scores"
                )
                fig_heat.update_layout(xaxis_title="", yaxis_title="")
                st.plotly_chart(fig_heat, use_container_width=True)

            with c22:
                radar_cols = [v for v in ["VIC","TESTIGO","BARRIO_INSEG","PERCEP_BARRIO","DENUNCIA_BIN","HOGAR_VIC"] if v in prof.columns]

                bog = prof.loc[prof["MUNICIPIO"] == "Bogotá", radar_cols]
                mun = prof.loc[prof["MUNICIPIO"] == mun2, radar_cols]

                if not bog.empty and not mun.empty:
                    bog_vals = bog.iloc[0].tolist()
                    mun_vals = mun.iloc[0].tolist()
                    theta = [lbl(v) for v in radar_cols]

                    fig_radar = go.Figure()

                    fig_radar.add_trace(go.Scatterpolar(
                        r=bog_vals + [bog_vals[0]],
                        theta=theta + [theta[0]],
                        fill="toself",
                        name="Bogotá",
                        line=dict(color="#c0392b")
                    ))

                    fig_radar.add_trace(go.Scatterpolar(
                        r=mun_vals + [mun_vals[0]],
                        theta=theta + [theta[0]],
                        fill="toself",
                        name=mun2,
                        line=dict(color="#1e3a5f")
                    ))

                    fig_radar.update_layout(
                        title=f"Bogotá vs {mun2}",
                        polar=dict(radialaxis=dict(visible=True))
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)


# =========================================================
# TAB 4
# =========================================================
with tab4:
    c1, c2 = st.columns([1, 3])

    with c1:
        avail_moran = [v for v in LABEL_MAP.keys() if v in prof.columns]
        ind_moran = st.selectbox(
            "Indicador",
            avail_moran,
            index=avail_moran.index("VIC") if "VIC" in avail_moran else 0,
            format_func=lbl
        )
        nperm = st.number_input("Permutaciones", min_value=99, max_value=2999, value=499, step=100)

        run = st.button("Calcular")

    with c2:
        if run:
            if ind_moran not in prof.columns:
                st.warning("Variable no disponible.")
            else:
                jd = prof.merge(MUN_COORDS, left_on="MUNICIPIO", right_on="municipio", how="left")
                jd = jd.dropna(subset=["lat", "lon", ind_moran]).copy()

                if len(jd) < 4:
                    st.warning("No hay suficientes datos para Moran/LISA.")
                else:
                    vals = pd.to_numeric(jd[ind_moran], errors="coerce").values
                    W = build_W(jd["lat"].values, jd["lon"].values)

                    glo = moran_global(vals, W, nperm=int(nperm))
                    loc = moran_lisa(vals, W)

                    lisa_df = pd.concat([jd.reset_index(drop=True), loc], axis=1)

                    st.markdown("### Moran Global")
                    if np.isfinite(glo["I"]):
                        st.write(f"**I = {glo['I']:.4f}**")
                        st.write(f"**p = {glo['p']:.4f}**")
                        if glo["p"] < 0.05:
                            st.success("Autocorrelación espacial significativa")
                        else:
                            st.error("Sin autocorrelación significativa")
                    else:
                        st.warning("No se pudo calcular Moran global.")

                    c21, c22 = st.columns(2)

                    with c21:
                        fig_moran = px.scatter(
                            lisa_df,
                            x="z",
                            y="lag_z",
                            color="quad",
                            text="MUNICIPIO",
                            color_discrete_map=QUAD_COL,
                            title=f"Diagrama de Moran — {lbl(ind_moran)}"
                        )

                        rng = [lisa_df["z"].min(), lisa_df["z"].max()]
                        if np.isfinite(glo["I"]):
                            fig_moran.add_trace(go.Scatter(
                                x=rng,
                                y=[rng[0] * glo["I"], rng[1] * glo["I"]],
                                mode="lines",
                                name=f"I={glo['I']:.4f}",
                                line=dict(dash="dash", color="#333")
                            ))

                        fig_moran.add_hline(y=0, line_color="#cccccc")
                        fig_moran.add_vline(x=0, line_color="#cccccc")
                        st.plotly_chart(fig_moran, use_container_width=True)

                    with c22:
                        fig_map = px.scatter_mapbox(
                            lisa_df,
                            lat="lat",
                            lon="lon",
                            color="quad",
                            text="MUNICIPIO",
                            color_discrete_map=QUAD_COL,
                            zoom=6.5,
                            height=450,
                            title="Mapa LISA"
                        )
                        fig_map.update_layout(mapbox_style="open-street-map")
                        st.plotly_chart(fig_map, use_container_width=True)

                    out_tbl = lisa_df[["MUNICIPIO", ind_moran, "z", "lag_z", "Ii", "quad"]].copy()
                    out_tbl = out_tbl.rename(columns={
                        "MUNICIPIO": "Municipio",
                        ind_moran: "Valor",
                        "quad": "Cuadrante"
                    })
                    for c in ["Valor", "z", "lag_z", "Ii"]:
                        out_tbl[c] = pd.to_numeric(out_tbl[c], errors="coerce").round(4)

                    out_tbl = out_tbl.sort_values("Ii", key=lambda s: s.abs(), ascending=False)
                    st.dataframe(out_tbl, use_container_width=True)
