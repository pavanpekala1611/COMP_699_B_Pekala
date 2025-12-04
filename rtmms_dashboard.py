import json, time, uuid
import numpy as np, pandas as pd
from io import StringIO
from pathlib import Path
from collections import deque
from datetime import datetime, UTC
import streamlit as st

st.set_page_config(page_title="RTMMS Dashboard", layout="wide")

# ========= Simple Authentication (User/Admin login) =========

USERS = {
    "admin": {
        "password": "rtmms_admin",         
        "full_name": "Admin User",
        "role": "admin",
    },
    "amy_09": {
        "password": "rtmms_amy",         
        "full_name": "Amy Smith",
        "role": "user",
    },
    "roger_98": {
        "password": "rtmms_roger",
        "full_name": "Roger Adams",
        "role": "user",
    },
}


def _find_by_full_name(full_name: str):
    """Return (username, user_info) given a full name, or (None, None)."""
    full_name_norm = full_name.strip().lower()
    for username, info in USERS.items():
        if info["full_name"].strip().lower() == full_name_norm:
            return username, info
    return None, None


def login_flow():
    """
    Called ONLY when role == 'Care Team'.

    - If already logged in -> show 'Logged in as' + logout in sidebar, and return user.
    - If not logged in -> show login / forgot username / forgot password UI and stop.
    """
    # Already logged in?
    if "auth_user" in st.session_state:
        user = st.session_state["auth_user"]
        # Small sidebar status + logout
        with st.sidebar:
            st.markdown(
                f"**Logged in as:** {user['full_name']} "
                f"(_{user['username']}_, {user['role']})"
            )
            if st.button("Log out"):
                st.session_state.pop("auth_user", None)
                st.rerun()
        return user

    # Not logged in yet → show login screen and stop app below.
    st.title("RTMMS Login")

    tab_login, tab_forgot_user, tab_forgot_pass = st.tabs(
        ["Login", "I forgot my username", "I forgot my password"]
    )

    # ---------- Normal login ----------
    with tab_login:
        login_type = st.radio("Login as", ["User", "Admin"], horizontal=True)
        expected_role = "admin" if login_type == "Admin" else "user"

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Sign in"):
            user_info = USERS.get(username)
            if not user_info:
                st.error("Unknown username.")
            elif user_info["password"] != password:
                st.error("Incorrect password.")
            elif user_info["role"] != expected_role:
                st.error(f"This account is not a {login_type} account.")
            else:
                st.success("Login successful ✅")
                st.session_state["auth_user"] = {
                    "username": username,
                    "full_name": user_info["full_name"],
                    "role": user_info["role"],
                }
                st.rerun()

    # ---------- Forgot username ----------
    with tab_forgot_user:
        st.write("If you forgot your username, tell me your **full name**.")
        full_name = st.text_input("Full name")

        if st.button("Find my username"):
            if not full_name.strip():
                st.error("Please enter your full name.")
            else:
                username, info = _find_by_full_name(full_name)
                if username:
                    st.success(f"Your username is: **{username}**")
                else:
                    st.error("No user found with that full name.")

    # ---------- Forgot password ----------
    with tab_forgot_pass:
        st.write("If you forgot your password, enter your **username** and **full name**.")
        uname = st.text_input("Username (for password recovery)")
        full_name_pw = st.text_input("Full name (for password recovery)")

        if st.button("Help me with my password"):
            user_info = USERS.get(uname)
            if not user_info:
                st.error("No user found with that username.")
            else:
                username, info = _find_by_full_name(full_name_pw)
                if username is None or username != uname:
                    st.error("Username and full name do not match.")
                else:
                    # Simple: just show the password directly (demo only).
                    st.success(f"Your password is: **{user_info['password']}**")

    # Stop the rest of the app until user logs in
    st.stop()


# ========= Fixed live dataset path (hidden in UI) =========
PREFERRED_MERGED = Path('/Users/pavan/Documents/Capstone_Project/Dataset/all_merged.csv')

@st.cache_resource(show_spinner=False)
def find_all_merged_candidates() -> list[Path]:
    docs = Path.home() / "Documents"
    out = []
    try:
        for p in docs.rglob("all_merged.csv"):
            out.append(p)
    except Exception:
        pass
    return sorted(set(out), key=lambda p: (len(str(p)), str(p)))

def resolve_merged_path() -> Path | None:
    if PREFERRED_MERGED.exists():
        return PREFERRED_MERGED
    cands = find_all_merged_candidates()
    return cands[0] if cands else None

RESOLVED_LIVE_PATH = resolve_merged_path()
MERGED_PATH_TEXT = str(RESOLVED_LIVE_PATH) if RESOLVED_LIVE_PATH else str(PREFERRED_MERGED)

# ========= Helpers =========
def resident_label(idx: int) -> str: return f"Resident {idx + 1}"

def infer_columns(df: pd.DataFrame):
    def pick(df, cands):
        cols = [c for c in df.columns for tok in cands if tok in c.lower()]
        return cols[0] if cols else None
    ts = "TIME" if "TIME" in df.columns else pick(df, ["timestamp","time","ts","datetime","date"])
    prefixes = ["BELT","NECK","TORS","PCKT","WRST"]
    ax = ay = az = device = None
    for p in prefixes:
        cx, cy, cz = f"{p}_ACC_X", f"{p}_ACC_Y", f"{p}_ACC_Z"
        if all(c in df.columns for c in (cx, cy, cz)):
            ax, ay, az, device = cx, cy, cz, p
            break
    if not ax:
        ax = pick(df, ["_acc_x","accx","acc_x","acc x","ax","x"])
        ay = pick(df, ["_acc_y","accy","acc_y","acc y","ay","y"])
        az = pick(df, ["_acc_z","accz","acc_z","acc z","az","z"])
    hr   = pick(df, ["hr","heart","bpm"])
    spo2 = pick(df, ["spo2","ox","oxygen"])
    rr   = pick(df, ["resp","rr","breath"])
    return {"ax": ax, "ay": ay, "az": az, "ts": ts, "hr": hr, "spo2": spo2, "rr": rr, "device": device}

def classify_table(df: pd.DataFrame):
    cols = infer_columns(df)
    imu_ok = all(cols.get(k) for k in ("ax","ay","az"))
    vit_ok = any([cols.get("hr"), cols.get("spo2"), cols.get("rr")])
    if imu_ok: return "imu", cols
    if vit_ok: return "vitals", cols
    return "generic", cols

def sliding_windows(arr: np.ndarray, win: int, hop: int):
    for start in range(0, len(arr) - win + 1, hop):
        yield start, arr[start:start+win]

def vitals_sim(seed=None):
    if seed is not None: np.random.seed(seed)
    hr = 72 + np.random.randn()*2.0
    spo2 = 97 + np.random.randn()*0.4
    rr = 16 + np.random.randn()*1.0
    return int(np.clip(hr,45,150)), int(np.clip(spo2,85,100)), int(np.clip(rr,8,28))

def fall_score(window_mag: np.ndarray):
    peak = float(np.nanmax(window_mag)); std = float(np.nanstd(window_mag))
    if peak > 5.0: peak /= 9.81; std /= 9.81
    s = 0.0
    s += np.clip((peak - 1.8)/(3.0 - 1.8), 0, 1)*0.7
    s += np.clip((std  - 0.15)/(0.6 - 0.15), 0, 1)*0.3
    return float(np.clip(s, 0, 1))

def parse_upload(file):
    name = getattr(file, "name", "uploaded")
    suf = name.split(".")[-1].lower()
    if suf == "csv": return pd.read_csv(file), name
    if suf in ("xlsx","xls"): return pd.read_excel(file), name
    if suf == "json":
        try:
            data = json.loads(file.read())
            if isinstance(data, list): return pd.DataFrame(data), name
            if isinstance(data, dict): return pd.DataFrame(data), name
            raise ValueError("Unsupported JSON structure.")
        finally:
            file.seek(0)
    try:
        file.seek(0); txt = file.read()
        if isinstance(txt, bytes): txt = txt.decode("utf-8", errors="ignore")
        return pd.read_csv(StringIO(txt)), name
    except Exception as e:
        raise ValueError(f"Unsupported file type or parse error: {e}")

# ========= Alerts & roster =========
def render_alerts():
    st.subheader("Live Alerts")
    alerts = st.session_state.get("alerts", [])
    if not alerts:
        st.caption("No active alerts."); return
    alerts_list = list(alerts)[:10]; remove_idx = None
    for idx, a in enumerate(alerts_list):
        c1, c2, c3, c4 = st.columns([2, 2, 5, 2])
        c1.write(a.get("ts","—")); c2.write(a.get("resident","—"))
        c3.write(f"{a.get('type','—')} (score {a.get('score','—')}) — {a.get('source','')}")
        if c4.button("Acknowledge", key=f"ack_{idx}_{uuid.uuid4().hex[:6]}"): remove_idx = idx
    if remove_idx is not None:
        try:
            a = alerts_list[remove_idx]; st.session_state.alerts.remove(a)
            st.toast(f"Acknowledged {a.get('type','alert')} for {a.get('resident','—')}", icon="✅")
        except Exception: pass

def render_roster_snapshot(roster_placeholder, hr, sp, rr, score, should_alert, source):
    try: hr = int(hr)
    except: hr = int(st.session_state.last_vitals[0])
    try: sp = int(sp)
    except: sp = int(st.session_state.last_vitals[1])
    try: rr = int(rr)
    except: rr = int(st.session_state.last_vitals[2])
    row = {
        "Resident": resident_label(st.session_state.current_resident_idx),
        "HR (bpm)": hr, "SpO2 (%)": sp, "RR (/min)": rr,
        "Fall score": f"{score:.2f}", "Status": "ALERT" if should_alert else "OK",
        "Source": source,
    }
    df_row = pd.DataFrame.from_records([row], columns=[
        "Resident","HR (bpm)","SpO2 (%)","RR (/min)","Fall score","Status","Source"
    ])
    roster_placeholder.dataframe(df_row, use_container_width=True)

# ========= Session State =========
if "running" not in st.session_state: st.session_state.running = False
if "alerts" not in st.session_state: st.session_state.alerts = deque(maxlen=500)
if "roster" not in st.session_state: st.session_state.roster = ["Resident-A7Q4"]
if "current_resident_idx" not in st.session_state: st.session_state.current_resident_idx = 0
if "last_vitals" not in st.session_state: st.session_state.last_vitals = (72,97,16)
if "incidents_df" not in st.session_state:
    st.session_state.incidents_df = pd.DataFrame(columns=["ts","resident","type","score","note","source"])
if "vitals_hist" not in st.session_state: st.session_state.vitals_hist = deque(maxlen=600)
if "device_pref" not in st.session_state: st.session_state.device_pref = "auto"
# Device modal state
if "show_device_modal" not in st.session_state: st.session_state.show_device_modal = False
if "device_connected"  not in st.session_state: st.session_state.device_connected  = False
if "device_mode"       not in st.session_state: st.session_state.device_mode       = "Simulator"
if "device_tail_path"  not in st.session_state: st.session_state.device_tail_path  = str(Path.home() / "Documents" / "device_vitals.csv")

def push_vitals(hr:int, spo2:int, rr:int):
    st.session_state.last_vitals = (hr, spo2, rr)
    st.session_state.vitals_hist.append({"t": datetime.now(UTC).isoformat(timespec="seconds"),
                                         "HR": hr, "SpO2": spo2, "RR": rr})

def vitals_history_df():
    if not st.session_state.vitals_hist:
        hr, sp, rr = st.session_state.last_vitals; push_vitals(hr, sp, rr)
    df = pd.DataFrame(list(st.session_state.vitals_hist)); df.index = range(len(df))
    return df[["HR","SpO2","RR"]]

# ========= Connectivity checks (used by overlay) =========
def _tail_file_has_signal_csv(p: Path) -> tuple[bool, str]:
    try:
        df = pd.read_csv(p)
        if df.empty: return False, "File exists but has no rows yet."
        hr_col = next((c for c in df.columns if c.lower() in ("hr","heart","bpm")), None)
        sp_col = next((c for c in df.columns if "spo" in c.lower() or "ox" in c.lower()), None)
        rr_col = next((c for c in df.columns if c.lower() in ("rr","resp","breath")), None)
        if not all([hr_col, sp_col, rr_col]): return False, "Missing hr/spo2/rr columns."
        row = df.iloc[-1]
        _ = int(row[hr_col]); _ = int(row[sp_col]); _ = int(row[rr_col])
        return True, "OK"
    except Exception as e:
        return False, f"CSV parse error: {e}"

def _tail_file_has_signal_jsonl(p: Path) -> tuple[bool, str]:
    try:
        lines = p.read_text().strip().splitlines()
        if not lines: return False, "File exists but has no lines yet."
        rec = json.loads(lines[-1])
        _ = int(rec.get("hr") or rec.get("bpm") or 0)
        _ = int(rec.get("spo2") or rec.get("oxygen") or 0)
        _ = int(rec.get("rr") or rec.get("resp") or 0)
        return True, "OK"
    except Exception as e:
        return False, f"JSONL parse error: {e}"

def check_device_connected(mode: str, path_val: str) -> tuple[bool, str]:
    if mode == "Simulator":  # always OK
        return True, "Simulator ready."
    p = Path(path_val)
    if not p.exists(): return False, f"File not found: {p}"
    if p.stat().st_size == 0: return False, "File is empty; no data yet."
    try:
        if (time.time() - p.stat().st_mtime) > 120:
            return False, "No recent data (file not updated in the last 2 minutes)."
    except Exception:
        pass
    if p.suffix.lower() == ".csv":
        return _tail_file_has_signal_csv(p)
    return _tail_file_has_signal_jsonl(p)

# ========= Popup-like overlay (works on older Streamlit) =========
def device_connect_modal():
    st.markdown("""
        <style>
        .overlay-bg { position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 9990; }
        .overlay-card {
            position: fixed; top: 10vh; left: 50%; transform: translateX(-50%);
            width: min(720px, 92vw); background: white; padding: 18px 18px 8px 18px;
            border-radius: 14px; box-shadow: 0 12px 28px rgba(0,0,0,0.18); z-index: 9991;
        }
        </style>
        <div class="overlay-bg"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="overlay-card">', unsafe_allow_html=True)
    st.markdown("## Connect Live Device Vitals")

    # Banner stays until connected
    if not st.session_state.device_connected:
        st.error("Device is not connected yet — please put on the device and then click **Connect**.")

    st.write("**Please put on the device.** When it’s on and sending data, choose how to ingest and click **Connect**.")

    mode = st.radio("Device mode", ["Simulator", "Tail a file"], index=0, horizontal=True, key="device_mode_radio")
    path_val = st.session_state.device_tail_path
    if mode == "Tail a file":
        path_val = st.text_input(
            "Path to CSV/JSONL file (appended by your device gateway)",
            value=st.session_state.device_tail_path,
            help="CSV needs: hr, spo2, rr. JSONL needs keys: hr, spo2, rr."
        )

    c1, c2, _ = st.columns([1,1,3])
    connect = c1.button("Connect", type="primary", key="btn_modal_connect")
    cancel  = c2.button("Cancel", key="btn_modal_cancel")

    if connect:
        ok, detail = check_device_connected(mode, path_val)
        if not ok:
            st.error(f"Device is not connected: {detail}")     # popup message
            st.toast("Device is not connected", icon="❌")
        else:
            st.session_state.device_mode = mode
            st.session_state.device_tail_path = path_val
            st.session_state.device_connected = True
            st.session_state.show_device_modal = False
            st.toast("Device connected", icon="✅")
            try: st.rerun()
            except Exception: pass

    if cancel:
        st.session_state.show_device_modal = False
        try: st.rerun()
        except Exception: pass

    st.markdown("</div>", unsafe_allow_html=True)

# ========= Sidebar =========
st.sidebar.header("Session")
role = st.sidebar.radio("Choose view", ["Patient","Care Team"], horizontal=True)

# If Care Team selected, enforce login (Patient never sees login screen)
if role == "Care Team":
    current_user = login_flow()
else:
    current_user = None

# Patient sidebar
if role == "Patient":
    st.sidebar.markdown(
        "<style>div[data-testid='stSidebar'] button[kind='primary']{font-size:22px;padding:18px 24px;height:auto;}</style>",
        unsafe_allow_html=True
    )
    st.sidebar.subheader("My Vitals")
    hr_sb, spo2_sb, rr_sb = st.session_state.last_vitals
    st.sidebar.metric("Heart Rate", f"{hr_sb} bpm")
    st.sidebar.metric("SpO₂", f"{spo2_sb} %")
    st.sidebar.metric("Resp Rate", f"{rr_sb} /min")

    def raise_manual_alert():
        st.session_state.alerts.appendleft({
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "resident": resident_label(st.session_state.current_resident_idx),
            "type": "assist_now", "score": 1.0, "note": "Manual help button", "source": "manual"
        })
        st.toast("Help requested — alert sent to care team.", icon="🚨")

    st.sidebar.button("🚨 I Need Help", type="primary", use_container_width=True, on_click=raise_manual_alert)

# Care Team sidebar (only shown after successful login)
DEVICE_SOURCES = ["Live merged dataset", "Uploads", "Live device vitals"]
if role == "Care Team":
    data_source = st.sidebar.selectbox("Data Source", DEVICE_SOURCES, index=0)
    fs = st.sidebar.number_input("Sampling rate (Hz) for IMU", value=50, min_value=10, max_value=200, step=5)
    win_s = st.sidebar.number_input("IMU window (sec)", value=1.0, min_value=0.5, max_value=3.0, step=0.1)
    overlap = st.sidebar.slider("IMU overlap", 0.0, 0.9, 0.5, 0.1)
    fall_thresh = st.sidebar.slider("Fall alert threshold", 0.5, 0.99, 0.85, 0.01)
    simulate_vitals = st.sidebar.checkbox("Simulate vitals if missing", value=True)
    device_pref = st.sidebar.selectbox("IMU device", ["auto","BELT","NECK","TORS","PCKT","WRST"], index=0)
    st.session_state.device_pref = device_pref
    demo_seconds = st.sidebar.slider("Max seconds to process this run", 5, 300, 60, 5)

    uploaded_files = []
    if data_source == "Uploads":
        uploaded_files = st.sidebar.file_uploader("Upload CSV/XLSX/JSON", type=["csv","xlsx","xls","json"], accept_multiple_files=True)

    # ---- LIVE DEVICE VITALS: connect trigger ----
    if data_source == "Live device vitals":
        st.sidebar.markdown("**Live device setup**")
        if not st.session_state.device_connected:
            # Clicking shows a toast immediately and sets the flag
            if st.sidebar.button("Connect device", type="primary", use_container_width=True, key="btn_connect_device"):
                st.session_state.show_device_modal = True
                st.toast("Device is not connected. Please put on the device and click Connect.", icon="❌")
        else:
            st.sidebar.success(f"Connected ({st.session_state.device_mode})")
            if st.session_state.device_mode == "Tail a file":
                st.sidebar.caption(st.session_state.device_tail_path)

        # Render the overlay NOW if the flag is set
        if st.session_state.show_device_modal:
            device_connect_modal()

    start_btn = st.sidebar.button("Start / Restart")
else:
    fs, win_s, overlap, fall_thresh, simulate_vitals = 50, 1.0, 0.5, 0.85, True
    data_source, uploaded_files, demo_seconds, start_btn = None, [], 60, False

if role == "Care Team" and start_btn:
    st.session_state.running = True
    st.session_state.alerts.clear()
    st.session_state.incidents_df = st.session_state.incidents_df.iloc[0:0]

# ========= Header =========
st.title("RTMMS — Real-Time Multi-Modal Monitoring Systems for Assisted Living")

# ========= Care Team header widgets =========
def render_caretaker_header():
    h, s, r = st.session_state.last_vitals
    st.markdown("<h3 style='text-align:center;margin:0.25rem 0 0.5rem 0;'>Current Vitals</h3>", unsafe_allow_html=True)
    _, col1, col2, col3, _ = st.columns([1, 2, 2, 2, 1])
    with col1: st.metric("Heart Rate", f"{h} bpm")
    with col2: st.metric("SpO₂", f"{s} %")
    with col3: st.metric("Resp Rate", f"{r} /min")
    st.subheader("Vitals — Live Trend")
    st.line_chart(vitals_history_df(), height=220, use_container_width=True)

# ========= Processing =========
def process_imu(df, cols, src_name, device_pref, fs, win_s, overlap, fall_thresh, simulate_vitals, runtime_cap_s, roster_placeholder):
    chosen = cols.get("device")
    if device_pref and device_pref != "auto":
        p = device_pref; cx, cy, cz = f"{p}_ACC_X", f"{p}_ACC_Y", f"{p}_ACC_Z"
        if all(c in df.columns for c in (cx, cy, cz)):
            cols["ax"], cols["ay"], cols["az"] = cx, cy, cz; chosen = p
    imu = df[[cols["ax"], cols["ay"], cols["az"]]].astype(float).to_numpy()
    mag = np.linalg.norm(imu, axis=1)
    win = int(fs * win_s); hop = int(max(1, win * (1 - overlap)))
    started = time.time()
    for _, segment in sliding_windows(mag, win, hop):
        if (time.time() - started) > runtime_cap_s: break
        now = datetime.now(UTC)
        score = fall_score(segment)
        peak_idx = int(np.nanargmax(segment)); recent_peak = peak_idx >= int(len(segment)*0.6)
        tail = segment[int(len(segment)*0.7):] if len(segment) >= 3 else segment
        inactive = float(np.nanstd(tail)) < 0.05 if len(tail) else False
        should_alert = (score >= fall_thresh) and recent_peak and inactive
        if simulate_vitals: hr, sp, rr = vitals_sim()
        else: hr, sp, rr = st.session_state.last_vitals
        push_vitals(hr, sp, rr)
        if should_alert:
            alert = {"ts": now.isoformat(timespec="seconds"),
                     "resident": resident_label(st.session_state.current_resident_idx),
                     "type": "fall_possible", "score": round(score, 2),
                     "note": "IMU spike + inactivity",
                     "source": src_name if not chosen else f"{src_name}/{chosen}"}
            st.session_state.alerts.appendleft(alert)
            st.session_state.incidents_df = pd.concat([pd.DataFrame([alert]), st.session_state.incidents_df], ignore_index=True)
        if roster_placeholder is not None:
            render_roster_snapshot(roster_placeholder, hr, sp, rr, score, should_alert,
                                   src_name if not chosen else f"{src_name}/{chosen}")
        time.sleep(hop / float(fs))

def process_vitals(df, cols, src_name, roster_placeholder):
    hr = df[cols["hr"]] if cols["hr"] in df else pd.Series(dtype=float)
    sp = df[cols["spo2"]] if cols["spo2"] in df else pd.Series(dtype=float)
    rr = df[cols["rr"]] if cols["rr"] in df else pd.Series(dtype=float)
    cur_hr = int(hr.iloc[-1]) if not hr.empty else st.session_state.last_vitals[0]
    cur_sp = int(sp.iloc[-1]) if not sp.empty else st.session_state.last_vitals[1]
    cur_rr = int(rr.iloc[-1]) if not rr.empty else st.session_state.last_vitals[2]
    push_vitals(cur_hr, cur_sp, cur_rr)
    flags = []
    if cur_hr > 120: flags.append("tachycardia")
    if cur_sp < 90:  flags.append("hypoxemia")
    if cur_rr > 24:  flags.append("tachypnea")
    if roster_placeholder is not None:
        render_roster_snapshot(roster_placeholder, cur_hr, cur_sp, cur_rr, score=0.0,
                               should_alert=bool(flags), source=src_name)
    if flags:
        alert = {"ts": datetime.now(UTC).isoformat(timespec="seconds"),
                 "resident": resident_label(st.session_state.current_resident_idx),
                 "type": "vitals_advisory", "score": 0.75,
                 "note": ", ".join(flags), "source": src_name}
        st.session_state.alerts.appendleft(alert)
        st.session_state.incidents_df = pd.concat([pd.DataFrame([alert]), st.session_state.incidents_df], ignore_index=True)

def process_generic(df, src_name):
    st.info(f"Showing preview of '{src_name}' (no specific model matched)")
    st.dataframe(df.head(50), use_container_width=True)

# ---------- Runners ----------
def run_live_merged(path_str: str):
    path = Path(path_str)
    if not path.exists(): st.error(f"Merged CSV not found: {path}"); return
    try: df = pd.read_csv(path)
    except Exception as e: st.error(f"Failed to read merged CSV: {e}"); return
    if df.empty: st.warning("Merged CSV is empty."); return
    kind, cols = classify_table(df)
    st.write(f"**Streaming (live merged)**: {path.name} → detected **{kind}**")
    roster_placeholder = st.session_state.get("_roster_placeholder", st.empty())
    if kind == "imu" and cols["ax"] and cols["ay"] and cols["az"]:
        process_imu(df, cols, path.name, st.session_state.device_pref, fs, win_s, overlap, fall_thresh,
                    simulate_vitals, runtime_cap_s=int(demo_seconds), roster_placeholder=roster_placeholder)
    elif kind == "vitals":
        process_vitals(df, cols, path.name, roster_placeholder)
    else:
        process_generic(df, path.name)

def run_uploads(files):
    any_data = False
    roster_placeholder = st.session_state.get("_roster_placeholder", st.empty())
    for uf in files:
        try: df, src = parse_upload(uf)
        except Exception as e: st.error(f"Failed to read '{getattr(uf,'name','uploaded')}': {e}"); continue
        if df.empty: st.warning(f"'{src}' has no rows."); continue
        any_data = True
        kind, cols = classify_table(df)
        st.write(f"**Processing (uploaded)**: {src} → detected **{kind}**")
        if kind == "imu" and cols["ax"] and cols["ay"] and cols["az"]:
            process_imu(df, cols, src, st.session_state.device_pref, fs, win_s, overlap, fall_thresh,
                        simulate_vitals, runtime_cap_s=int(demo_seconds), roster_placeholder=roster_placeholder)
        elif kind == "vitals":
            process_vitals(df, cols, src, roster_placeholder)
        else:
            process_generic(df, src)
    if not any_data: st.info("Upload one or more files in the sidebar, then click Start.")

def run_device_vitals():
    if not st.session_state.device_connected:
        st.warning("Device not connected. Click **Connect device** in the sidebar."); return
    mode, tail_path = st.session_state.device_mode, st.session_state.device_tail_path
    roster_placeholder = st.session_state.get("_roster_placeholder", st.empty())
    st.write(f"**Streaming (device vitals)**: **{mode}**")
    start = time.time(); last_line_seen = -1
    while (time.time() - start) < int(demo_seconds):
        if mode == "Simulator":
            tnow = time.time() - start
            hr = int(72 + 5*np.sin(2*np.pi*tnow/6.0) + np.random.randn()*1.0)
            sp = int(np.clip(97 + 0.5*np.sin(2*np.pi*tnow/10.0) + np.random.randn()*0.2, 90, 100))
            rr = int(np.clip(16 + 2*np.sin(2*np.pi*tnow/7.0) + np.random.randn()*0.5, 8, 28))
        else:
            p = Path(tail_path)
            if not p.exists(): st.error(f"Tail file not found: {p}"); return
            try:
                if p.suffix.lower() == ".csv":
                    df = pd.read_csv(p)
                    if df.empty: time.sleep(1); continue
                    hr_col = next((c for c in df.columns if c.lower() in ("hr","heart","bpm")), None)
                    sp_col = next((c for c in df.columns if "spo" in c.lower() or "ox" in c.lower()), None)
                    rr_col = next((c for c in df.columns if c.lower() in ("rr","resp","breath")), None)
                    if not all([hr_col, sp_col, rr_col]): time.sleep(1); continue
                    if last_line_seen == len(df)-1: time.sleep(0.5); continue
                    last_line_seen = len(df)-1; row = df.iloc[-1]
                    hr, sp, rr = int(row[hr_col]), int(row[sp_col]), int(row[rr_col])
                else:
                    lines = p.read_text().strip().splitlines()
                    if not lines: time.sleep(1); continue
                    if last_line_seen == len(lines)-1: time.sleep(0.5); continue
                    last_line_seen = len(lines)-1; rec = json.loads(lines[-1])
                    hr = int(rec.get("hr") or rec.get("bpm") or 72)
                    sp = int(rec.get("spo2") or rec.get("oxygen") or 97)
                    rr = int(rec.get("rr") or rec.get("resp") or 16)
            except Exception as e:
                st.error(f"Failed to read tail file: {e}"); return
        push_vitals(hr, sp, rr)
        flags = []
        if hr > 120: flags.append("tachycardia")
        if sp < 90:  flags.append("hypoxemia")
        if rr > 24:  flags.append("tachypnea")
        if flags:
            alert = {"ts": datetime.now(UTC).isoformat(timespec="seconds"),
                     "resident": resident_label(st.session_state.current_resident_idx),
                     "type": "vitals_advisory", "score": 0.75,
                     "note": ", ".join(flags), "source": f"device/{mode.lower()}"}
            st.session_state.alerts.appendleft(alert)
            st.session_state.incidents_df = pd.concat([pd.DataFrame([alert]), st.session_state.incidents_df], ignore_index=True)
        render_roster_snapshot(roster_placeholder, hr, sp, rr, score=0.0,
                               should_alert=bool(flags), source=f"device/{mode.lower()}")
        time.sleep(1.0)

# ========= Main UI =========

if role == "Patient":
    st.info("Patient view — personal vitals and assistance")
    if st.button("Refresh vitals"):
        try: st.rerun()
        except Exception: pass
    hr, spo2, rr = st.session_state.last_vitals
    t = np.linspace(0, 30, 300)
    trend_df = pd.DataFrame({
        "time_s": t,
        "HR":   hr   + 3.0*np.sin(2*np.pi*t/6.0),
        "SpO2": spo2 + 0.5*np.sin(2*np.pi*t/8.0),
        "RR":   rr   + 1.0*np.sin(2*np.pi*t/5.0),
    }).set_index("time_s")
    st.subheader("Vitals (30s trend)")
    st.line_chart(trend_df, height=260, use_container_width=True)
    with st.expander("Prevention tips", expanded=True):
        st.markdown("- Keep walkways clear and well lit\n- Use non-slip mats in bathroom and kitchen\n- Wear supportive footwear indoors\n- Rise slowly from sitting/lying positions\n- Keep emergency contact within reach")

if role == "Care Team":
    st.success("Care-team view — roster, live alerts, incidents & downloads")
    # Center metrics + trend
    h, s, r = st.session_state.last_vitals
    st.markdown("<h3 style='text-align:center;margin:0.25rem 0 0.5rem 0;'>Current Vitals</h3>", unsafe_allow_html=True)
    _, col1, col2, col3, _ = st.columns([1, 2, 2, 2, 1])
    with col1: st.metric("Heart Rate", f"{h} bpm")
    with col2: st.metric("SpO₂", f"{s} %")
    with col3: st.metric("Resp Rate", f"{r} /min")
    st.subheader("Vitals — Live Trend")
    st.line_chart(vitals_history_df(), height=220, use_container_width=True)

    roster_placeholder = st.empty()
    st.session_state._roster_placeholder = roster_placeholder
    render_alerts()
    with st.expander("Download incidents as CSV", expanded=False):
        csv_bytes = st.session_state.incidents_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download incidents.csv", data=csv_bytes, file_name="incidents.csv", mime="text/csv")

# ========= Run selected source =========
if role == "Care Team" and st.session_state.running:
    if data_source == "Live merged dataset":
        run_live_merged(MERGED_PATH_TEXT)
    elif data_source == "Uploads":
        run_uploads(uploaded_files)
    elif data_source == "Live device vitals":
        run_device_vitals()
else:
    if role == "Care Team":
        if data_source == "Live device vitals" and not st.session_state.device_connected:
            st.info("Click **Connect device** in the sidebar to open the popup, then **Start / Restart**.")
        else:
            st.info("Choose a data source and click **Start / Restart**.")
    else:
        st.caption("Patient view shows sidebar vitals, a trend chart, and a big Help button. Care Team handles alerts and data sources.")
