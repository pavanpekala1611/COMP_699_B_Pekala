import streamlit as st

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

def care_team_login():
    """
    Gate for the Care Team view.

    - If already logged in → show small sidebar status + optional logout, return user dict.
    - If not logged in → show login + 'forgot username/password' UI and stop the app.

    Returns:
        dict with keys: {"username", "full_name", "role"} when logged in.
    """

    if "auth_user" in st.session_state:
        user = st.session_state["auth_user"]
        with st.sidebar:
            st.markdown(
                f"**Logged in as:** {user['full_name']} "
                f"(_{user['username']}_, {user['role']})"
            )
            if st.button("Log out"):
                st.session_state.pop("auth_user", None)
                st.experimental_rerun()
        return user

    st.title("Care Team Login")

    tab_login, tab_forgot_user, tab_forgot_pass = st.tabs(
        ["Login", "I forgot my username", "I forgot my password"]
    )

    # Normal login:
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
                st.experimental_rerun()

    # Forgot username:
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

    # Forgot password:
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
                    st.success(f"Your password is: **{user_info['password']}**")
    st.stop()
