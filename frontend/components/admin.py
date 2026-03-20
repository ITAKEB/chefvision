import streamlit as st
import requests
import api_client

# ── Parameters ────────────────────────────────────────────────────────────────
MAX_PDF_FILES = 10          # ← Change this to restrict the number of PDFs


# ── Topbar ────────────────────────────────────────────────────────────────────
def _topbar():
    col_logo, col_space, col_user = st.columns([2, 5, 2])
    with col_logo:
        st.markdown("""
        <div class="cv-topbar" style="border-bottom:none; padding:1rem 0; margin:0;">
            <span class="cv-logo">Chef<span>Vision</span></span>
        </div>
        """, unsafe_allow_html=True)
    with col_user:
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:flex-end;
                    gap:0.6rem; padding-top:1.1rem;">
            <div style="text-align:right;">
                <div style="font-weight:600; font-size:0.88rem; color:var(--cocoa);">
                    {st.session_state.username}
                </div>
                <div style="font-size:0.75rem; color:var(--amber); font-weight:600;">
                    Administrador
                </div>
            </div>
            <div style="width:36px; height:36px; background:var(--cocoa); border-radius:50%;
                        display:flex; align-items:center; justify-content:center;
                        color:var(--cream); font-size:1.1rem;">👑</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="margin:0 0 2rem; border:none; border-top:1px solid var(--mist);">',
                unsafe_allow_html=True)


# ── Main admin view ───────────────────────────────────────────────────────────
def show_admin():
    _topbar()

    # Fetch file list from backend API
    try:
        records = api_client.list_files()
    except (requests.HTTPError, requests.ConnectionError) as e:
        st.error(f"Error al conectar con el servidor: {e}")
        records = []

    count = len(records)
    slots = MAX_PDF_FILES - count

    # ── Page title
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <h1 style="font-size:2rem; margin:0 0 0.4rem;">📚 Aprender Recetas</h1>
        <p style="color:var(--espresso); opacity:0.7; margin:0; font-size:0.95rem;">
            Sube los libros o documentos de recetas para entrenar el asistente culinario
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="cv-card">
            <div class="cv-stat">
                <div class="cv-stat-num">{count}</div>
                <div class="cv-stat-label">Documentos cargados</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="cv-card">
            <div class="cv-stat">
                <div class="cv-stat-num">{slots}</div>
                <div class="cv-stat-label">Espacios disponibles</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="cv-card">
            <div class="cv-stat">
                <div class="cv-stat-num">{MAX_PDF_FILES}</div>
                <div class="cv-stat-label">Límite máximo</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Upload section
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    st.markdown("""
    <h3 style="font-size:1.15rem; margin:0 0 0.4rem;">Subir nuevos documentos</h3>
    <p style="font-size:0.86rem; color:var(--espresso); opacity:0.65; margin:0 0 1rem;">
        Formatos aceptados: PDF · Máx. {max} documentos en total
    </p>
    """.format(max=MAX_PDF_FILES), unsafe_allow_html=True)

    if slots <= 0:
        st.markdown(f"""
        <div class="cv-warn">
            ⚠️ Has alcanzado el límite de <strong>{MAX_PDF_FILES} documentos</strong>.
            Elimina alguno antes de subir nuevos.
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader(
            f"Arrastra o selecciona hasta {slots} PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader",
        )

        if uploaded:
            # Filter duplicates & enforce limit
            existing_names = {r["name"] for r in records}
            new_files      = [f for f in uploaded if f.name not in existing_names]
            new_files      = new_files[:slots]  # enforce remaining slots

            duplicates = [f.name for f in uploaded if f.name in existing_names]
            over_limit = uploaded[slots:]       # files that didn't fit

            if duplicates:
                st.markdown(f"""
                <div class="cv-warn">ℹ️ Ya existen: {', '.join(duplicates)}</div>
                """, unsafe_allow_html=True)

            if over_limit:
                st.markdown(f"""
                <div class="cv-warn">⚠️ Solo se pueden agregar {slots} más.
                Algunos archivos fueron ignorados.</div>
                """, unsafe_allow_html=True)

            # Preview list
            if new_files:
                for f in new_files:
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:0.75rem;
                                padding:0.6rem 0.8rem; border-radius:var(--radius-sm);
                                background:rgba(107,143,113,0.08); margin:0.3rem 0;">
                        <span style="font-size:1.4rem;">📄</span>
                        <div>
                            <div style="font-weight:600; font-size:0.9rem;">{f.name}</div>
                            <div style="font-size:0.78rem; color:var(--espresso); opacity:0.6;">
                                {round(f.size/1024,1)} KB
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                btn_disabled = len(new_files) == 0
                st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

                with st.container():
                    col_btn, col_space = st.columns([1, 3])
                    with col_btn:
                        if not btn_disabled and st.button(
                            f"⬆ Cargar {len(new_files)} documento(s)",
                            use_container_width=True
                        ):
                            saved = 0
                            errors = []
                            for f in new_files:
                                try:
                                    api_client.upload_pdf(f)
                                    saved += 1
                                except requests.HTTPError as e:
                                    if e.response is not None and e.response.status_code == 409:
                                        errors.append(f"{f.name}: ya está embebido")
                                    else:
                                        errors.append(f"{f.name}: {e}")
                                except requests.ConnectionError:
                                    errors.append(f"{f.name}: no se pudo conectar al servidor")
                            if saved:
                                st.success(f"✅ {saved} documento(s) cargados exitosamente.")
                            for err in errors:
                                st.error(err)
                            if saved:
                                st.rerun()
            else:
                st.markdown("""
                <div class="cv-info">
                    Todos los archivos seleccionados ya estaban cargados.
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close card

    # ── Document library
    if records:
        st.markdown("""
        <h3 style="font-size:1.1rem; margin:1.5rem 0 0.75rem;">📖 Biblioteca de recetas</h3>
        """, unsafe_allow_html=True)

        for idx, rec in enumerate(records):
            col_icon, col_info, col_status, col_del = st.columns([0.5, 5, 1.5, 1.5])
            with col_icon:
                st.markdown('<div style="font-size:2rem; padding-top:0.3rem">📕</div>',
                            unsafe_allow_html=True)
            with col_info:
                st.markdown(f"""
                <div style="padding:0.25rem 0;">
                    <div style="font-weight:600; font-size:0.95rem;">{rec['name']}</div>
                    <div style="font-size:0.78rem; color:var(--espresso); opacity:0.6; margin-top:0.2rem;">
                        {rec['size_kb']} KB · Cargado el {rec['uploaded']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_status:
                if rec.get("embedded", False):
                    st.markdown("""
                    <div style="padding-top:0.5rem;">
                        <span style="background:#6b8f71; color:white; padding:0.2rem 0.6rem;
                                     border-radius:1rem; font-size:0.75rem; font-weight:600;">
                            ✓ Embebido
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="padding-top:0.5rem;">
                        <span style="background:#d4a574; color:white; padding:0.2rem 0.6rem;
                                     border-radius:1rem; font-size:0.75rem; font-weight:600;">
                            ○ Sin embeber
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            with col_del:
                if st.button("🗑 Eliminar", key=f"del_{idx}", use_container_width=True):
                    try:
                        api_client.delete_file(rec["name"])
                        st.rerun()
                    except (requests.HTTPError, requests.ConnectionError) as e:
                        st.error(f"Error al eliminar {rec['name']}: {e}")

            st.markdown('<hr style="border:none; border-top:1px solid var(--mist); margin:0.4rem 0;">',
                        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cv-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">📭</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                        color:var(--espresso); opacity:0.7;">
                Aún no hay documentos cargados
            </div>
            <div style="font-size:0.85rem; opacity:0.5; margin-top:0.5rem;">
                Sube tu primer PDF de recetas para comenzar
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Logout
    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    col_lx, _ = st.columns([1, 4])
    with col_lx:
        if st.button("← Cerrar sesión", use_container_width=True):
            for k in ["authenticated", "role", "username"]:
                st.session_state[k] = None if k == "role" else (
                    False if k == "authenticated" else "")
            st.rerun()
