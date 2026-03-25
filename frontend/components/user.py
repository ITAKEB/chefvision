import streamlit as st
from datetime import datetime
import api_client

MEAL_OPTIONS = {
    "Desayuno": {"emoji": "🌅", "hint": "Energía para empezar el día"},
    "Almuerzo": {"emoji": "☀️", "hint": "El plato fuerte del día"},
    "Cena":     {"emoji": "🌙", "hint": "Ligero y reconfortante"},
}


# ── Topbar ────────────────────────────────────────────────────────────────────
def _topbar():
    col_logo, col_space, col_user = st.columns([2, 5, 2])
    with col_logo:
        st.markdown("""
        <div style="padding:1.1rem 0 0.5rem;">
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
                <div style="font-size:0.75rem; color:var(--sage); font-weight:600;">Usuario</div>
            </div>
            <div style="width:36px; height:36px; background:var(--sage); border-radius:50%;
                        display:flex; align-items:center; justify-content:center;
                        color:white; font-size:1.1rem;">👤</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="margin:0 0 2rem; border:none; border-top:1px solid var(--mist);">',
                unsafe_allow_html=True)


# ── Main user view ────────────────────────────────────────────────────────────
def show_user():
    _topbar()

    # ── Greeting
    hour = datetime.now().hour
    greeting = "Buenos días" if hour < 12 else ("Buenas tardes" if hour < 19 else "Buenas noches")

    st.markdown(f"""
    <div style="margin-bottom:2rem;">
        <h1 style="font-size:2rem; margin:0 0 0.4rem;">
            {greeting}, {st.session_state.username.split()[0]} 👋
        </h1>
        <p style="color:var(--espresso); opacity:0.7; margin:0; font-size:0.95rem;">
            Fotografía tus ingredientes y te sugiero una receta perfecta
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 1: Meal type selector
    st.markdown("""
    <div class="cv-card">
        <h3 style="font-size:1.1rem; margin:0 0 0.25rem;">1 · ¿Para qué momento del día?</h3>
        <p style="font-size:0.85rem; color:var(--espresso); opacity:0.65; margin:0 0 1rem;">
            Generar receta para:
        </p>
    """, unsafe_allow_html=True)

    # Initialize meal selection
    if "selected_meal" not in st.session_state:
        st.session_state.selected_meal = "Almuerzo"

    # Meal buttons as columns
    m1, m2, m3 = st.columns(3)
    for col, (meal, info) in zip([m1, m2, m3], MEAL_OPTIONS.items()):
        with col:
            is_active = st.session_state.selected_meal == meal

            if st.button(
                f"{info['emoji']}\n{meal}",
                key=f"meal_{meal}",
                use_container_width=True,
            ):
                st.session_state.selected_meal = meal
                st.rerun()

            # Visual indicator under each button
            indicator_color = "var(--amber)" if is_active else "transparent"
            st.markdown(f"""
            <div style="height:3px; background:{indicator_color}; border-radius:2px;
                        margin-top:-0.5rem; transition:background 0.2s;"></div>
            <div style="text-align:center; font-size:0.75rem; color:var(--espresso);
                        opacity:0.55; margin-top:0.3rem;">{info['hint']}</div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1rem; padding:0.6rem 1rem; background:rgba(212,132,58,0.08);
                border-radius:var(--radius-sm); font-size:0.88rem; font-weight:600;
                color:var(--amber);">
        ✓ Seleccionado: {MEAL_OPTIONS[st.session_state.selected_meal]['emoji']}
        {st.session_state.selected_meal}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── STEP 2: Photo upload
    st.markdown("""
    <div class="cv-card">
        <h3 style="font-size:1.1rem; margin:0 0 0.25rem;">2 · Fotografía tus ingredientes</h3>
        <p style="font-size:0.85rem; color:var(--espresso); opacity:0.65; margin:0 0 1rem;">
            Puedes subir una o varias fotos · Desde cámara o galería
        </p>
    """, unsafe_allow_html=True)

    uploaded_images = st.file_uploader(
        "Sube tus fotos de alimentos",
        type=["jpg", "jpeg", "png", "webp", "heic"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="food_photos",
    )

    if uploaded_images:
        # Show image previews in a grid
        st.markdown(f"""
        <div style="font-size:0.82rem; color:var(--espresso); opacity:0.6;
                    margin:0.5rem 0; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.06em;">
            {len(uploaded_images)} imagen(es) cargada(s)
        </div>
        """, unsafe_allow_html=True)

        # Preview grid — up to 4 thumbnails
        preview_cols = st.columns(min(len(uploaded_images), 4))
        for i, img in enumerate(uploaded_images[:4]):
            with preview_cols[i]:
                st.image(img, use_container_width=True,
                         caption=f"📷 {img.name[:18]}…" if len(img.name) > 18 else img.name)

        if len(uploaded_images) > 4:
            st.markdown(f"""
            <div style="font-size:0.82rem; color:var(--espresso); opacity:0.55;
                        text-align:center; margin-top:0.3rem;">
                +{len(uploaded_images)-4} imagen(es) adicional(es)
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem 0; color:var(--espresso); opacity:0.45;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">📷</div>
            <div style="font-size:0.88rem;">
                Toca aquí para seleccionar fotos<br>
                <span style="font-size:0.78rem; opacity:0.7;">
                    Compatible con cámara del celular
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── STEP 3: Generate
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    can_generate = bool(uploaded_images)

    if not can_generate:
        st.markdown("""
        <div style="text-align:center; padding:0.5rem 0;">
        """, unsafe_allow_html=True)

    col_gen, col_clear = st.columns([3, 1])
    with col_gen:
        meal_emoji = MEAL_OPTIONS[st.session_state.selected_meal]["emoji"]
        generate_clicked = st.button(
            f"{meal_emoji} Generar receta para {st.session_state.selected_meal}",
            use_container_width=True,
            disabled=not can_generate,
        )

    with col_clear:
        if st.button("🔄 Reiniciar", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("food_") or k in ("recipe_results", "detected_ingredients", "search_result", "generated_recipe"):
                    del st.session_state[k]
            st.rerun()

    if not can_generate:
        st.markdown("""
        <div style="text-align:center; font-size:0.82rem; color:var(--espresso);
                    opacity:0.5; margin-top:0.5rem;">
            Sube al menos una foto para habilitar la generación
        </div>
        """, unsafe_allow_html=True)

    # ── Run detection and search when button is clicked
    if generate_clicked and uploaded_images:
        all_ingredients = []

        with st.spinner("Analizando ingredientes en tus fotos…"):
            for img in uploaded_images:
                img.seek(0)
                try:
                    result = api_client.detect_ingredients(img)
                except Exception as e:
                    st.error(f"Error al analizar la imagen {img.name}: {e}")
                    continue

                if result.get("error_message"):
                    st.error(f"Error en {img.name}: {result['error_message']}")
                    continue

                for ingredient in result.get("ingredients", []):
                    all_ingredients.append(ingredient)

        # Store results in session_state so they survive reruns
        # Deduplicate ingredients by name, keeping highest confidence
        seen: dict[str, dict] = {}
        for ing in all_ingredients:
            name = ing.get("name_en", "").lower()
            if name not in seen or ing.get("confidence", 0) > seen[name].get("confidence", 0):
                seen[name] = ing
        all_ingredients = list(seen.values())

        st.session_state.detected_ingredients = all_ingredients

        if all_ingredients:
            ingredient_names_en = [ing.get("name_en", "") for ing in all_ingredients if ing.get("name_en")]
            ingredient_names_es = [ing.get("name_es", "") for ing in all_ingredients if ing.get("name_es")]

            with st.spinner("🧑‍🍳 Generando receta personalizada con IA…"):
                try:
                    gen_result = api_client.generate_recipe(
                        ingredients_en=ingredient_names_en,
                        ingredients_es=ingredient_names_es,
                        meal_type=st.session_state.selected_meal,
                    )
                except Exception as e:
                    st.error(f"Error al generar receta: {e}")
                    gen_result = None

            st.session_state.generated_recipe = gen_result
        else:
            st.session_state.generated_recipe = None

    # ── Display results from session_state (persists across reruns)
    _display_results()

    # ── Logout
    st.markdown('<div style="height:2.5rem"></div>', unsafe_allow_html=True)
    col_lx, _ = st.columns([1, 4])
    with col_lx:
        if st.button("← Cerrar sesión", use_container_width=True):
            for k in ["authenticated", "role", "username"]:
                st.session_state[k] = None if k == "role" else (
                    False if k == "authenticated" else "")
            st.rerun()


def _display_results():
    """Render detected ingredients and generated recipe from session_state."""
    all_ingredients = st.session_state.get("detected_ingredients")

    if all_ingredients is None:
        return

    if not all_ingredients:
        st.warning("🔍 No se detectaron ingredientes. Intenta con una foto diferente donde los ingredientes sean más visibles y estén bien iluminados.")
        return

    # ── Detected ingredients
    st.divider()
    st.subheader("🧺 Ingredientes detectados")

    cols_per_row = 3
    for i in range(0, len(all_ingredients), cols_per_row):
        row_items = all_ingredients[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, ing in enumerate(row_items):
            name_en = ing.get("name_en", "")
            name_es = ing.get("name_es", "")
            confidence = ing.get("confidence", 0)
            pct = int(confidence * 100)
            label = name_es if name_es != name_en else name_en
            with cols[j]:
                st.metric(label=label, value=f"{pct}%", help=f"{name_en} → {name_es}")

    # ── Generated recipe
    gen_result = st.session_state.get("generated_recipe")

    if gen_result and gen_result.get("recipe"):
        st.divider()
        st.subheader("🧑‍🍳 Receta sugerida por IA")

        with st.container(border=True):
            st.markdown(gen_result["recipe"])

        # Source references
        source_chunks = gen_result.get("source_chunks", [])
        if source_chunks:
            with st.expander("📚 Recetas de referencia utilizadas"):
                for chunk in source_chunks:
                    source = chunk.get("source_filename", "")
                    page = chunk.get("page_number", "")
                    score = chunk.get("relevance_score", 0)
                    st.caption(f"📄 {source} · pág. {page} · Relevancia: {int(score * 100)}%")

        # Debug: raw chunks
        debug_chunks = gen_result.get("debug_chunks")
        if debug_chunks:
            with st.expander("🐛 Debug: Raw chunks from ChromaDB"):
                for i, chunk in enumerate(debug_chunks):
                    st.text_area(
                        f"Chunk {i + 1}",
                        value=chunk,
                        height=400,
                        disabled=True,
                        key=f"debug_chunk_{i}",
                    )

    elif gen_result:
        st.info("🔍 No se encontraron recetas de referencia. Intenta con otros ingredientes o pide al administrador que cargue más libros de recetas.")
