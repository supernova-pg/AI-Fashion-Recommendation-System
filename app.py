import streamlit as st
import os
import requests
import json
import random
from pathlib import Path

import pandas as pd

from recommendation_engine import OutfitRecommender
from visualize import HIGH_LEVEL_CATEGORIES

# Page Configuration
st.set_page_config(
    page_title="DARE XAI - AI Fashion Stylist Assistant",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek CSS for Premium Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.4);
    }
    
    .product-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .product-card:hover {
        transform: scale(1.02);
        border-color: #6366f1;
        box-shadow: 0 10px 15px rgba(99, 102, 241, 0.15);
    }
    
    .product-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f8fafc;
        margin-top: 8px;
        margin-bottom: 4px;
        height: 48px;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .product-brand {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .product-price {
        font-size: 1.15rem;
        font-weight: 800;
        color: #38bdf8;
    }
    
    .stylist-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border-left: 5px solid #a855f7;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin-top: 25px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Recommender
@st.cache_resource
def get_recommender():
    return OutfitRecommender()


try:
    recommender = get_recommender()
except Exception as e:
    st.error(f"Failed to load recommendation engine: {e}")
    st.info("Ensure you have run training first using `python train.py` to generate the embedding index.")
    st.stop()


# LLM Query Helper using Gemini API via direct HTTP Requests
def query_gemini_api(prompt: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"API Error (Status {response.status_code}): {response.text}"
    except Exception as e:
        return f"Request failed: {e}"

# Fashion Query Validator — rejects off-topic inputs
FASHION_KEYWORDS = [
    "outfit", "wear", "wearing", "cloth", "dress", "shirt", "pant", "trouser",
    "jeans", "shoe", "footwear", "sandal", "sneaker", "heel", "boot", "loafer",
    "jacket", "blazer", "coat", "saree", "kurta", "lehenga", "sherwani",
    "t-shirt", "tshirt", "top", "skirt", "shorts", "suit", "ethnic", "western",
    "fashion", "style", "look", "casual", "formal", "party", "wedding", "office",
    "festive", "sports", "vacation", "beach", "summer", "winter", "occasion",
    "accessory", "necklace", "watch", "bag", "handbag", "clutch", "sunglasses",
    "color", "palette", "wardrobe", "attire", "apparel", "ensemble", "combo",
    "recommendation", "suggest", "pair", "match", "coordinate", "outfit for",
    "what to", "what should i", "what can i", "help me with"
]

def is_fashion_query(query: str, api_key: str) -> bool:
    """Return True if the query is fashion/outfit related, False otherwise."""
    q_lower = query.lower().strip()
    
    # Very short or empty
    if len(q_lower) < 3:
        return False

    # Use Gemini for accurate intent classification if API key is available
    if api_key:
        validation_prompt = f"""
You are a strict fashion assistant. Determine if the following user query is related to fashion, clothing, outfits, styling, accessories, or shopping for clothes.

User query: "{query}"

Respond with ONLY a single word: "YES" if it is fashion-related, or "NO" if it is not.
Examples of fashion-related: "I need a party outfit", "suggest shoes for a wedding", "what to wear to the office"
Examples of NOT fashion-related: "what is 2+2", "write a poem", "who is the president", "tell me a joke"

Answer:"""
        result = query_gemini_api(validation_prompt, api_key).strip().upper()
        return result.startswith("YES")

    # Fallback: keyword-based check
    return any(kw in q_lower for kw in FASHION_KEYWORDS)


def parse_query_with_gemini(query: str, products_list: list[dict], api_key: str) -> dict:
    products_summary = "\n".join([
        f"- ID: {p['id']} | Name: {p['name']} | Category: {p['category_label']} | Occasion: {p['occasion']} | Gender: {p['gender']}"
        for p in products_list
    ])
    
    prompt = f"""
You are an expert fashion stylist AI assistant.
The user is asking: "{query}"

Review this list of products from our database:
{products_summary}

Your job is to select the single best "anchor" product ID from our database that matches the user's intent to base a complete outfit recommendation on.
Also identify the user's intended gender and occasion from the query.

You must output your response in JSON format. The JSON must contain exactly these fields:
1. "detected_gender": "men" or "women"
2. "detected_occasion": one of "casual", "office", "wedding", "party", "festive", "sports", "vacation"
3. "anchor_product_id": the selected product ID string (must be exactly one of the IDs listed above)

JSON output:
"""
    response_text = query_gemini_api(prompt, api_key)
    try:
        # Strip code block decorators if present
        clean_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_text)
        return parsed
    except Exception as e:
        st.warning("Failed to parse Gemini JSON response. Falling back to local search.")
        return None


# Local Fallback Rule-Based Parser
def parse_query_local(query: str, products_df: pd.DataFrame, default_gender: str) -> dict:
    q = query.lower()
    
    # 1. Detect Gender
    gender = default_gender.lower()
    if any(w in q for w in ["women", "female", "girl", "she", "her", "saree", "dress"]):
        gender = "women"
    elif any(w in q for w in ["men", "male", "boy", "he", "him", "sherwani"]):
        gender = "men"

    # 2. Detect Occasion
    occasion = "casual"
    if any(w in q for w in ["wedding", "marriage", "bride", "groom", "sherwani"]):
        occasion = "wedding"
    elif any(w in q for w in ["office", "work", "meeting", "formal", "interview"]):
        occasion = "office"
    elif any(w in q for w in ["party", "club", "night", "cocktail"]):
        occasion = "party"
    elif any(w in q for w in ["beach", "vacation", "trip", "summer", "sun", "linen"]):
        occasion = "vacation"
    elif any(w in q for w in ["festive", "diwali", "eid", "festival"]):
        occasion = "festive"
    elif any(w in q for w in ["sport", "workout", "gym", "run", "active"]):
        occasion = "sports"

    # 3. Detect Category keyword and select anchor product
    filtered_df = products_df[products_df["gender"].isin([gender, "unisex"])]
    
    # Search by category keywords
    candidates = []
    keywords_map = {
        "shirt": ["shirt", "formal-shirts", "casual-shirts", "linen-shirts"],
        "t-shirt": ["tshirt", "polo-tshirts", "tshirts"],
        "dress": ["dress", "party-dresses", "casual-dresses", "maxi-dresses"],
        "saree": ["saree", "wedding-sarees"],
        "suit": ["suit", "suits"],
        "pant": ["trouser", "jeans", "chinos", "shorts", "pants"],
        "shoes": ["shoes", "sneakers", "running-shoes", "heels", "loafers", "sandals"],
        "necklace": ["necklace", "necklaces"]
    }
    
    found_keyword = False
    for kw, categories in keywords_map.items():
        if kw in q:
            matches = filtered_df[
                filtered_df["category"].isin(categories) | 
                filtered_df["category_label"].str.lower().str.contains(kw)
            ]
            if not matches.empty:
                candidates = matches["id"].astype(str).tolist()
                found_keyword = True
                break

    if not found_keyword:
        # Filter by occasion
        occ_matches = filtered_df[filtered_df["occasion"] == occasion]
        if not occ_matches.empty:
            candidates = occ_matches["id"].astype(str).tolist()
        else:
            candidates = filtered_df["id"].astype(str).tolist()

    anchor_id = random.choice(candidates) if candidates else str(products_df.iloc[0]["id"])
    
    return {
        "detected_gender": gender,
        "detected_occasion": occasion,
        "anchor_product_id": anchor_id,
        "stylist_reasoning": ""
    }


# Local Fallback Stylist Rationale Generator
def generate_stylist_rationale_local(outfit: dict, occasion: str, style_pref: str) -> str:
    anchor_name = outfit["anchor"]["name"]
    brand = outfit["anchor"]["brand"]
    
    if not outfit["outfit_items"]:
        return f"This **{anchor_name}** by **{brand}** is a great choice for a {occasion.capitalize()} look."
        
    recs_names = [item["name"] for item in outfit["outfit_items"]]
    recs_str = ", ".join(recs_names[:-1]) + f", and {recs_names[-1]}" if len(recs_names) > 1 else recs_names[0]
    
    return (
        f"This **{anchor_name}** by **{brand}** serves as the focal point for this {occasion.capitalize()} ensemble. "
        f"To build a balanced, {style_pref.lower()} silhouette, our AI paired it with **{recs_str}**. "
        f"The combination contrasts styles while keeping visual weights consistent, ensuring a sleek, coordinated look that meets the occasion's requirements."
    )


# --- SIDEBAR: USER PROFILE & SETTINGS ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/gender-neutral-user.png", width=70)
    st.title("Stylist Profile")
    st.markdown("---")
    
    # 1. User Info
    gender = st.selectbox("Gender Preference", ["Men", "Women"], index=0)
    age = st.slider("User Age", 18, 70, 24)
    style_preference = st.selectbox("Style Direction", ["Formal", "Smart Casual", "Casual"], index=0)
    occasion = st.selectbox("Occasion Type", ["Office", "Wedding", "Casual", "Party", "Vacation", "Sports", "Festive"], index=0)
    
    st.markdown("---")
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Enter Gemini API Key (optional)",
        value=env_api_key,
        type="password",
        help="If provided, unlocks live Gemini LLM outfit generation & reasoning. Otherwise, runs locally."
    )
    
    st.markdown("---")
    st.caption("Developed for Dare XAI Recommendation Prototype. Powered by PyTorch Siamese Net & DistilBERT/ResNet50.")


# --- MAIN INTERFACE ---
st.title("👔 AI Fashion Assistant & Recommendation Engine")
st.markdown("Evaluate compatibility, customize outfit layouts, and chat with your personal stylist.")

tab1, tab2 = st.tabs(["✨ Outfit Builder Showcase", "💬 Conversational Assistant"])

# --- TAB 1: OUTFIT BUILDER SHOWCASE ---
with tab1:
    st.subheader("Select an Anchor Product")
    st.markdown("Start with a single product, and the Siamese Network will recommend a complete compatible outfit.")

    # Filter products by selected profile gender
    gender_req = gender.lower()
    df_filtered = recommender.products_df[recommender.products_df["gender"].isin([gender_req, "unisex"])]

    # Group options for selector
    product_options = {}
    for _, row in df_filtered.iterrows():
        label = f"{row['brand']} - {row['name']} (INR {row['price_inr']})"
        product_options[label] = str(row["id"])

    selected_label = st.selectbox("Choose Product:", list(product_options.keys()))
    selected_id = product_options[selected_label]

    # Show selected product details
    selected_prod = df_filtered[df_filtered["id"] == selected_id].iloc[0]
    
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        img_path = Path(selected_prod["image"])
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning("Product photo not found.")
            
    with col2:
        st.markdown(f"### {selected_prod['name']}")
        st.markdown(f"**Brand:** {selected_prod['brand']} | **Category:** {selected_prod['category_label']}")
        st.markdown(f"**Price:** <span class='product-price'>INR {selected_prod['price_inr']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Occasion:** {selected_prod['occasion'].title()}")
        st.markdown(f"**Description:** {selected_prod['description']}")
        
        generate_btn = st.button("Generate Compatible Outfit")

    if generate_btn:
        st.markdown("---")
        st.subheader("Recommended Compatible Outfit")
        
        # Get outfit recommendations
        outfit = recommender.recommend_full_outfit(
            anchor_id=selected_id,
            gender_filter=gender_req,
            occasion_filter=occasion
        )
        
        # Display recommended items in columns
        recommended_items = outfit["outfit_items"]
        cols = st.columns(len(recommended_items) + 1)
        
        # Anchor column first
        with cols[0]:
            st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
            img_path = Path(selected_prod["image"])
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            st.markdown(f"<div class='product-title'>{selected_prod['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='product-brand'>{selected_prod['brand']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='product-price'>INR {selected_prod['price_inr']}</div>", unsafe_allow_html=True)
            st.caption("⚓ Anchor Item")
            st.markdown("</div>", unsafe_allow_html=True)

        # Recommendation columns
        for idx, item in enumerate(recommended_items):
            with cols[idx + 1]:
                st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                img_path = Path(item["image"])
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                st.markdown(f"<div class='product-title'>{item['name']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='product-brand'>{item['brand']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='product-price'>INR {item['price_inr']}</div>", unsafe_allow_html=True)
                st.caption(f"✨ Compatible {recommender.get_high_level_category(pd.Series(item))}")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"**Total Outfit Price:** <span class='product-price'>INR {outfit['total_price_inr']}</span>", unsafe_allow_html=True)

        # Stylist Commentary (Gemini or local fallback)
        st.markdown("<div class='stylist-box'>", unsafe_allow_html=True)
        st.markdown("#### 🖊️ Stylist Commentary")
        
        if api_key:
            with st.spinner("Asking AI Stylist..."):
                prompt = f"""
You are an expert fashion stylist. Explain why this outfit matches and fits a {age}-year-old {gender_req} for a {occasion} occasion in a {style_preference} style.
Anchor: {selected_prod['name']} by {selected_prod['brand']}.
Compatible items: {', '.join([item['name'] + ' by ' + item['brand'] for item in recommended_items])}.
Explain the color coordination, suitability, and stylistic harmony in a friendly, professional paragraph.
"""
                rationale = query_gemini_api(prompt, api_key)
        else:
            rationale = generate_stylist_rationale_local(outfit, occasion, style_preference)
            
        st.markdown(rationale)
        st.markdown("</div>", unsafe_allow_html=True)


# --- TAB 2: CONVERSATIONAL ASSISTANT ---
with tab2:
    st.subheader("Chat with the AI Fashion Stylist")
    st.markdown("Ask for outfit recommendations using natural language! Example: *'I need a formal outfit for an interview'* or *'Suggest a stylish holiday look for women'*.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "outfit" in message:
                outfit = message["outfit"]
                # Render visual outfit columns in chat
                cols = st.columns(len(outfit["outfit_items"]) + 1)
                
                # Anchor
                with cols[0]:
                    st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                    img_path = Path(outfit["anchor"]["image"])
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    st.markdown(f"<div class='product-title'>{outfit['anchor']['name']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='product-price'>INR {outfit['anchor']['price_inr']}</div>", unsafe_allow_html=True)
                    st.caption("Anchor")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommendations
                for idx, item in enumerate(outfit["outfit_items"]):
                    with cols[idx + 1]:
                        st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                        img_path = Path(item["image"])
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        st.markdown(f"<div class='product-title'>{item['name']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='product-price'>INR {item['price_inr']}</div>", unsafe_allow_html=True)
                        st.caption(f"Compatible {recommender.get_high_level_category(pd.Series(item))}")
                        st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(f"**Total Price:** INR {outfit['total_price_inr']}")

    # User Input
    if prompt := st.chat_input("Tell me what outfit you need (e.g. 'I need a formal look for a wedding'):"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # ── Input Guard: Reject off-topic queries ──────────────────────────
        with st.spinner("Checking your request..."):
            is_valid = is_fashion_query(prompt, api_key)

        if not is_valid:
            with st.chat_message("assistant"):
                st.markdown("""
<div style="background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.05));
            border-left: 4px solid #ef4444;
            border-radius: 0 10px 10px 0;
            padding: 16px 20px;
            margin: 4px 0;">
<strong>❌ Invalid Query</strong><br><br>
I'm a fashion stylist AI — I can only help with outfit recommendations, styling advice, and clothing suggestions.
<br><br>
<em>Try something like:</em>
<ul>
  <li>"I need a formal outfit for a job interview"</li>
  <li>"Suggest a party look for women"</li>
  <li>"What should I wear to a beach vacation?"</li>
</ul>
</div>
""", unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ **Invalid Query** — I only handle fashion and outfit recommendations. Please ask me about clothing, outfits, or styling!"
            })
        else:
            # Process Response
            with st.chat_message("assistant"):
                with st.spinner("Stylist is thinking..."):
                    parsed_params = None
                    
                    if api_key:
                        # Parse using Gemini
                        products_list = recommender.products_df.to_dict(orient="records")
                        parsed_params = parse_query_with_gemini(prompt, products_list, api_key)
                    
                    if parsed_params is None:
                        # Fallback to local heuristic
                        parsed_params = parse_query_local(prompt, recommender.products_df, gender)

                    det_gender = parsed_params.get("detected_gender", gender.lower())
                    det_occasion = parsed_params.get("detected_occasion", occasion.lower())
                    anchor_id = parsed_params.get("anchor_product_id")
                    
                    # Recommend full outfit
                    outfit = recommender.recommend_full_outfit(
                        anchor_id=anchor_id,
                        gender_filter=det_gender,
                        occasion_filter=det_occasion
                    )
                    
                    # Generate rationale
                    if api_key:
                        rationale_prompt = f"""
You are an AI Stylist. The user asked: "{prompt}".
We selected this Anchor item based on their request: {outfit['anchor']['name']} by {outfit['anchor']['brand']}.
Our AI model recommended these compatible items to complete the outfit: {', '.join([item['name'] + ' by ' + item['brand'] for item in outfit['outfit_items']])}.

Write a stylish, engaging, and professional paragraph explaining why this EXACT combination works perfectly for a {det_occasion} occasion.
Do not recommend or mention items that are not in this list. Focus on color coordination, suitability, and stylistic harmony of the provided items.
"""
                        rationale = query_gemini_api(rationale_prompt, api_key)
                    else:
                        rationale = generate_stylist_rationale_local(outfit, det_occasion, style_preference)
                    
                    response_text = (
                        f"### Stylist Recommendations for your request:\n"
                        f"**Context**: {det_gender.capitalize()} | Occasion: {det_occasion.capitalize()}\n\n"
                        f"{rationale}\n\n"
                        f"Here is the compatible outfit combo built for you:"
                    )
                    st.markdown(response_text)
                    
                    # Render visual columns
                    cols = st.columns(len(outfit["outfit_items"]) + 1)
                    
                    # Anchor
                    with cols[0]:
                        st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                        img_path = Path(outfit["anchor"]["image"])
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        st.markdown(f"<div class='product-title'>{outfit['anchor']['name']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='product-price'>INR {outfit['anchor']['price_inr']}</div>", unsafe_allow_html=True)
                        st.caption("Anchor")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Recommendations
                    for idx, item in enumerate(outfit["outfit_items"]):
                        with cols[idx + 1]:
                            st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                            img_path = Path(item["image"])
                            if img_path.exists():
                                st.image(str(img_path), use_container_width=True)
                            st.markdown(f"<div class='product-title'>{item['name']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='product-price'>INR {item['price_inr']}</div>", unsafe_allow_html=True)
                            st.caption(f"Compatible {recommender.get_high_level_category(pd.Series(item))}")
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"**Total Price:** INR {outfit['total_price_inr']}")
                    
                    # Save assistant message including the outfit dict for re-rendering
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "outfit": outfit
                    })

