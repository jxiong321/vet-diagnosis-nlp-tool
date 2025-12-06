# streamlit web interface for the vet diagnosis tool

import streamlit as st
import json
import pickle
from src.bm25_retriever import BM25Retriever
from src.blended_system import BlendedDiagnosisSystem
from src.explanation_generator import ExplanationGenerator
from src.symptom_extractor import SymptomExtractor
from src.log_regression import MulticlassTargetEncoder, SymptomMultiHot


@st.cache_resource
def load_system():
    # load all components
    retriever = BM25Retriever(
        symptoms_path='data/symptoms.json',
        conditions_path='data/conditions.json'
    )

    passages = []
    with open('data/passages.jsonl', 'r') as f:
        for line in f:
            passages.append(json.loads(line))
    retriever.index(passages)

    with open('src/trained_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    # load weights if they exist
    try:
        with open('best_weights.json', 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        weights = {'retrieval': 1/3, 'classifier': 1/3, 'rules': 1/3}

    system = BlendedDiagnosisSystem(retriever, classifier, weights=weights)
    explainer = ExplanationGenerator()
    extractor = SymptomExtractor()

    return system, explainer, extractor, weights


def main():
    st.set_page_config(page_title="Vet Diagnosis Tool", page_icon="🐕", layout="wide")

    st.title("🐕 Veterinary Diagnosis NLP Tool")
    st.markdown("*An NLP system combining retrieval, classification, and rule-based reasoning*")

    # load components
    system, explainer, extractor, current_weights = load_system()

    # sidebar for weights
    st.sidebar.header("⚙️ System Configuration")
    st.sidebar.subheader("🎯 Current Weights")
    st.sidebar.write(f"**Retrieval:** {current_weights['retrieval']:.1%}")
    st.sidebar.write(f"**Classifier:** {current_weights['classifier']:.1%}")
    st.sidebar.write(f"**Rules:** {current_weights['rules']:.1%}")

    with st.sidebar.expander("📊 Weight Recommendation"):
        st.markdown("""
        **Grid search optimal:** Classifier 90%, Retrieval 5%, Rules 5%
        - Highest validation accuracy
        - Overly classifier-dependent
        - Poor generalization to new diseases

        **Recommended (current):** Classifier 60%, Retrieval 30%, Rules 10%
        - Balanced performance
        - Better for unknown diseases
        - Maintains textbook evidence
        - More interpretable for vets

        *We recommend balanced weights for better generalizability.*
        """)

    # optional weight adjustment (normalize to sum=1)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adjust Weights (Optional)")

    w_retrieval = st.sidebar.slider("Retrieval Weight", 0.0, 1.0, current_weights['retrieval'], 0.1)
    w_classifier = st.sidebar.slider("Classifier Weight", 0.0, 1.0, current_weights['classifier'], 0.1)
    w_rules = st.sidebar.slider("Rules Weight", 0.0, 1.0, current_weights['rules'], 0.1)

    # normalize
    total = w_retrieval + w_classifier + w_rules
    if total > 0:
        adjusted_weights = {
            'retrieval': w_retrieval / total,
            'classifier': w_classifier / total,
            'rules': w_rules / total
        }
        system.set_weights(adjusted_weights)

    st.sidebar.write(f"**Sum:** {total:.2f} (normalized to 1.0)")

    # main content
    st.header("Enter Patient Notes")

    st.markdown("""
    💡 **Tip:** You can include age, gender, vitals (weight, temp, heart rate), and duration in your query.
    The system will extract these features automatically!
    """)

    # example queries dropdown - showcasing different features
    examples = [
        "6 month old puppy has bloody diarrhea and vomiting, weight 8 kg, temp 39.5 C",
        "2 year old male dog has persistent dry cough and labored breathing, heart rate 120 bpm",
        "young female dog with nasal discharge and eye discharge, duration 3 days",
        "senior dog vomiting after eating, weight 30 kg, not eating for 2 days",
        "adult male dog not eating and lethargic, body temperature 40 C, heart rate 110 bpm",
        "puppy with diarrhea",  # minimal info example
        "3 year old cat with runny nose"  # wrong species example to show blocking
    ]

    selected_example = st.selectbox(
        "Or select an example:",
        [""] + examples,
        help="Try different examples to see how the system extracts features like age, vitals, and symptoms!"
    )

    # text input
    query = st.text_area(
        "Patient notes:",
        value=selected_example if selected_example else "",
        height=100,
        placeholder="e.g., 6 month old puppy has bloody diarrhea and is vomiting"
    )

    # top-k slider
    top_k = st.slider("Number of diagnoses to show:", 1, 5, 3)

    # diagnose button
    if st.button("🔍 Diagnose", type="primary") or query:
        if query.strip():
            # run diagnosis
            results = system.diagnose(query, top_k=top_k)

            # extract features for display
            features = results['features']

            # show extracted features
            st.markdown("---")
            st.subheader("📋 Extracted Features")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Demographics:**")
                st.write(f"- Species: {features.get('species', 'N/A')}")
                st.write(f"- Age: {features.get('Age', 'N/A')} years")
                st.write(f"- Gender: {features.get('Gender', 'N/A')}")

            with col2:
                st.markdown("**Detected Symptoms:**")
                symptom_cols = [
                    'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
                    'Labored_Breathing', 'Lameness', 'Skin_Lesions',
                    'Nasal_Discharge', 'Eye_Discharge'
                ]
                detected = [s.replace('_', ' ') for s in symptom_cols if features.get(s, 0) == 1]

                # also check for fever (detected via body temp)
                if features.get('Body_Temperature_C', 38.5) > 39.5:
                    detected.append('Fever')

                if detected:
                    for symptom in detected:
                        st.write(f"✓ {symptom}")
                else:
                    st.write("(none detected)")

            # vitals
            with st.expander("📊 Vitals (defaults shown if not specified)"):
                st.write(f"- Weight: {features.get('Weight', 'N/A')} kg")
                st.write(f"- Heart Rate: {features.get('Heart_Rate', 'N/A')} bpm")
                st.write(f"- Body Temperature: {features.get('Body_Temperature_C', 'N/A')} °C")
                st.write(f"- Duration: {features.get('duration_days', 'N/A')} days")

            # show diagnoses
            st.markdown("---")
            st.subheader("🏥 Top Diagnoses")

            blended_top = results['blended_top']
            blended_scores = results['blended_scores']

            for rank, disease in enumerate(blended_top, 1):
                score = blended_scores[disease]

                # get component scores
                r_score = results['retrieval_scores'].get(disease, 0)
                c_score = results['classifier_scores'].get(disease, 0)
                ru_score = results['rule_scores'].get(disease, 0)

                # generate explanation
                explanation = explainer.generate_diagnosis_explanation(
                    disease=disease,
                    query=query,
                    scores={
                        'retrieval': r_score,
                        'classifier': c_score,
                        'rules': ru_score,
                        'blended': score
                    },
                    features=features
                )

                # display diagnosis card
                with st.container():
                    # Header with confidence badge
                    badge_emoji, badge_text = _get_confidence_badge(score)
                    st.markdown(f"### #{rank} {disease} {badge_emoji}")
                    st.caption(f"**{badge_text}** (Blended Score: {score:.3f})")

                    st.write(explanation['summary'])

                    # Visual score bars
                    st.markdown("**Component Scores:**")
                    st.markdown(_create_score_bar(r_score, "📚 Retrieval (Textbook Match)"), unsafe_allow_html=True)
                    st.markdown(_create_score_bar(c_score, "🧮 Classifier (Statistical Model)"), unsafe_allow_html=True)
                    st.markdown(_create_score_bar(ru_score, "✅ Rules (Symptom Validation)"), unsafe_allow_html=True)
                    st.markdown(_create_score_bar(score, "🎯 BLENDED SCORE"), unsafe_allow_html=True)

                    # Show how blending works
                    with st.expander("🔢 How is the blended score calculated?"):
                        st.markdown(_visualize_blending(r_score, c_score, ru_score, score, adjusted_weights),
                                    unsafe_allow_html=True)

                    # Enhanced detailed interpretation
                    with st.expander("📖 What do these scores mean? (Plain English)"):
                        st.markdown("#### Retrieval Score")
                        st.write(_explain_retrieval_score(r_score))

                        st.markdown("#### Classifier Score")
                        st.write(_explain_classifier_score(c_score))

                        st.markdown("#### Rules Score")
                        st.write(_explain_rules_score(ru_score))

                    # supporting evidence
                    with st.expander("📚 Supporting Evidence from Textbook"):
                        if explanation['evidence_sentences']:
                            for i, sent in enumerate(explanation['evidence_sentences'], 1):
                                st.write(f"**{i}.** {sent['text']}")
                                st.caption(f"Source: {sent['source']}")
                        else:
                            st.write("No textbook evidence found.")

                    st.markdown("---")

            # component comparison table
            st.subheader("🔬 Component Comparison")
            st.markdown("*How each component ranked the diseases:*")

            # build comparison data
            comparison_data = []
            for disease in blended_top:
                comparison_data.append({
                    'Disease': disease,
                    'Retrieval Rank': _get_rank(disease, results['retrieval_top']),
                    'Classifier Rank': _get_rank(disease, results['classifier_top']),
                    'Rules Rank': _get_rank(disease, results['rules_top']),
                    'Blended Rank': _get_rank(disease, blended_top)
                })

            st.table(comparison_data)

        else:
            st.warning("Please enter patient notes to diagnose.")

    # footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. "
               "Always consult a licensed veterinarian for actual diagnoses.")


def _get_rank(disease, ranked_list):
    """Helper to get rank of disease in a list."""
    try:
        return ranked_list.index(disease) + 1
    except ValueError:
        return "-"


def _get_score_color(score):
    """Return color based on score value."""
    if score >= 0.7:
        return "#28a745"  # green
    elif score >= 0.4:
        return "#ffc107"  # yellow
    else:
        return "#dc3545"  # red


def _get_confidence_badge(score):
    """Return emoji badge and text for confidence level."""
    if score >= 0.7:
        return "🟢", "High Confidence"
    elif score >= 0.4:
        return "🟡", "Moderate Confidence"
    else:
        return "🔴", "Low Confidence"


def _explain_retrieval_score(score):
    """Plain English explanation of retrieval score for non-statisticians."""
    if score >= 0.7:
        return ("This disease **strongly matches** the textbook description based on keyword overlap. "
                "The patient's symptoms align very well with documented cases.")
    elif score >= 0.4:
        return ("This disease **moderately matches** the textbook description. "
                "Some of the patient's symptoms appear in documented cases.")
    else:
        return ("This disease has **weak textbook support**. "
                "Few of the patient's symptoms match documented cases.")


def _explain_classifier_score(score):
    """Plain English explanation of classifier probability."""
    if score >= 0.6:
        return ("The statistical model gives this diagnosis a **high probability** based on patterns "
                "learned from thousands of similar cases. This is a strong statistical match.")
    elif score >= 0.3:
        return ("The statistical model gives this diagnosis a **moderate probability**. "
                "Some symptom patterns match what the model learned from training data.")
    else:
        return ("The statistical model gives this diagnosis a **low probability**. "
                "The symptom pattern doesn't strongly match cases the model was trained on.")


def _explain_rules_score(score):
    """Plain English explanation of rules validation."""
    if score == 0.0:
        return ("**BLOCKED**: This diagnosis is ruled out because the patient species doesn't match "
                "the disease type (e.g., canine disease for a cat patient).")
    elif score >= 0.9:
        return ("**Strong symptom match**: Almost all expected symptoms for this disease are present "
                "in the patient, and vice versa. High clinical alignment.")
    elif score >= 0.7:
        return ("**Good symptom match**: Most expected symptoms align between the disease profile "
                "and the patient presentation.")
    elif score > 0.0:
        return ("**Partial symptom match**: Some symptoms align, but there are gaps. "
                "Either the patient has symptoms not typical of this disease, or lacks key symptoms.")
    else:
        return ("**No symptom overlap**: The patient's symptoms don't match this disease's typical presentation.")


def _create_score_bar(score, label):
    """Create a visual progress bar for a score."""
    color = _get_score_color(score)
    percentage = score * 100

    # HTML/CSS for a nice progress bar
    html = f"""
    <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <span style="font-size: 14px; font-weight: 600;">{label}</span>
            <span style="font-size: 14px; font-weight: 600;">{score:.3f}</span>
        </div>
        <div style="background-color: #e9ecef; border-radius: 5px; height: 20px; overflow: hidden;">
            <div style="background-color: {color}; width: {percentage}%; height: 100%;
                        border-radius: 5px; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """
    return html


def _visualize_blending(r_score, c_score, ru_score, blended, weights):
    """Show visual breakdown of how blended score is calculated."""
    html = f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 10px;">
        <div style="font-weight: 600; margin-bottom: 10px; font-size: 14px;">
            How the blended score is calculated:
        </div>
        <div style="font-family: monospace; font-size: 13px; line-height: 1.8;">
            ({weights['retrieval']:.3f} × {r_score:.3f}) +
            ({weights['classifier']:.3f} × {c_score:.3f}) +
            ({weights['rules']:.3f} × {ru_score:.3f})
            <br/>
            = ({weights['retrieval'] * r_score:.3f}) +
              ({weights['classifier'] * c_score:.3f}) +
              ({weights['rules'] * ru_score:.3f})
            <br/>
            <strong>= {blended:.3f}</strong>
        </div>
        <div style="font-size: 12px; color: #6c757d; margin-top: 8px;">
            Each component is weighted and combined to produce the final confidence score.
        </div>
    </div>
    """
    return html


if __name__ == "__main__":
    main()
