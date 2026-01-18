import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import random
from datetime import datetime
from collections import Counter

st.title("Cognitive Distortion Identifier")
st.subheader("Cognitive distortion: Untrue thoughts that make you perceive the situation more negativiely")

@st.cache_resource #to train the model only once even when refreshed page
def load_models():
    #Training first model
    st.info("Training Model I")
    df = pd.read_csv("data_CogD.csv")

    X =df['Thought']
    y =df['Cognitive_Distortion']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    textVectorizer_thoughts = TfidfVectorizer(max_features=500,stop_words="english")
    X_train = textVectorizer_thoughts.fit_transform(X_train)
    X_test = textVectorizer_thoughts.transform(X_test)

    SVM=LinearSVC(C=1)
    SVM.fit(X_train,y_train)
    y_pred=SVM.predict(X_test)
    accuracy1=accuracy_score(y_test,y_pred)

    st.success(f"Model I trained. Accuracy: {accuracy1:.2%}")

    #Training second model
    st.info("Training Model II")
    df_res = pd.read_csv("data_res.csv")

    db = {}
    print("DB TYPE:", type(db))
    print("DB KEYS:", db.keys())
    response_database = {}

    for distortion in df_res['Cognitive_Distortion'].unique():
        responses = df_res[
            df_res["Cognitive_Distortion"] == distortion
        ]['Response'].tolist()
    
        vectorizer_res = TfidfVectorizer(
            max_features=200,
            stop_words='english'
        )

        vectorized_res = vectorizer_res.fit_transform(responses)

        KNN = NearestNeighbors( 
        n_neighbors=min(3,len(responses)),
        metric = 'cosine' #using cosine distance
        )

        KNN.fit(vectorized_res)

        response_database[distortion] = {
            'KNN': KNN, #store model to find similar responses
            'responses': responses,
            'vectors': vectorized_res,
            'vectorizer': vectorizer_res, #store vectorizer to convert new text
        }

    st.success(f"Model II trained. {len(response_database)} distortion types.")

    return SVM, textVectorizer_thoughts, label_encoder, accuracy1, response_database #stop execution and send value back

def generate_hybrid_res(thought, distortion, response_database):
    if distortion not in response_database:
        return "I understand you're going through something difficult. Can you describe a thought you are having?"
    
    db = response_database[distortion]

    thought_vector = db['vectorizer'].transform([thought])
    distances, indices = db['KNN'].kneighbors(thought_vector)
    similar_responses = [db['responses'][idx] for idx in indices[0]]

    starters =[]
    middlers = []
    endings = []

    for response in similar_responses:
        sentences = response.split('.') #split responses into sentences

        if len(sentences)>=1: #extract first sentence
            starters.append(sentences[0].strip()+'.')
        if len(sentences)>=2:
            middlers.append(sentences[1].strip()+'.')
        if len(sentences)>=3:
            endings.append(sentences[2].strip()+'.')
    
    base_response =""
    if starters:
        base_response = random.choice(starters)
        if middlers and random.random() > 0.3:
            base_response += " " + random.choice(middlers)
        if endings and random.random() > 0.7:
            base_response += " " + random.choice(endings)
    else:
        base_response = "I hear you, please explain the thought you are having."
    
    user_words = set(thought.lower().split()) #extract meaningful words from user's thought

    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}

    key_words = user_words - common_words #take only meaningful words

    if key_words and random.random() > 0.5:
        key_word = random.choice(list(key_words))

        personalizations = [
            f"I notice you mentioning '{key_word}'. Can you tell me more about that?",
            f"You used the word '{key_word}'. What does that mean to you?",
            f"Let's explore '{key_word}' more. What feelings are brought up?"
        ]

        base_response += " " + random.choice(personalizations) #add random personlization choice
    return base_response


def analyze_thought(thought, SVM, vectorizer, label_encoder, response_database):
    thought_tfidf = vectorizer.transform([thought])

    #get probability for each distortion type
    decision_scores = SVM.decision_function(thought_tfidf)[0] 


    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores/ exp_scores.sum()

    sorted_indices = np.argsort(probabilities)[::-1]
    top_3_indices = sorted_indices[:3]

    results = []
    for idx in top_3_indices:

        distortion = label_encoder.classes_[idx] #convert index back to distortion name
        confidence = probabilities[idx] #get confidence %
        results.append({
            'distortion':distortion,
            'confidence':confidence,
        })
    
    top_distortion = label_encoder.classes_[sorted_indices[0]]
    response = generate_hybrid_res(thought, top_distortion, response_database)

    return results, response


if 'models_loaded' not in st.session_state:
    with st.spinner('Training models... taking a moment...'):
        SVM, vectorizer, label_encoder, accuracy1, response_db = load_models()
        
        #store in session state so we don't reload
        st.session_state.SVM = SVM
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = label_encoder
        st.session_state.accuracy = accuracy1
        st.session_state.response_db= response_db
        st.session_state.models_loaded = True

if 'history' not in st.session_state: #initialize conversation history
    st.session_state.history =[]

# UI
with st.sidebar:
    st.header("About")
    st.info("""
    ***How this works***
    1. Share a thought that's bothering you
    2. AI classifies the cognitive distortion ex: Labeling
    3. AI generates a response that's unique and personlized
    """)


    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("""
        :rainbow[Trained] with data on cognitive distortions.
        """)

st.text("🌻 Share your thought.")

thought_input = st.text_area(
    "What's on your mind?", 
    placeholder="Example: I'm such a failure at everything...",
    height=80,
    label_visibility="collapsed"
)

col1, col2 = st.columns([1,2])

with col1:
    analyze_button = st.button("Analyze", use_container_width=True)

if analyze_button and thought_input.strip():
    results, response = analyze_thought(
        thought_input,
        st.session_state.SVM,
        st.session_state.vectorizer,
        st.session_state.label_encoder,
        st.session_state.response_db
    )

    top_result = results[0]

#store in history
    st.session_state.history.append({
        'timestamp':datetime.now(),
        'thought':thought_input,
        'distortion': top_result['distortion'],
        'confidence':top_result['confidence']
    })

    st.markdown("---")

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown(f"##{top_result['distortion']}")
    with col2:
        st.metric("Confidence",f"{top_result['confidence']:.0%}")


    st.markdown("### Response")
    st.write(response)

    if top_result['confidence']<0.7:
        st.markdown("### Alternative Possibilities")
        cols = st.columns(3)
        for i, (col, result) in enumerate(zip(cols,results)):
            with col:
                st.markdown(f"""
                <div style=background-color: #4a4d52; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4> {['1.','2.','3.'][i]} {result['distortion']}</h4>
                    <h2>{result['confidence']:.0%}</h2>
                </div>
                """, unsafe_allow_html=True)

if st.session_state.history:
    st.markdown("---")
    st.subheader("Your Thought History")

    distortions = [entry['distortion'] for entry in st.session_state.history]
    distortion_counts = Counter(distortions)

    st.markdown("### Your patterns")

    chart_data = pd.DataFrame({
        'Distortion':list(distortion_counts.keys()),
        'Count':list(distortion_counts.values())
    })

    st.bar_chart(chart_data.set_index('Distortion'))

    st.markdown("### Timeline")
    for i, entry in enumerate(reversed(st.session_state.history),1):
        with st.expander(f"{entry['timestamp'].strftime('%H:%M:%S')}-{entry['thought'][:50]}..."):
            st.write(f"**Distortion:** {entry['distortion']}")
            st.write(f"**Confidence:** {entry['confidence']:.0%}")
            st.write(f"**Full thought:** {entry['thought']}")
