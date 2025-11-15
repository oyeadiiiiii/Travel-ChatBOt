import pandas as pd
import random
import joblib
from rapidfuzz import process, fuzz
import os

# --- Load trained intent model ---
intent_clf = joblib.load("intent_clf.pkl")

# --- Load packages and FAQs ---
packages = pd.read_csv("packages.csv")
packages['type'] = packages['type'].str.lower()
packages['destination'] = packages['destination'].str.lower()

faqs = pd.read_csv("faq.csv")
faqs['question'] = faqs['question'].str.lower()
faqs['answer'] = faqs['answer']

# --- Helper functions ---
def clean_text(text):
    return text.lower().strip()

def extract_package_type(user_input):
    types = ['beach', 'adventure', 'honeymoon', 'family', 'budget']
    for t in types:
        if t in user_input.lower():
            return t
    return None

def find_best_faq(user_input):
    question_list = faqs['question'].tolist()
    best_match, score, idx = process.extractOne(user_input, question_list, scorer=fuzz.token_sort_ratio)
    return faqs.iloc[idx]['answer']

def find_best_packages(user_input, top_k=3):
    package_type = extract_package_type(user_input)
    if package_type:
        filtered = packages[packages['type'].str.contains(package_type, case=False)]
    else:
        filtered = packages.copy()
    
    if filtered.empty:
        return []
    
    package_list = (filtered['type'] + ' ' + filtered['destination'] + ' ' + filtered['description']).tolist()
    matches = process.extract(user_input, package_list, scorer=fuzz.token_sort_ratio, limit=top_k)
    
    recommended = []
    for match_text, score, idx in matches:
        recommended.append(filtered.iloc[idx])
    return recommended

# --- Chatbot response and loop ---
EXIT_KEYWORDS = ['bye', 'exit', 'quit', 'close', 'goodbye', 'see you']
booking_state = {}

def get_response(user_input, booking_state):
    text = clean_text(user_input)
    if any(word in text for word in EXIT_KEYWORDS):
        return 'Goodbye! Have a great trip!', None

    if booking_state.get('step') == 'destination':
        dest_input = text
        matched_pkg = packages[packages['destination'].str.contains(dest_input, case=False)]
        if matched_pkg.empty:
            return 'Sorry, no package for that destination. Please enter another:', booking_state
        booking_state['package'] = matched_pkg.iloc[0].to_dict()
        booking_state['step'] = 'members'
        return 'Great! How many travellers?', booking_state

    if booking_state.get('step') == 'members':
        try:
            members = int(text)
        except:
            return 'Please enter a valid number of travellers:', booking_state
        booking_state['members'] = members
        total_price = members * int(booking_state['package']['price'])
        booking_state['total_price'] = total_price
        booking_file = 'bookings.csv'
        booking_df = pd.DataFrame([booking_state])
        if os.path.exists(booking_file):
            booking_df.to_csv(booking_file, mode='a', index=False, header=False)
        else:
            booking_df.to_csv(booking_file, index=False)
        response = f"Booking confirmed for {members} traveller(s) to {booking_state['package']['destination']}. Total price: ₹{total_price}. Thank you!"
        return response, None

    intent = intent_clf.predict([text])[0]

    if intent == 'greet':
        return random.choice(["Hi! How can I help you plan your trip?", "Hello! Ready to explore destinations?"]), booking_state
    if intent == 'ask_faq':
        answer = find_best_faq(user_input)
        return f"Here's what I found:\n- {answer}", booking_state
    if intent == 'recommend':
        recommended_pkgs = find_best_packages(user_input)
        if recommended_pkgs:
            response = 'You might love these destinations:'
            for pkg in recommended_pkgs:
                response += f"\n- **{pkg['destination']}**: A {pkg['type']} trip. {pkg['description']} (Approx. ₹{pkg['price']})"
            return response, booking_state
        else:
            return 'Sorry, no matching package found.', booking_state
    if intent == 'book':
        booking_state['step'] = 'destination'
        return 'Sure! Which destination would you like to book?', booking_state

    return "I'm sorry, I don't understand. You can ask me to recommend trips, answer FAQs, or help you book a package.", booking_state

# --- Chat loop ---
def start_chat():
    print("Bot: Hi! Ready to plan your next trip?\n")
    booking_state = {}
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        response, booking_state = get_response(user_input, booking_state)
        print(f"\nBot: {response}\n")
        if booking_state is None or any(word in user_input.lower() for word in EXIT_KEYWORDS):
            break

if __name__ == '__main__':
    start_chat()
