import sys
import os
import logging
import json
import re
import pickle
import numpy as np
import torch  # <--- CHANGED: Use PyTorch
from flask import Flask, request, jsonify, render_template
from transformers import BertForSequenceClassification # <--- CHANGED: PyTorch Model

# Import Logic
from logic import CourseRepository, ContextManager, AcademicAdvisor

app = Flask(__name__)
app.secret_key = "gju_super_secret"

# ==============================================================================
# 🧠 AI & MODEL LOADING
# ==============================================================================

print("⏳ Loading AI Brain... (This might take a moment)")

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load Model (PyTorch)
try:
    model = BertForSequenceClassification.from_pretrained("bert_intent_model")
    model.eval() # Set to eval mode
    print("✅ BERT Model (PyTorch) Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading BERT model: {e}")
    sys.exit(1)

CONFIDENCE_THRESHOLD = 0.45

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def predict_intent(text):
    text_lower = text.lower()
    
    # --- [LAYER 1] Hybrid Rules (The "Reflexes") ---
    # Restore these to catch "Can I take" and "Dr Ismail" accurately
    
    if "prereq" in text_lower or "pre-req" in text_lower:
        return 'ask_prereqs', 1.0
        
    if "can i take" in text_lower or "eligible" in text_lower:
        return 'check_eligibility', 1.0

    if "who teaches" in text_lower or "professor" in text_lower or "instructor" in text_lower or "office" in text_lower or "where is dr" in text_lower or "dr." in text_lower:
        return 'ask_instructor_info', 1.0
        
    if any(w in text_lower for w in ["joke", "laugh", "funny", "humor"]):
        return 'humor', 1.0
        
    if any(w in text_lower for w in ["hi", "hello", "hey", "greetings"]):
        return 'greeting', 1.0
        
    if any(w in text_lower for w in ["bye", "goodbye", "quit", "exit"]):
        return 'goodbye', 1.0

    if text_lower in ["yes", "yep", "yeah", "sure", "ok", "please"]:
        return 'affirm', 1.0
        
    if text_lower in ["no", "nope", "nah", "cancel"]:
        return 'deny', 1.0

    # --- [LAYER 2] BERT Prediction (PyTorch) ---
    clean_input = clean_text(text)
    inputs = tokenizer(clean_input, return_tensors="pt") # "pt" for PyTorch
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    confidence = probs.max().item()
    idx = probs.argmax().item()
    tag = encoder.inverse_transform([idx])[0]

    if confidence >= CONFIDENCE_THRESHOLD:
        return tag, confidence

    # Fallback: Exact Keyword Match
    for intent in intents:
        for pattern in intent["patterns"]:
            if clean_text(pattern) in clean_input:
                return intent["tag"], 0.99

    return "unknown", confidence


# ==============================================================================
# 🎓 ACADEMIC LOGIC
# ==============================================================================

repo = CourseRepository()
ctx_mgr = ContextManager()
advisor = AcademicAdvisor(repo)


def handle_intent(user_id, text, intent_tag, confidence):
    ctx = ctx_mgr.get_context(user_id)
    
    detected_course = repo.normalize_code(text)
    if detected_course and len(detected_course) > 3:
        ctx_mgr.set_last_entity(user_id, 'course', detected_course)
        
    last_course = ctx_mgr.get_last_entity(user_id)['value']

    # --- 🚦 FLOW CONTROL ---
    
    # 1. CLARIFICATION LOOP
    if ctx['status'] == 'waiting_for_specific_course':
        if detected_course:
            target = detected_course
            original_intent = ctx.get('pending_intent')
            ctx['status'] = 'idle'
            ctx['pending_intent'] = None
            intent_tag = original_intent 
        else:
            return "I still didn't catch a course name. Which course are you asking about? (e.g., CS116)"

    # 2. Track & Course Flows
    # Fix: Put "waiting_for_track" checks early to catch "General" / "Cyber" answers
    if ctx['status'] == 'waiting_for_track':
        track = ctx_mgr.extract_track(text)
        if track:
            ctx['track'] = track
            if ctx['passed_courses']:
                ctx['status'] = 'idle'
                return advisor.generate_plan(ctx['track'], ctx['passed_courses'])
            else:
                ctx['status'] = 'waiting_for_courses'
                return f"Got it, **{track}**. Now list your completed courses."
        # If user didn't say a track, maybe they are asking a new question?
        # Let it fall through, or warn them.
        if "general" not in text.lower() and "cyber" not in text.lower() and "data" not in text.lower():
             pass # Fall through to normal intent handling if they changed topic
        else:
             return "Please specify: General, Data Science, or Cybersecurity."

    if ctx['status'] == 'waiting_for_courses':
        added = ctx_mgr.update_passed_courses(user_id, text)
        if added:
            ctx['status'] = 'idle'
            return advisor.generate_plan(ctx['track'], ctx['passed_courses'])
        return "Please list your passed courses so I can plan your schedule."

    if ctx['status'] == 'waiting_for_eligibility':
        target = ctx['target_course']
        passed = ctx_mgr.update_passed_courses(user_id, text)
        if passed:
            ctx['status'] = 'idle'
            result = advisor.check_eligibility(target, ctx['passed_courses'])
            # Smart Follow-up
            if result.startswith("❌"):
                missing_match = re.search(r'(?:missing:|pass:)\s*(?:-|\n)?\s*([A-Z]{2,4}\d{3,5})', result, re.IGNORECASE | re.DOTALL)
                if missing_match:
                    missing_course = missing_match.group(1)
                    if missing_course != target:
                        ctx['status'] = 'waiting_for_prereq_confirmation'
                        ctx['pending_course_check'] = missing_course
                        return result + f"\n\n🤔 **Would you like to check the prerequisites for {missing_course}?** (Yes/No)"
            return result
        return f"I'm checking eligibility for **{target}**. Please list the courses you have passed."


    # --- 🤖 LOGIC LAYER ---

    # Case 1: Instructor Info (Handle separately to fix "Which course?" bug)
   # Case 1: Instructor Info (Check this FIRST)
    if intent_tag == 'ask_instructor_info':
        instructor = repo.fuzzy_find_instructor(text)
        if instructor:
            return f"👨‍🏫 **{instructor['name']}**\nOffice: {instructor['office_location']}\nEmail: {instructor['email']}\nPhone: {instructor['phone']}"
        
        # Only if no instructor is found, check if they meant a course
        target = detected_course if detected_course else last_course
        if target:
             return f"I can't find that professor, but for **{target}**, check the GJU schedule."
        return "I couldn't find that instructor. Please try searching by name (e.g., 'Dr. Ismail')."

   
        

    # Case 2: Course Info & Prereqs
    if intent_tag in ['ask_course_info', 'ask_prereqs']:
        target = detected_course if detected_course else last_course
        
        if not target:
            ctx['status'] = 'waiting_for_specific_course'
            ctx['pending_intent'] = intent_tag 
            return "Which course are you asking about? (e.g., CS116)"
        
        if intent_tag == 'ask_course_info':
            details = repo.get_course_details(target)
            if details:
                return f"📘 **{details['course_code']}**\n{details['course_name']}\n{details['description']}\nCredits: {details['credit_hours']}"
            return f"I couldn't find details for {target}."
            
        if intent_tag == 'ask_prereqs':
            prereqs = repo.get_prerequisites(target)
            if prereqs:
                p_list = ", ".join([f"{p[0]} ({p[1]})" for p in prereqs])
                return f"🔗 **Prerequisites for {target}**:\n{p_list}"
            return f"**{target}** has no prerequisites."

    # Case 3: Eligibility
    if intent_tag == 'check_eligibility':
        target = detected_course if detected_course else last_course
        if target:
            if ctx['passed_courses']:
                return advisor.check_eligibility(target, ctx['passed_courses'])
            else:
                ctx['status'] = 'waiting_for_eligibility'
                ctx['target_course'] = target
                return f"Okay, let's check **{target}**. Please list your completed courses."
        return "Which course do you want to check eligibility for?"

    # Case 4: Follow-up (Yes/No)
    if ctx['status'] == 'waiting_for_prereq_confirmation':
        if intent_tag == 'affirm':
            new_target = ctx.get('pending_course_check')
            ctx['status'] = 'idle' 
            prereqs = repo.get_prerequisites(new_target)
            if prereqs:
                p_list = ", ".join([f"{p[0]} ({p[1]})" for p in prereqs])
                return f"👍 **Good idea.** Here are the prerequisites for **{new_target}**:\n{p_list}"
            return f"**{new_target}** has no prerequisites."
        elif intent_tag == 'deny':
            ctx['status'] = 'idle'
            return "Okay, let me know if you need help with anything else!"

    # Case 5: Semester Plan
    if intent_tag == 'make_schedule' or intent_tag == 'request_advice': # Added request_advice alias
        ctx_mgr.clear_flow(user_id)
        track = ctx_mgr.extract_track(text)
        if track: ctx['track'] = track
        ctx_mgr.update_passed_courses(user_id, text)
        if ctx['track'] and ctx['passed_courses']: return advisor.generate_plan(ctx['track'], ctx['passed_courses'])
        if ctx['track']:
            ctx['status'] = 'waiting_for_courses'
            return f"Planning for **{ctx['track']}**. Which courses have you passed?"
        ctx['status'] = 'waiting_for_track'
        return "Sure! First, are you **Cybersecurity**, **Data Science**, or **General**?"

    # Case 6: Graduation
    if intent_tag == 'graduation_check':
        ctx_mgr.clear_flow(user_id)
        track = ctx_mgr.extract_track(text)
        if track: ctx['track'] = track
        ctx_mgr.update_passed_courses(user_id, text)
        if ctx['track'] and ctx['passed_courses']: return advisor.check_graduation(ctx['track'], ctx['passed_courses'])
        ctx['status'] = 'waiting_for_grad_info'
        return "To check graduation status, I need your **Track** and **Passed Courses**."

    # Fallback from JSON
    for intent in intents:
        if intent['tag'] == intent_tag:
            import random
            return random.choice(intent['responses'])
    
    return "I'm listening, but I'm not sure how to help with that specifically."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=['POST'])
def chat():
    user_id = request.remote_addr
    data = request.json
    msg = data.get('message', '')
    if not msg: return jsonify({"response": "Empty message."})
    
    tag, conf = predict_intent(msg)
    response = handle_intent(user_id, msg, tag, conf)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)