import sys
import os
import logging
from flask import Flask, request, jsonify, render_template
from logic import CourseRepository, ContextManager, AcademicAdvisor, AIModel

app = Flask(__name__)
app.secret_key = "gju_super_secret"

# Initialize
repo = CourseRepository()
ctx_mgr = ContextManager()
advisor = AcademicAdvisor(repo)
ai = AIModel() # Loads the brain you built in Step 2

def handle_intent(user_id, text, intent_tag, confidence):
    ctx = ctx_mgr.get_context(user_id)
    
    # --- 🧠 MEMORY LAYER: Track the Topic ---
    # 1. Look for a course code in the CURRENT message
    detected_course = repo.normalize_code(text)
    
    # 2. If we found one, save it to memory (Update Context)
    if detected_course and len(detected_course) > 3: # Avoid saving garbage
        ctx_mgr.set_last_entity(user_id, 'course', detected_course)
        
    # 3. Retrieve the topic from memory (if we need it later)
    last_course = ctx_mgr.get_last_entity(user_id)['value']


    # --- 🚦 FLOW CONTROL: Handle Multi-step Conversations ---
    
    # Flow: Check Eligibility (Step 2) -> User provides courses
    if ctx['status'] == 'waiting_for_eligibility':
        target = ctx['target_course']
        passed = ctx_mgr.update_passed_courses(user_id, text)
        if passed:
            ctx['status'] = 'idle'
            return advisor.check_eligibility(target, ctx['passed_courses'])
        return f"I'm checking eligibility for **{target}**. Please list the courses you have passed (e.g., CS116, MATH101)."

    # Flow: Graduation Check (Step 2) -> User provides courses
    if ctx['status'] == 'waiting_for_grad_info':
        if not ctx['track']:
            track = ctx_mgr.extract_track(text)
            if track: ctx['track'] = track
        
        ctx_mgr.update_passed_courses(user_id, text)
        
        if ctx['track'] and ctx['passed_courses']:
            ctx['status'] = 'idle'
            return advisor.check_graduation(ctx['track'], ctx['passed_courses'])
        elif not ctx['track']:
            return "First, tell me your track (Cybersecurity, Data Science, or General)."
        else:
            return f"Okay, checking {ctx['track']}. Now list your passed courses."

    # Flow: Semester Planning (Step 2) -> User provides courses
    if ctx['status'] == 'waiting_for_courses':
        added = ctx_mgr.update_passed_courses(user_id, text)
        if added:
            ctx['status'] = 'idle'
            return advisor.generate_plan(ctx['track'], ctx['passed_courses'])
        return "Please list your passed courses so I can plan your schedule."

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
        return "Please specify: General, Data Science, or Cybersecurity."


    # --- 🤖 INTENT HANDLING (Using Memory) ---

    # Case 1: Course Info / Prereqs / Instructors
    # "Tell me about CS116" OR "Who teaches it?"
    if intent_tag in ['ask_course_info', 'ask_prereqs', 'ask_instructor_info']:
        # If user didn't say a course name, use the one from Memory
        target = detected_course if detected_course else last_course
        
        if not target:
            return "Which course are you asking about?"
        
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
            
        if intent_tag == 'ask_instructor_info':
            # Check if user asked for a specific person ("Where is Adam?")
            # We use the raw text for this, not the course code
            instructor = repo.fuzzy_find_instructor(text)
            if instructor:
                return f"👨‍🏫 **{instructor['name']}**\nOffice: {instructor['office_location']}\nEmail: {instructor['email']}\nPhone: {instructor['phone']}"
            
            # If no person found, maybe they asked "Who teaches CS116?"
            # (Requires a new DB query we haven't built yet, so for now we just show course info or generic response)
            return f"I can find instructor offices (e.g., 'Where is Dr. Adam?'), but I don't have the semester schedule linked yet to tell you who is teaching **{target}** right now."

    # Case 2: Eligibility
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

    # Case 3: Semester Plan
    if intent_tag == 'make_schedule':
        ctx_mgr.clear_flow(user_id)
        track = ctx_mgr.extract_track(text)
        if track: ctx['track'] = track
        ctx_mgr.update_passed_courses(user_id, text)
        
        if ctx['track'] and ctx['passed_courses']:
            return advisor.generate_plan(ctx['track'], ctx['passed_courses'])
        if ctx['track']:
            ctx['status'] = 'waiting_for_courses'
            return f"Planning for **{ctx['track']}**. Which courses have you passed?"
        ctx['status'] = 'waiting_for_track'
        return "Sure! First, are you **Cybersecurity**, **Data Science**, or **General**?"

    # Case 4: Graduation Check
    if intent_tag == 'graduation_check':
        ctx_mgr.clear_flow(user_id)
        track = ctx_mgr.extract_track(text)
        if track: ctx['track'] = track
        ctx_mgr.update_passed_courses(user_id, text)
        if ctx['track'] and ctx['passed_courses']: return advisor.check_graduation(ctx['track'], ctx['passed_courses'])
        ctx['status'] = 'waiting_for_grad_info'
        return "To check graduation status, I need your **Track** and **Passed Courses**."

    # Standard Fallback to AI responses (Greetings, Jokes, etc.)
    resp = ai.get_response_for_tag(intent_tag)
    if resp: return resp
    
    return "I'm listening, but I'm not sure how to help with that specifically."

@app.route("/")
def index(): return render_template("index.html")

@app.route("/chat", methods=['POST'])
def chat():
    user_id = request.remote_addr
    data = request.json
    msg = data.get('message', '')
    if not msg: return jsonify({"response": "Empty message."})
    
    tag, conf = ai.predict_intent(msg)
    response = handle_intent(user_id, msg, tag, conf)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)