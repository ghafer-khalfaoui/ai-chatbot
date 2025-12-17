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
    
    # 1. Check Active Context Flow (Overrides BERT)
    if ctx['status'] == 'waiting_for_track':
        track = ctx_mgr.extract_track(text)
        if track:
            ctx['track'] = track
            if ctx['passed_courses']:
                ctx['status'] = 'idle'
                return advisor.generate_plan(ctx['track'], ctx['passed_courses'])
            else:
                ctx['status'] = 'waiting_for_courses'
                return f"Got it, **{track}**. Now list your completed courses (e.g., CS116, MATH101)."
        return "Please specify: General, Data Science, or Cybersecurity."

    if ctx['status'] == 'waiting_for_courses':
        added = ctx_mgr.update_passed_courses(user_id, text)
        if added:
            ctx['status'] = 'idle'
            return advisor.generate_plan(ctx['track'], ctx['passed_courses'])
        return "I didn't see any course codes. Please list them."

    if ctx['status'] == 'waiting_for_eligibility':
        target = ctx['target_course']
        added = ctx_mgr.update_passed_courses(user_id, text)
        if added:
            ctx['status'] = 'idle'
            return advisor.check_eligibility(target, ctx['passed_courses'])
        return f"To check eligibility for {target}, I need your completed courses."
        
    if ctx['status'] == 'waiting_for_grad_info':
        track = ctx_mgr.extract_track(text)
        if track: ctx['track'] = track
        ctx_mgr.update_passed_courses(user_id, text)
        
        if ctx['track'] and ctx['passed_courses']:
            ctx['status'] = 'idle'
            return advisor.check_graduation(ctx['track'], ctx['passed_courses'])
        elif ctx['track']: return f"Okay {track}, now list your passed courses."
        elif ctx['passed_courses']: return "Got the courses. Which track? (General, Data, Cyber)"
        return "I need your Track and Passed Courses."

    # 2. Handle BERT Intents
    if confidence < 0.35:
        # Fallback for low confidence
        if "plan" in text.lower(): intent_tag = 'request_advice'
        elif "eligib" in text.lower(): intent_tag = 'check_eligibility'
        elif "grad" in text.lower(): intent_tag = 'graduation_check'
        else: return "I'm not sure I understand. Try 'Plan my semester' or 'Check prereqs for CS116'."

    if intent_tag == 'ask_course_info':
        code = repo.normalize_code(text)
        if code:
            data = repo.get_course_details(code)
            if data:
                ctx_mgr.set_last_entity(user_id, 'course', code)
                return f"📘 **{data['course_code']}** - {data['course_name']}\n{data['description']}\nCredits: {data['credit_hours']}"
            return f"I couldn't find course {code}."
        return "Which course code are you asking about?"

    if intent_tag == 'ask_prerequisites':
        target = None
        if 'it' in text.lower():
            last = ctx_mgr.get_last_entity(user_id)
            if last['type'] == 'course': target = last['value']
        if not target: target = repo.normalize_code(text)
        
        if target:
            ctx_mgr.set_last_entity(user_id, 'course', target)
            prereqs = repo.get_prerequisites(target)
            if prereqs: return f"Prerequisites for **{target}**:\n" + "\n".join([f"- {p[0]} ({p[1]})" for p in prereqs])
            return f"**{target}** has no listed prerequisites."
        return "Please specify the course code."

    if intent_tag == 'ask_instructor_info':
        data = repo.fuzzy_find_instructor(text)
        if data:
            ctx_mgr.set_last_entity(user_id, 'instructor', data['name'])
            return f"👨‍🏫 **{data['name']}**\nOffice: {data['office_location']}\nEmail: {data['email']}\nPhone: {data['phone']}"
        return "I couldn't find that instructor."

    if intent_tag == 'request_advice':
        ctx_mgr.clear_flow(user_id)
        ctx['status'] = 'waiting_for_track'
        return "I can help plan your semester. First, which **Track** are you in? (General, Data Science, Cybersecurity)"

    if intent_tag == 'check_eligibility':
        ctx_mgr.clear_flow(user_id)
        target = repo.normalize_code(text)
        ctx_mgr.update_passed_courses(user_id, text)
        if target:
            if ctx['passed_courses']: return advisor.check_eligibility(target, ctx['passed_courses'])
            else:
                ctx['status'] = 'waiting_for_eligibility'
                ctx['target_course'] = target
                return f"Okay, let's check **{target}**. Please list your completed courses."
        return "Which course do you want to check eligibility for?"

    if intent_tag == 'graduation_check':
        ctx_mgr.clear_flow(user_id)
        track = ctx_mgr.extract_track(text)
        if track: ctx['track'] = track
        ctx_mgr.update_passed_courses(user_id, text)
        if ctx['track'] and ctx['passed_courses']: return advisor.check_graduation(ctx['track'], ctx['passed_courses'])
        ctx['status'] = 'waiting_for_grad_info'
        return "To check graduation status, I need your **Track** and **Passed Courses**."

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