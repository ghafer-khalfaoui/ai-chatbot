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
    detected_course = repo.normalize_code(text)
    
    if detected_course and len(detected_course) > 3:
        ctx_mgr.set_last_entity(user_id, 'course', detected_course)
        
    last_course = ctx_mgr.get_last_entity(user_id)['value']


    # --- 🚦 FLOW CONTROL: The "Stitching" Logic ---
    
    # 1. CLARIFICATION LOOP (New for Day 3)
    # If we asked "Which course?", handle the answer here.
    if ctx['status'] == 'waiting_for_specific_course':
        if detected_course:
            # We found the missing piece! Restore the original intent.
            target = detected_course
            original_intent = ctx.get('pending_intent')
            
            # Reset flow
            ctx['status'] = 'idle'
            ctx['pending_intent'] = None
            
            # NOW, proceed as if the user said the full sentence originally
            # (We fall through to the Logic block below with the corrected target/intent)
            intent_tag = original_intent 
        else:
            return "I still didn't catch a course name. Which course are you asking about? (e.g., CS116)"

    # 2. Existing Flows (Eligibility, Graduation, Plan)
   # 2. Existing Flows (Eligibility, Graduation, Plan)
    if ctx['status'] == 'waiting_for_eligibility':
        target = ctx['target_course']
        passed = ctx_mgr.update_passed_courses(user_id, text)
        
        if passed:
            ctx['status'] = 'idle'
            result = advisor.check_eligibility(target, ctx['passed_courses'])
            
            # --- [FIX] ADD FOLLOW-UP LOGIC HERE TOO ---
            if result.startswith("❌"):
                import re
                # Robust Regex to find the missing course
                missing_match = re.search(r'(?:missing:|pass:)\s*(?:-|\n)?\s*([A-Z]{2,4}\d{3,5})', result, re.IGNORECASE | re.DOTALL)
                
                if missing_match:
                    missing_course = missing_match.group(1)
                    if missing_course != target:
                        ctx['status'] = 'waiting_for_prereq_confirmation'
                        ctx['pending_course_check'] = missing_course
                        return result + f"\n\n🤔 **Would you like to check the prerequisites for {missing_course}?** (Yes/No)"
            # ------------------------------------------
            
            return result
            
        return f"I'm checking eligibility for **{target}**. Please list the courses you have passed."

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


    # --- 🤖 LOGIC LAYER ---

    # Case 1: Course Info / Prereqs / Instructors
    # app.py (Case 1) - Removed 'check_eligibility'
    if intent_tag in ['ask_course_info', 'ask_prereqs', 'ask_instructor_info']:
        
        # Use Detected > Memory > None
        target = detected_course if detected_course else last_course
        
        # If we STILL don't have a course, trigger the Clarification Loop
        if not target:
            ctx['status'] = 'waiting_for_specific_course'
            ctx['pending_intent'] = intent_tag # Save what they wanted to do
            return "Which course are you asking about? (e.g., CS116)"
        
        # If we have the target, execute the logic
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
            # 1. Search for specific person ("Where is Adam?")
            instructor = repo.fuzzy_find_instructor(text)
            if instructor:
                return f"👨‍🏫 **{instructor['name']}**\nOffice: {instructor['office_location']}\nEmail: {instructor['email']}\nPhone: {instructor['phone']}"
            # 2. Fallback to generic course help
            return f"I can find instructors if you ask by name (e.g., 'Where is Dr. Adam?'). For **{target}**, check the schedule on the GJU website."

        if intent_tag == 'check_eligibility':
            if ctx['passed_courses']:
                return advisor.check_eligibility(target, ctx['passed_courses'])
            else:
                ctx['status'] = 'waiting_for_eligibility'
                ctx['target_course'] = target
                return f"Okay, let's check **{target}**. Please list your completed courses."
    # ... inside handle_intent ...

    # Case 2: Eligibility Check (Enhanced with Follow-up)
    if intent_tag == 'check_eligibility':
        if ctx['passed_courses']:
            result = advisor.check_eligibility(target, ctx['passed_courses'])
            
            # [NEW] Smart Follow-up Logic
            # If the result starts with "❌" (Not Eligible), we offer help.
            # [NEW] Smart Follow-up Logic
            if result.startswith("❌"):
                import re
                # OLD/STRICT: missing_match = re.search(r'- ([A-Z]{2,4}\d{3,5})', result)
                
                # NEW/ROBUST: Look for ANY course code that appears after "missing" or on a new line
                # This finds "CS223" whether it has a dash, a space, or just a newline before it.
                missing_match = re.search(r'(?:missing:|pass:)\s*(?:-|\n)?\s*([A-Z]{2,4}\d{3,5})', result, re.IGNORECASE | re.DOTALL)
                
                if missing_match:
                    missing_course = missing_match.group(1)
                    
                    # Verify this isn't the target course itself (just to be safe)
                    if missing_course != target:
                        ctx['status'] = 'waiting_for_prereq_confirmation'
                        ctx['pending_course_check'] = missing_course
                        return result + f"\n\n🤔 **Would you like to check the prerequisites for {missing_course}?** (Yes/No)"
            
            return result
        else:
            ctx['status'] = 'waiting_for_eligibility'
            ctx['target_course'] = target
            return f"Okay, let's check **{target}**. Please list your completed courses."

    # [NEW] Handle the "Yes/No" Answer
    if ctx['status'] == 'waiting_for_prereq_confirmation':
        if intent_tag == 'affirm':
            # User said "Yes" -> Check the missing course
            new_target = ctx.get('pending_course_check')
            ctx['status'] = 'idle' # Reset
            
            # Reuse the Prerequisite Logic
            prereqs = repo.get_prerequisites(new_target)
            if prereqs:
                p_list = ", ".join([f"{p[0]} ({p[1]})" for p in prereqs])
                return f"👍 **Good idea.** Here are the prerequisites for **{new_target}**:\n{p_list}"
            return f"**{new_target}** has no prerequisites. You should be able to take it!"
            
        elif intent_tag == 'deny':
            # User said "No"
            ctx['status'] = 'idle'
            return "Okay, let me know if you need help with anything else!"

    

    # Case 3: Semester Plan & Graduation (UNCHANGED)
    if intent_tag == 'make_schedule':
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

    if intent_tag == 'graduation_check':
        ctx_mgr.clear_flow(user_id)
        track = ctx_mgr.extract_track(text)
        if track: ctx['track'] = track
        ctx_mgr.update_passed_courses(user_id, text)
        if ctx['track'] and ctx['passed_courses']: return advisor.check_graduation(ctx['track'], ctx['passed_courses'])
        ctx['status'] = 'waiting_for_grad_info'
        return "To check graduation status, I need your **Track** and **Passed Courses**."

    # Fallback to AI
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