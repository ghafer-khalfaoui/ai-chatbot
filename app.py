import sys
import os
import re
import json
import traceback
from flask import Flask, request, jsonify, render_template, session  # <-- ADDED session
from functools import lru_cache  # <-- NEW
import spacy  # <-- NEW
from spacy.pipeline import EntityRuler  # <-- NEW
import mysql.connector

# --- Flask App Initialization ---
app = Flask(__name__)
# --- [NEW] Secret key is required for Flask sessions ---
app.secret_key = "a-very-secret-key-for-gju-chatbot"

# --- Database Configuration ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'chatbot_db'
}

# --- Database Connection Function (Moved up) ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"--- DB CONNECTION ERROR ---: {err}")
        return None

# --- [NEW] Load the spaCy "brain" and teach it entities ---
def load_spacy_brain():
    print("Loading spaCy 'brain'...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except IOError:
        print("---------------------------------------------------------------------")
        print("FATAL: spaCy model 'en_core_web_sm' not found.")
        print("Please run this command in your terminal:")
        print("python -m spacy download en_core_web_sm")
        print("Then restart the server.")
        print("---------------------------------------------------------------------")
        sys.exit(1)

    print("Teaching 'brain' from database...")
    # Connect to the DB to get all courses and instructors
    conn = get_db_connection()
    if not conn:
        print("FATAL: Could not connect to DB to teach spaCy. Exiting.")
        sys.exit(1)
    
    cursor = conn.cursor(dictionary=True)
    patterns = [] # This is our "study guide" for the brain

    # 1. Teach it all Course Codes
    try:
        cursor.execute("SELECT course_code FROM courses")
        courses = cursor.fetchall()
        for course in courses:
            code = course['course_code']
            patterns.append({"label": "COURSE", "pattern": code})
            # Add a pattern for the code with a space (e.g., "CS 116")
            match = re.match(r'([A-Z]+)(\d+)', code)
            if match:
                patterns.append({"label": "COURSE", "pattern": f"{match.group(1)} {match.group(2)}"})

        print(f"Loaded {len(courses)} course patterns.")

        # 2. Teach it all Instructor Names
        cursor.execute("SELECT name FROM instructors")
        instructors = cursor.fetchall()
        for instructor in instructors:
            name = instructor['name']
            # Add a pattern for the full name (e.g., "Dr. Ismail Hababeh")
            patterns.append({"label": "INSTRUCTOR", "pattern": name})
            # Add patterns for partial names
            parts = name.split()
            if len(parts) > 1:
                if parts[0].lower() in ['dr.', 'prof.']:
                    # Add "Dr. Hababeh"
                    patterns.append({"label": "INSTRUCTOR", "pattern": f"{parts[0]} {parts[-1]}"})
                    # Add "Ismail Hababeh" (name without title)
                    patterns.append({"label": "INSTRUCTOR", "pattern": " ".join(parts[1:])})
                
                # Add just "Hababeh" (case-sensitive)
                patterns.append({"label": "INSTRUCTOR", "pattern": parts[-1]})


        print(f"Loaded {len(instructors)} instructor patterns.")
        
    except mysql.connector.Error as err:
        print(f"FATAL: DB Error while training spaCy: {err}")
    finally:
        cursor.close()
        conn.close()

    # 3. Add the "EntityRuler" to the spaCy pipeline
    # This adds our custom entities (COURSE, INSTRUCTOR) before spaCy's built-in ones
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns) # "ruler, study these patterns!"

    print("✅ spaCy 'brain' loaded and trained successfully.")
    return nlp

# --- Load the brain when the app starts ---
nlp = load_spacy_brain()

# --- Elective Course Definitions (KEPT) ---
ELECTIVES_GENERAL = {"CS333", "CS357", "CS358", "CS359", "CS364", "CS365", "CS371", "CS430", "CS432", "CS439", "CS450", "CS457", "CS458", "CS460", "CS462", "CS481", "CS482", "CS484", "CS489", "CS4512", "CS4811", "CS4831", "CS4832", "CS4833"}
ELECTIVES_DATA_SCIENCE = {"CS358", "CS359", "CS364", "CS371", "CS432", "CS450", "CS456", "CS457", "CS462", "CS484", "CS4512", "CS4811", "CS4813", "CS4831", "CS4832", "CS4833"}
ELECTIVES_CYBERSECURITY = {"CS354", "CS357", "CS359", "CS370", "CS372", "CS373", "CS374", "CS458", "CS4511", "CS4711", "CS4712", "CS4715", "CS4812", "CS4831", "CS4832", "CS4833"}
ALL_ELECTIVES = ELECTIVES_GENERAL.union(ELECTIVES_DATA_SCIENCE).union(ELECTIVES_CYBERSECURITY)
# ----------------------------------------------------

# --- [KEPT] Simple Dialogue State ---
# handle_instructor_query uses this, so we keep it.
last_mentioned_entity = {
    "type": None, 
    "value": None
}

# --- Save Conversation Function ---
def save_conversation(user_msg, bot_msg):
    max_len_db = 65535
    user_msg_t = (user_msg[:max_len_db-3]+'...') if len(user_msg)>max_len_db else user_msg
    bot_msg_t = (bot_msg[:max_len_db-3]+'...') if len(bot_msg)>max_len_db else bot_msg
    conn = get_db_connection()
    cursor = None
    if conn:
        try:
            cursor = conn.cursor()
            query = "INSERT INTO conversation_history (user_message, bot_response) VALUES (%s, %s)"
            cursor.execute(query, (user_msg_t, bot_msg_t))
            conn.commit()
        except mysql.connector.Error as err:
            print(f"Error saving conversation: {err}")
        finally:
            if cursor:
                cursor.close()
            if conn.is_connected():
                conn.close()

# --- Helper Functions (KEPT) ---
# [DELETED] extract_course_code()
# [DELETED] extract_instructor_name()

def extract_track_and_courses(text):
    norm = text.lower()
    track = None
    if "cybersecurity" in norm or "cyber security" in norm or "cyber" in norm:
        track = "Cybersecurity"
    elif "data science" in norm or "data" in norm:
        track = "Data Science"
    elif "general" in norm:
        track = "General"
    
    # [NEW] Use spaCy for course code extraction here too!
    doc = nlp(text)
    courses = set()
    for ent in doc.ents:
        if ent.label_ == "COURSE":
            courses.add(ent.text.replace(" ", ""))
            
    # [NEW] Fallback regex for codes spaCy might miss
    codes_raw = re.findall(r'\b([A-Z]{2,4})\s*(\d{3,5}|0099|0098|100)\b', text, re.IGNORECASE)
    for prefix, number in codes_raw:
        courses.add(f"{prefix.upper()}{number}")

    print(f"DEBUG extract_track_and_courses: Track='{track}', Courses={courses}")
    return track, courses

# --- Database Query Handlers (KEPT, but with Caching & Signature Updates) ---

# --- [UPDATED] Added Caching, Changed signature to (code) ---
@lru_cache(maxsize=128)
def handle_course_query(code):
    print(f"\n--- DEBUG handle_course_query ---")
    print(f"Code: '{code}'")
    if not code:
        return "Please tell me the full course code (e.g., CS116)."
    
    conn = get_db_connection()
    cursor = None
    course = None
    if not conn:
        return "DB connection error."
    
    try:
        cursor = conn.cursor(dictionary=True)
        q = "SELECT course_code, course_name, credit_hours, description FROM courses WHERE course_code = %s"
        print(f"Query: ... WHERE course_code = '{code}'")
        cursor.execute(q, (code,))
        course = cursor.fetchone()
        print(f"Result: {course}")
        if course:
            print(f"Found: {course['course_code']}")
            resp = (f"**{course['course_code']} - {course['course_name']}**:\nCredits: {course.get('credit_hours','N/A')}\nDescription: {course.get('description','N/A')}")
            global last_mentioned_entity
            last_mentioned_entity = {"type": "course", "value": course['course_code']}
            print(f"DEBUG State: Updated last entity: {last_mentioned_entity}")
            return resp
        else:
            print(f"Not found.")
            return f"Sorry, couldn't find course '{code}'."
    except mysql.connector.Error as err:
        print(f"DB error: {err}")
        return f"DB error: {err}"
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("DB closed.")

# --- [UPDATED] Added Caching, Changed signature to (code) ---
@lru_cache(maxsize=128)
def handle_prereq_query(code):
    print(f"\n--- DEBUG handle_prereq_query ---")
    print(f"Code: '{code}'")
    if not code:
        return "What course code for prerequisites?"
    
    conn = get_db_connection()
    cursor = None
    check = None
    prereqs = None
    if not conn:
        return "DB connection error."
    
    try:
        cursor = conn.cursor()
        q = "SELECT p.prerequisite_code, c.course_name FROM prerequisites p JOIN courses c ON p.prerequisite_code = c.course_code WHERE p.course_code = %s"
        print(f"Query: ... WHERE p.course_code = '{code}'")
        cursor.execute(q, (code,))
        prereqs = cursor.fetchall()
        print(f"Result: {prereqs}")
        
        if prereqs:
            print(f"Prereqs found.")
            resp = f"Prerequisites for **{code}**:\n" + "\n".join([f"- {p_code} ({p_name})" for p_code, p_name in prereqs])
            global last_mentioned_entity
            last_mentioned_entity = {"type": "course", "value": code}
            print(f"DEBUG State: Updated last entity: {last_mentioned_entity}")
            return resp
        else:
            print(f"No prereqs found.")
            check = conn.cursor()
            check.execute("SELECT 1 FROM courses WHERE course_code = %s", (code,))
            exists = check.fetchone()
            if exists:
                return f"**{code}** has no prerequisites listed!"
            else:
                return f"Sorry, couldn't find course '{code}'."
    except mysql.connector.Error as err:
        print(f"DB error: {err}")
        return f"DB error: {err}"
    finally:
        if check:
            check.close()
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("DB closed.")

# --- [UPDATED] Added Caching, Changed signature to (name_ext) ---
@lru_cache(maxsize=128)
def handle_instructor_query(name_ext):
    print(f"\n--- DEBUG handle_instructor_query (v10) ---")
    print(f"Name received: '{name_ext}'")
    if not name_ext:
        return "Who are you looking for?"
    
    conn = get_db_connection()
    cursor = None
    instr = None
    if not conn:
        return "DB connection error."
    
    try:
        cursor = conn.cursor(dictionary=True)
        # Try finding the exact name spaCy gave us
        print(f"Exact match query: '{name_ext}'")
        cursor.execute("SELECT * FROM instructors WHERE name = %s", (name_ext,))
        instr = cursor.fetchone()
        print(f"Result EXACT: {instr}")
        
        if not instr:
            # Fallback to LIKE search for partial names
            print(f"LIKE search...")
            parts = name_ext.replace('Dr.','').replace('Prof.','').strip().split()
            term = parts[-1] if parts else name_ext
            pattern = f"%{term}%"
            print(f"LIKE pattern: '{pattern}'")
            cursor.execute("SELECT * FROM instructors WHERE name LIKE %s ORDER BY name LIMIT 1", (pattern,))
            instr = cursor.fetchone()
            print(f"Result LIKE: {instr}")
            
        if instr:
            print(f"Found: {instr['name']}")
            name = instr.get('name','N/A')
            title = instr.get('title','N/A')
            office = instr.get('office_location','N/A')
            email = instr.get('email','N/A')
            phone = instr.get('phone','N/A')
            status = instr.get('status')
            resp = f"**{name}** ({title}):\n🏢 Office: {office}\n📧 Email: {email}\n📞 Phone: {phone}"
            if status:
                resp += f"\nℹ️ Status: {status}"
            global last_mentioned_entity
            last_mentioned_entity = {"type": "instructor", "value": instr['name']}
            print(f"DEBUG State: Updated last entity: {last_mentioned_entity}")
            return resp
        else:
            print(f"Not found.")
            return f"Sorry, couldn't find instructor '{name_ext}'."
    except mysql.connector.Error as err:
        print(f"DB error: {err}")
        return f"DB search error."
    except Exception as e:
        print(f"Format error: {e}")
        traceback.print_exc()
        return "Error formatting."
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("DB closed.")

# --- [KEPT] All other handlers are unchanged ---
def handle_advisor_request(track, user_completed_codes):
    # This entire function is kept as-is. It's a perfect "tool".
    print(f"\n--- DEBUG Advisor v10.1 ---")
    print(f"Track: {track}, Completed: {user_completed_codes}")
    assumed_remedials = {'ARB0099','ENGL0098','ENGL0099','MATH0099'}
    actual_comp = user_completed_codes.union(assumed_remedials)
    if not track or not user_completed_codes:
        return "Missing info for advice."
    
    conn = get_db_connection()
    cursor = None
    if not conn:
        return "DB connection error."
    
    try:
        cursor = conn.cursor(dictionary=True)
        # 1. Fetch Data
        cursor.execute("SELECT course_code, course_name, credit_hours FROM courses")
        all_courses_dict = {r['course_code']:r for r in cursor.fetchall()}
        cursor.execute("SELECT course_code, prerequisite_code FROM prerequisites")
        prereqs_raw = cursor.fetchall()
        prereqs_dict = {}
        for r in prereqs_raw:
            c, p = r['course_code'], r['prerequisite_code']
            if c not in prereqs_dict:
                prereqs_dict[c] = set()
            if p not in assumed_remedials:
                prereqs_dict[c].add(p)

        # 2. Define Track & Elective Sets
        track_comp_spec = {"General": {"CS330","CS332","CS419","CS477"},"Data Science": {"CS330","CE377","CS460","EE570"},"Cybersecurity": {"CE354","CS470","CS4713","CS4714"}}
        common_comp = {"CS201","CE201","CS222","CS223","CS263","CS264","CS323","CS342","CE352","CS355","CS356","CE357","CE3570","CS416","CS451","MATH101","MATH102","CS116","CS1160","CS117","CS1170","CE212","CE2120","EE317","IE0121","CS391","CS491","CS492"}
        all_comp = common_comp.union(track_comp_spec.get(track, set()))
        
        track_electives = set()
        if track == "General":
            track_electives = ELECTIVES_GENERAL
        elif track == "Data Science":
            track_electives = ELECTIVES_DATA_SCIENCE
        elif track == "Cybersecurity":
            track_electives = ELECTIVES_CYBERSECURITY
        print(f"Compulsory: {len(all_comp)}")

        # 3. Calculate Hours
        comp_hrs = sum(all_courses_dict[c]['credit_hours'] for c in user_completed_codes if c in all_courses_dict and c not in assumed_remedials)
        total_req = 145
        remain_hrs = max(0, total_req - comp_hrs)

        # 4. Determine Eligible Courses
        elig_courses = []
        exclude = set(assumed_remedials)
        SENIOR_THR = 90
        senior = {'CS391','CS491','CS492'}
        missing_prereqs = {}
        if comp_hrs < SENIOR_THR:
            exclude.update(senior)
        for code, info in all_courses_dict.items():
            lab_done = ('0' == code[-1] and code[:-1] in actual_comp)
            if code in actual_comp or code in exclude or lab_done:
                continue
            cursor.execute("SELECT prerequisite_code FROM prerequisites WHERE course_code = %s", (code,))
            req_raw = {r['prerequisite_code'] for r in cursor.fetchall()}
            missing = req_raw - actual_comp
            if not missing:
                elig_courses.append(info)
            else:
                core_miss = missing - assumed_remedials
                if not core_miss:
                    elig_courses.append(info)
                else:
                    missing_prereqs[code] = list(core_miss)
        print(f"Eligible: {len(elig_courses)}")

        # 5. Prioritize and Select Courses
        sched = []
        curr_hrs = 0
        TARGET = 18

        def get_pri_v11(c):
            code = c['course_code']
            is_compulsory = code in all_comp
            is_elective = code in ALL_ELECTIVES
            level_match = re.match(r'([A-Z]+)(\d)', code)
            
            if level_match:
                lvl_num = int(level_match.group(2)) # Group 2 is the number
                pre_str = level_match.group(1) # Group 1 is the prefix
            else:
                lvl_num = 9 # Default level
                pre_str = '' # Default prefix
            
            p = 100 
            if pre_str in ['CS', 'CE', 'MATH', 'IE', 'EE']:
                if code.endswith('0') and code[:-1] in all_courses_dict: # Lab check
                    p = lvl_num * 10 + 5 
                else: # Theory
                    p = lvl_num * 10
            elif is_elective:
                p = 60 + lvl_num 
            elif pre_str in ['ENGL', 'ARB', 'GERL', 'MILS', 'NE101']:
                p = 80 + lvl_num 

            if is_compulsory:
                p -= 50 
            
            if code in track_electives:
                p -= 2

            return p
        
        elig_courses.sort(key=get_pri_v11) 
        print(f"Top 5 Eligible Priority v11: {[c['course_code'] for c in elig_courses[:5]]}")
        
        elig_comp = [c for c in elig_courses if c['course_code'] in all_comp]
        elig_elec_etc = [c for c in elig_courses if c['course_code'] not in all_comp]
        
        schedule_codes = set() 

        def add_courses(course_list, current_h, max_h, outer_schedule_codes, outer_eligible_courses, outer_all_courses_dict, outer_actual_comp):
            sched_part = []
            codes_add = set()
            temp_elig = list(course_list)
            
            while current_h < max_h and temp_elig:
                added_pass = False
                next_pass = []
                for course in temp_elig:
                    code = course['course_code']
                    hrs = course['credit_hours']
                    
                    if hrs <= 0 or code in outer_schedule_codes or code in codes_add:
                        continue
                    
                    if current_h + hrs <= max_h:
                         is_lab = code.endswith('0') and code[:-1] in outer_all_courses_dict
                         theory = code[:-1] if is_lab else None
                         can_add = True
                         theory_add = None
                         
                         if is_lab and theory not in outer_actual_comp and theory not in outer_schedule_codes.union(codes_add):
                              info = outer_all_courses_dict.get(theory)
                              elig = any(ec['course_code'] == theory for ec in outer_eligible_courses)
                              theory_hrs = info.get('credit_hours', 0) if info else 0
                              if elig and info and (current_h + hrs + theory_hrs <= max_h):
                                  theory_add = info
                              else:
                                  can_add = False
                         
                         if can_add:
                              if theory_add and theory_add.get('course_code') and theory_add['course_code'] not in outer_schedule_codes.union(codes_add):
                                  theory_hrs_add = theory_add.get('credit_hours', 0)
                                  if theory_hrs_add > 0:
                                      sched_part.append(theory_add)
                                      current_h += theory_hrs_add
                                      codes_add.add(theory_add['course_code'])
                                      print(f"DEBUG v11 Added co-req {theory_add['course_code']}")
                                  else:
                                       print(f"WARN: Invalid hours for theory {theory_add.get('course_code')}")
                              
                              sched_part.append(course)
                              current_h += hrs
                              codes_add.add(code)
                              added_pass = True
                              print(f"DEBUG v11 Added {code} - Total: {current_h} Cr")
                         else:
                              next_pass.append(course)
                    else:
                         next_pass.append(course) 
                
                temp_elig = [c for c in next_pass if c['course_code'] not in codes_add]
                if not added_pass:
                    break
            return sched_part, current_h, codes_add
        
        comp_part, curr_hrs, codes_comp = add_courses(elig_comp, curr_hrs, TARGET, schedule_codes, elig_courses, all_courses_dict, actual_comp)
        sched.extend(comp_part)
        schedule_codes.update(codes_comp)
        
        if curr_hrs < TARGET:
             print(f"Filling remaining {TARGET - curr_hrs} hrs.")
             elec_part, curr_hrs, codes_elec = add_courses(elig_elec_etc, curr_hrs, TARGET, schedule_codes, elig_courses, all_courses_dict, actual_comp)
             sched.extend(elec_part)
             schedule_codes.update(codes_elec)

        # 6. Format the Response
        if sched:
            print(f"DEBUG v9.3: Final schedule before sort: {sched}")
            try:
                sched.sort(key=lambda x: x.get('course_code', ''))
                print(f"DEBUG v9.3: Schedule sorted successfully.")
            except Exception as e_sort:
                print(f"ERROR during schedule sort: {e_sort}")
                traceback.print_exc()
                return "Error sorting the suggested schedule."
                
            resp = (f"Okay, **{track}** ({comp_hrs} hrs), suggested (**{curr_hrs} Cr** / aim {TARGET}):\n\n")
            comp_in_details = []
            elec_in_details = []
            other_in_details = []

            print(f"DEBUG v9.3: Starting formatting loop for {len(sched)} courses...")
            for idx, course_dict in enumerate(sched):
                print(f"DEBUG v9.3: Formatting course #{idx+1}: {course_dict}")
                try:
                    code = course_dict.get('course_code', 'ERR_CODE')
                    name = course_dict.get('course_name', 'Unknown Name')
                    hours = course_dict.get('credit_hours', '?')
                    detail_str = f"- **{code}**: {name} ({hours} Cr)"
                    if code in all_comp:
                        highlight = "(Track Req)" if code in track_comp_spec.get(track, set()) else ""
                        comp_in_details.append(f"{detail_str} {highlight}".strip())
                    elif code in ALL_ELECTIVES:
                        elec_in_details.append(detail_str)
                    else:
                        other_in_details.append(detail_str)
                except Exception as e_format_loop:
                    print(f"ERROR inside formatting loop for item {idx+1} ({course_dict}): {e_format_loop}")
                    traceback.print_exc()
            
            print(f"DEBUG v9.3: Formatting loop finished.")

            if comp_in_details:
                resp += "**Compulsory Courses:**\n" + "\n".join(comp_in_details) + "\n"
            if elec_in_details:
                resp += "\n**Elective/Other Courses:**\n" + "\n".join(elec_in_details) + "\n"
                ELEC_THR = 90
                if comp_hrs >= ELEC_THR:
                    resp += "_Note: Electives typically taken in German year._\n"
            if other_in_details:
                resp += "\n**Other Requirements:**\n" + "\n".join(other_in_details) + "\n"

            if abs(curr_hrs - TARGET) > 1:
                resp += f"\n**Note:** Schedule totals {curr_hrs} hours. Reaching {TARGET} may require other choices or completing prereqs."
                important_skipped = [c for c in ['CS201','CE212','CS263','CS222','CS223','CS342'] if c in missing_prereqs and c not in actual_comp]
                if important_skipped:
                    resp += " Key upcoming courses needing prerequisites include:\n"
                    for skipped_code in important_skipped[:3]:
                        missing_list = ", ".join(missing_prereqs.get(skipped_code, ['Unknown']))
                        resp += f"- {skipped_code} (Requires: {missing_list})\n"

            resp += f"\n\nYou have approximately **{remain_hrs}** credit hours remaining to graduate."
            resp += "\n\n*Disclaimer: This is an automated suggestion..."
        else:
            resp = "Couldn't generate schedule. Check prereqs."
        
        print("DEBUG v9.3: Returning final response.")
        return resp

    except mysql.connector.Error as err:
        print(f"DB error: {err}")
        return "DB error."
    except Exception as e:
        print(f"Advisor Error: {e}")
        traceback.print_exc()
        return "Unexpected error."
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception as e_cursor:
                 print(f"Error closing cursor: {e_cursor}")
        if conn and conn.is_connected():
            try:
                conn.close()
                print("DB closed.")
            except Exception as e_conn:
                 print(f"Error closing connection: {e_conn}")

# --- [KEPT] This handler's signature (message) is correct ---
def handle_eligibility_check(user_message):
    print(f"\n--- DEBUG handle_eligibility_check ---")
    print(f"Received: {user_message}")
    
    # We still need a regex extractor for the *target* course
    target_code = None
    match = re.search(r'\b(cs|ce|ee|ie|math|engl|arb)\s*(\d{3,5}|0099|0098|100)\b', user_message, re.IGNORECASE)
    if match:
        target_code = f"{match.group(1).upper()}{match.group(2)}"
    print(f"Target: {target_code}")
    
    if not target_code:
        return "Which course? Also list completed courses."
    
    track, user_comp = extract_track_and_courses(user_message)
    user_comp.discard(target_code) # Remove the target from the completed list
    print(f"Completed: {user_comp}")
    
    if not user_comp:
        return f"Okay, checking **{target_code}**. List completed courses."
    
    assumed_rem = {'ARB0099','ENGL0098','ENGL0099','MATH0099'}
    actual_comp = user_comp.union(assumed_rem)
    conn = get_db_connection()
    cursor = None
    if not conn:
        return "DB error."
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT course_name FROM courses WHERE course_code = %s", (target_code,))
        target_info = cursor.fetchone()
        if not target_info:
            return f"Course '{target_code}' not found."
        
        cursor.execute("SELECT prerequisite_code FROM prerequisites WHERE course_code = %s", (target_code,))
        req_raw = {r['prerequisite_code'] for r in cursor.fetchall()}
        print(f"Raw prereqs: {req_raw}")
        missing = req_raw - actual_comp
        print(f"Missing: {missing}")
        core_miss = missing - assumed_rem
        print(f"Core missing: {core_miss}")
        
        if not core_miss:
             resp = f"✅ Yes, eligible for **{target_code}** ({target_info['course_name']})."
             if not req_raw:
                 resp += "\n (No prereqs listed)."
             elif missing and not core_miss:
                 resp += "\n (Assumes remedials met)."
        else:
             resp = f"❌ No, not eligible for **{target_code}**. Need:\n"
             missing_names = []
             if core_miss:
                 place = ', '.join(['%s'] * len(core_miss))
                 q_names = f"SELECT course_code, course_name FROM courses WHERE course_code IN ({place})"
                 try:
                     cursor.execute(q_names, tuple(core_miss))
                     miss_info = cursor.fetchall()
                 except mysql.connector.Error as name_err:
                     print(f"Error fetching names for missing prereqs: {name_err}")
                     miss_info = [] # Fallback
                 missing_names = [f"{c['course_code']} ({c.get('course_name', 'Unknown Name')})" for c in miss_info]
                 found = {c['course_code'] for c in miss_info}
                 not_found = core_miss - found
                 if not_found:
                     missing_names.extend(list(not_found))
             resp += "   - " + "\n   - ".join(sorted(missing_names))
        
        resp += "\n\n*Disclaimer: Verify with official system.*"
        return resp
    
    except mysql.connector.Error as err:
        print(f"DB error: {err}")
        return "DB error."
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return "Unexpected error."
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("DB closed.")

# --- [KEPT] This handler's signature (message) is correct ---
def handle_graduation_check(user_message):
    print(f"\n--- DEBUG handle_graduation_check ---")
    print(f"Received user message: {user_message}")
    track, user_completed_codes = extract_track_and_courses(user_message)
    assumed_remedials = {'ARB0099', 'ENGL0098', 'ENGL0099', 'MATH0099'}
    if not track:
        return "Need your **track** and completed courses."
    if not user_completed_codes:
        return f"Okay, {track} track. Now list **non-remedial courses** completed."
    
    print(f"Grad Check: Track={track}, Completed={user_completed_codes}")
    conn = get_db_connection()
    cursor = None
    if not conn:
        return "DB error."
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT course_code, course_name, credit_hours FROM courses")
        all_courses_dict = {row['course_code']: row for row in cursor.fetchall()}
        if not all_courses_dict:
            raise ValueError("No courses found.")
        
        track_comp_specific = {"General": {"CS330","CS332","CS419","CS477"},"Data Science": {"CS330","CE377","CS460","EE570"},"Cybersecurity": {"CE354","CS470","CS4713","CS4714"}}
        common_compulsory_core = {"CS201","CE201","CS222","CS223","CS263","CS264","CS323","CS342","CE352","CS355","CS356","CE357","CE3570","CS416","CS451","MATH101","MATH102","CS116","CS1160","CS117","CS1170","CE212","CE2120","EE317","IE0121","CS391","CS491","CS492","ARB100","ENGL1001","ENGL1002","GERL101B1","GERL102B1","GERL201B1","GERL202B1","GERL301B1","GERL302B1","MILS100","NE101"}
        all_compulsory_for_track = common_compulsory_core.union(track_comp_specific.get(track, set()))
        print(f"DEBUG Grad Check: Total compulsory: {len(all_compulsory_for_track)}")
        
        completed_hours = sum(all_courses_dict[code]['credit_hours'] for code in user_completed_codes if code in all_courses_dict and code not in assumed_remedials)
        total_required_hours = 145
        remaining_hours = max(0, total_required_hours - completed_hours)
        remaining_compulsory = set()
        
        for code in all_compulsory_for_track:
             if code not in user_completed_codes:
                  if code in all_courses_dict:
                      remaining_compulsory.add(code)
                  else:
                      print(f"WARN Grad Check: Compulsory {code} not in DB.")
        print(f"DEBUG Grad Check: Remaining compulsory ({len(remaining_compulsory)}): {remaining_compulsory}")
        
        completed_elective_hours = sum(all_courses_dict[code]['credit_hours'] for code in user_completed_codes if code not in all_compulsory_for_track and code in ALL_ELECTIVES and code in all_courses_dict)
        required_elective_hours = 12
        remaining_elective_hours = max(0, required_elective_hours - completed_elective_hours)
        print(f"DEBUG Grad Check: Elective hours: completed={completed_elective_hours}, remaining={remaining_elective_hours}")
        
        response = f"Graduation Summary for **{track}**:\n\n📊 **Credit Hours:**\n - Completed: **{completed_hours}** / {total_required_hours}\n - Remaining: **{remaining_hours}**\n - Elective Hrs Remaining: **{remaining_elective_hours}** / {required_elective_hours} (German year)\n\n"
        if remaining_compulsory:
            response += f"📚 **Remaining Compulsory Courses ({len(remaining_compulsory)}):**\n"
            remaining_details = []
            for code in sorted(list(remaining_compulsory)):
                name = all_courses_dict[code].get('course_name','?')
                hours = all_courses_dict[code].get('credit_hours','?')
                remaining_details.append(f"{code} ({name} - {hours} Cr)")
            display_limit = 15
            response += "- " + "\n- ".join(remaining_details[:display_limit])
            if len(remaining_details) > display_limit:
                response += f"\n- ...and {len(remaining_details) - display_limit} more."
        else:
            response += "✅ All **compulsory** courses seem complete!"
        
        response += "\n\n*Disclaimer: Estimate only. Verify with Registration Dept.*"
        return response
    except mysql.connector.Error as err:
        print(f"DB error: {err}")
        return "DB error."
    except Exception as e:
        print(f"Grad Check Error: {e}")
        traceback.print_exc()
        return "Unexpected error."
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("DB closed.")

# ======================================================================
# --- [NEW] MAIN CHATBOT BRAIN (v17 - spaCy EntityRuler) ---
# ======================================================================
def get_bot_response(user_message):
    
    # 1. Process the user's message with the "brain"
    doc = nlp(user_message) # We use the original casing now
    
    # 2. Extract keywords (Intents) and entities
    intents = set([token.lemma_ for token in doc if not token.is_stop])
    
    # --- [NEW] Smart Entity Extraction ---
    # We find entities found by our new "brain"
    course_code = None
    instructor_name = None
    
    for ent in doc.ents:
        if ent.label_ == "COURSE":
            course_code = ent.text.replace(" ", "") # Combine "CS 116" -> "CS116"
        elif ent.label_ == "INSTRUCTOR":
            instructor_name = ent.text

    # --- Fallback for course codes spaCy might miss ---
    if not course_code:
        match = re.search(r'\b(cs|ce|ee|ie|math|engl|arb)\s*(\d{3,5}|0099|0098|100)\b', user_message, re.IGNORECASE)
        if match:
             course_code = f"{match.group(1).upper()}{match.group(2)}"

    print(f"DEBUG: Intents={intents}")
    print(f"DEBUG: Course={course_code}, Instructor={instructor_name}")
    print(f"DEBUG: Session state={session.get('advisor_status')}")

    # --- 3. The New "Router" Logic (State Machine) ---
    try:
        # Check if the user is already in the middle of a conversation
        if session.get('advisor_status') == 'WAITING_FOR_TRACK':
            track, courses = extract_track_and_courses(user_message)
            if track:
                session['advisor_track'] = track
                session['advisor_status'] = 'WAITING_FOR_COURSES'
                return "Got it. And what courses have you completed?"
            else:
                return "I'm sorry, I didn't recognize that track. Please tell me your track (General, Data Science, or Cybersecurity)."

        if session.get('advisor_status') == 'WAITING_FOR_COURSES':
            track = session.get('advisor_track', 'General') 
            extracted_track, courses = extract_track_and_courses(user_message)
            if courses:
                session.pop('advisor_status', None) # Clear the state
                session.pop('advisor_track', None)
                return handle_advisor_request(track, courses)
            else:
                return "I didn't find any course codes in that message. Please list the courses you've completed (e.g., CS116, MATH101)."

        # Handle eligibility follow-up
        if session.get('advisor_status') == 'WAITING_FOR_ELIGIBILITY':
            track, courses = extract_track_and_courses(user_message)
            if courses:
                target_course = session.get('target_course', 'that course')
                session.pop('advisor_status', None) # Clear state
                session.pop('target_course', None)
                # Pass the *original string* to the handler
                return handle_eligibility_check(f"check {target_course} with {user_message}")
            else:
                return "I'm sorry, I still need the list of courses you've completed to check eligibility."
        
        if session.get('advisor_status') == 'WAITING_FOR_GRAD_CHECK':
            track, courses = extract_track_and_courses(user_message)
            if track and courses:
                session.pop('advisor_status', None) # Clear state
                return handle_graduation_check(f"track {track} courses {user_message}")
            elif track:
                session['advisor_track'] = track
                return "Got it. Now what courses have you completed?"
            elif courses:
                # This state isn't perfect, but it's a start
                return "Got it. And what is your track?"
            else:
                return "I need both your track and your completed courses to check your graduation status."

    except Exception as e:
        print(f"--- ERROR IN STATE MACHINE ---: {e}")
        traceback.print_exc()
        session.clear() # Clear the broken state completely
        return "Sorry, I got a bit confused. Let's start over. What would you like to ask?"


    # --- [NEW] Keyword Lists from your last message ---
    prereq_keywords = {

        'prerequisite', 'require', 'prereq', 'before', 'need', 'needed', 'must',
        'requirement', 'required', 'dependency', 'depend', 'condition', 'prior',
        'previous', 'taken', 'take before', 'mandatory', 'necessary', 'eligibility',
        'corequisite', 'co-req', 'pre-req', 'foundation', 'introductory', 'basic',
        'needed course', 'entry requirement', 'background', 'starting point',
         'prepare', 'must complete', 'previous course', 'needed subject'
    }

    course_keywords = {
        'course', 'info', 'information', 'detail', 'describe', 'about', 'subject',
        'module', 'unit', 'topic', 'class', 'syllabus', 'overview', 'outline',
        'curriculum', 'lecture', 'lesson', 'content', 'objective', 'goal',
        'learning outcome', 'credit', 'duration', 'schedule', 'hours',
        'code', 'course code', 'summary', 'introduction', 'what is', 'explain',
        'explanation', 'coverage', 'scope', 'description', 'academic info',
        'structure', 'component', 'material', 'area', 'focus', 'section', 'field'
    }

    instructor_keywords = {
        'who', 'dr', 'prof', 'doctor', 'professor', 'instructor', 'lecturer',
        'teacher', 'supervisor', 'advisor', 'mentor', 'staff', 'faculty',
        'email', 'phone', 'contact', 'office', 'location', 'room', 'address',
        'office hours', 'time', 'available', 'availability', 'how to reach',
        'who teaches', 'who handles', 'who is teaching', 'contact info',
        'reach out', 'supervise', 'guidance', 'meet', 'consult', 'consultation',
        'communication', 'appointment', 'message', 'ask', 'email id', 'faculty name'
    }

    advice_keywords = {
        'advice', 'advise', 'next', 'should', 'take', 'register', 'plan', 'suggest',
        'recommend', 'recommendation', 'choose', 'selection', 'what to take',
        'guidance', 'path', 'pathway', 'route', 'follow', 'future', 'upcoming',
        'better', 'best', 'help me decide', 'pick', 'course plan', 'academic plan',
        'registration', 'next semester', 'schedule', 'study plan', 'what should I do',
        'what’s next', 'next step', 'which one', 'career path', 'course choice',
        'options', 'choice', 'registering', 'decision', 'planning', 'advising',
        'academic advice', 'what’s recommended', 'which to choose'
    }

    eligibility_keywords = {
        'eligible', 'can', 'allowed', 'permission', 'able', 'qualified', 'access',
        'available', 'approve', 'approval', 'accepted', 'acceptance', 'meet criteria',
        'requirement', 'fulfill', 'fit', 'capable', 'permitted', 'authorized',
        'may', 'possible', 'eligibility', 'approval needed', 'who can', 'who may',
        'right to', 'entitled', 'can apply', 'can register', 'qualification',
        'criteria', 'limit', 'restriction', 'condition', 'restriction apply',
        'approval process', 'check if eligible', 'eligibility check', 'apply for',
        'whether can', 'can enroll'
    }

    grad_keywords = {
        'graduate', 'graduation', 'left', 'remaining', 'finish', 'completed',
        'done', 'degree', 'diploma', 'final', 'semester', 'last', 'end', 'complete',
        'credits left', 'credits remaining', 'requirements', 'capstone', 'project',
        'thesis', 'graduating', 'nearly done', 'close to finish', 'countdown',
        'progress', 'how many left', 'how far', 'ready', 'ready to graduate',
        'finish line', 'nearly finished', 'final step', 'complete degree',
        'almost done', 'completion', 'remaining courses', 'leftover', 'remaining subjects',
        'about to graduate', 'graduation checklist', 'final year', 'last year', 'almost finish'
    }

    # --- 4. Main Router (No state active) ---
    
    # Check for complex queries first
    if advice_keywords.intersection(intents):
        session['advisor_status'] = 'WAITING_FOR_TRACK' 
        return "Absolutely! I can help plan your next semester. To start, what is your **Track**? (General, Data Science, or Cybersecurity)"
    
    if eligibility_keywords.intersection(intents) and course_code:
        session['advisor_status'] = 'WAITING_FOR_ELIGIBILITY'
        session['target_course'] = course_code
        return f"Okay, checking eligibility for **{course_code}**. What non-remedial courses have you completed?"

    if grad_keywords.intersection(intents):
        session['advisor_status'] = 'WAITING_FOR_GRAD_CHECK'
        return "I can help with that. What is your **Track** and what **courses** have you completed?"

    # --- Simple, direct queries ---
    # We now pass the specific ENTITY to the handler, not the whole message.
    
    if prereq_keywords.intersection(intents) and course_code:
        return handle_prereq_query(course_code)

    if instructor_keywords.intersection(intents) and instructor_name:
        return handle_instructor_query(instructor_name)

    if course_keywords.intersection(intents) and course_code:
        return handle_course_query(course_code)
    
    # --- Handle Fallbacks (If we only get partial info) ---
    if course_code: # User just typed "CS116"
        return handle_course_query(course_code)
    
    if instructor_name: # User just typed "Amani" or "Dr. Hababeh"
        return handle_instructor_query(instructor_name)
        
    # --- Handle Chit-Chat ---
    msg_lower_lemma = set([token.lemma_ for token in nlp(user_message.lower())])
    if {'hi', 'hello', 'hey'}.intersection(msg_lower_lemma):
        return "Hello! How can I help you today? 🎓"
    if {'bye', 'goodbye'}.intersection(msg_lower_lemma):
        return "Goodbye! 👋"
    if {'thanks', 'thank', 'shukran'}.intersection(msg_lower_lemma):
        return "You're welcome! 😊"
    if {'who', 'be', 'you'}.issubset(msg_lower_lemma):
        return "I’m the GJU Student Assistant 🤖 — an AI helper built to make your academic life easier. I can answer questions about courses, instructors, and your study plan."
    if {'help'}.intersection(msg_lower_lemma):
         return "I'm happy to help! You can ask me about: \n* **Course Info** (e.g., 'tell me about CS222')\n* **Prerequisites** (e.g., 'what are the prereqs for CS342?')\n* **Instructor Info** (e.g., 'office hours for Dr. Ismail Hababeh')\n* **Academic Advice** (e.g., 'what courses should I take next?')"

    # --- Final Fallback ---
    print(f"--- FALLBACK: No rule matched. ---")
    return "I'm sorry, I'm not sure how to help with that. You can ask me about courses, prerequisites, instructors, or academic advice."


# --- Main API Route (KEPT) ---
@app.route("/chat", methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        bot_response = get_bot_response(user_message)
        save_conversation(user_message, bot_response)
        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"--- FATAL ERROR IN CHAT ENDPOINT ---")
        traceback.print_exc()
        save_conversation(user_message, "Error occurred")
        return jsonify({"response": "Sorry, a critical error occurred. Please try again later."}), 500

# --- Homepage (KEPT) ---
@app.route("/")
def index():
    return render_template("index.html")

# --- Run the App (KEPT) ---
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)