import mysql.connector # type: ignore
import time
import re
import difflib
import logging
import os
import json
import pickle
from mysql.connector import pooling
import numpy as np
import torch # <--- CHANGED: PyTorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LogicLayer")

# Database Configuration
dbconfig = {
    "database": os.getenv("DB_NAME", "chatbot_db"),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", ""),
    "host":     os.getenv("DB_HOST", "localhost")
}

try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="gju_pool",
        pool_size=5,
        pool_reset_session=True,
        **dbconfig
    )
    logger.info("✅ Database Connection Pool Created")
except Exception as e:
    logger.error(f"❌ Could not create DB Pool: {e}")
    connection_pool = None


# --- 1. AI/Model Layer (Back-up Class) ---
class AIModel:
    def __init__(self, model_dir="bert_intent_model"):
        # We generally use the one in app.py, but this is good for testing logic.py alone
        pass 


# --- 2. Database Layer ---
class CourseRepository:
    def get_connection(self):
        try:
            if not connection_pool: return None
            connection = connection_pool.get_connection()
            if connection.is_connected(): return connection
        except Exception as err:
            logger.error(f"Pool Error: {err}")
            return None

    def normalize_code(self, text):
        # 1. Regex Search: Best for extracting "CS 116" from a sentence
        match = re.search(r'\b(cs|ce|ee|ie|math|engl|arb|gerl|mils|ne)\s*(\d{3,5}|0099|0098|100)\b', text.lower())
        if match:
            return f"{match.group(1).upper()}{match.group(2)}"
        
        # 2. Fallback: Handles "cs116" (no spaces)
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).upper()
        if 5 <= len(cleaned) <= 9 and any(char.isdigit() for char in cleaned):
             return cleaned
        return None

    def get_all_courses_dict(self):
        conn = self.get_connection()
        if not conn: return {}
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM courses")
            courses = {c['course_code']: c for c in cursor.fetchall()}
            
            # Add dummy German courses if missing (common issue)
            german_codes = ['GERL101', 'GERL102', 'GERL201', 'GERL202', 'GERL301', 'GERL302']
            for g in german_codes:
                if not any(g in k for k in courses.keys()):
                     courses[g] = {'course_code': g, 'course_name': f'German {g}', 'credit_hours': 3, 'description': 'German Language'}

            cursor.execute("SELECT * FROM prerequisites")
            for row in cursor.fetchall():
                c_code = row['course_code']
                p_code = row['prerequisite_code']
                if c_code in courses:
                    if 'prereqs' not in courses[c_code]: courses[c_code]['prereqs'] = set()
                    courses[c_code]['prereqs'].add(p_code)
            return courses
        finally:
            if conn.is_connected(): conn.close()

    def get_course_details(self, code):
        clean = self.normalize_code(code)
        conn = self.get_connection()
        if not conn: return None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM courses WHERE course_code = %s", (clean,))
            return cursor.fetchone()
        finally:
            if conn.is_connected(): conn.close()
            
    def get_prerequisites(self, code):
        clean_code = self.normalize_code(code)
        conn = self.get_connection()
        if not conn: return []
        try:
            cursor = conn.cursor()
            query = "SELECT p.prerequisite_code, c.course_name FROM prerequisites p LEFT JOIN courses c ON p.prerequisite_code = c.course_code WHERE p.course_code = %s"
            cursor.execute(query, (clean_code,))
            return cursor.fetchall()
        finally:
            if conn.is_connected(): conn.close()

    def fuzzy_find_instructor(self, user_text):
        conn = self.get_connection()
        if not conn: return None
        
        # Stop words to ignore
        stop_words = {"who", "is", "dr", "dr.", "prof", "prof.", "professor", "doctor", "where", "office", "email", "contact", "info", "the", "of", "tell", "me", "about"}
        words = re.findall(r'\b\w+\b', user_text.lower())
        search_terms = [w for w in words if w not in stop_words]
        
        if not search_terms: return None
        search_query = search_terms[-1] # Try the last significant word (e.g. "Hababeh")

        try:
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT * FROM instructors 
                WHERE LOWER(name) LIKE %s 
                OR LOWER(keywords) LIKE %s
            """
            param = f"%{search_query}%"
            cursor.execute(query, (param, param))
            results = cursor.fetchall()
            
            if results: return results[0] 
            
            # If SQL fails, try Python fuzzy match
            cursor.execute("SELECT * FROM instructors")
            all_instr = cursor.fetchall()
            names = [i['name'] for i in all_instr]
            matches = difflib.get_close_matches(user_text, names, n=1, cutoff=0.4)
            if matches:
                for i in all_instr:
                    if i['name'] == matches[0]: return i
            return None
        finally:
            if conn.is_connected(): conn.close()
            
    def get_course_attributes(self):
        conn = self.get_connection()
        if not conn: return {}
        
        attributes = {
            'common_compulsory': set(),
            'tracks': {
                'General': {'reqs': set(), 'electives': set()},
                'Data Science': {'reqs': set(), 'electives': set()},
                'Cybersecurity': {'reqs': set(), 'electives': set()}
            }
        }
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM course_attributes")
            rows = cursor.fetchall()
            
            for row in rows:
                code = row['course_code']
                attr_type = row['attribute_type']
                track = row['track_name']
                
                if attr_type == 'COMPULSORY':
                    attributes['common_compulsory'].add(code)
                elif attr_type == 'TRACK_REQ':
                    if track in attributes['tracks']:
                        attributes['tracks'][track]['reqs'].add(code)
                elif attr_type == 'ELECTIVE':
                    if track in attributes['tracks']:
                        attributes['tracks'][track]['electives'].add(code)
            return attributes
        finally:
            if conn.is_connected(): conn.close()


# --- 3. Context Manager ---
class ContextManager:
    TIMEOUT_SECONDS = 300  # <--- CHANGED: 5 Minutes (was 30s)

    def __init__(self):
        self.sessions = {}

    def get_context(self, user_id):
        current_time = time.time()

        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'status': 'idle',
                'track': None,
                'passed_courses': set(),
                'target_course': None,
                'last_entity': {'type': None, 'value': None},
                'last_interaction': current_time,
                'pending_intent': None
            }

        ctx = self.sessions[user_id]

        if current_time - ctx.get('last_interaction', 0) > self.TIMEOUT_SECONDS:
            # Reset only short-term status, try to keep data
            ctx['status'] = 'idle'
            ctx['target_course'] = None
            ctx['pending_intent'] = None

        ctx['last_interaction'] = current_time
        return ctx

    def update_passed_courses(self, user_id, text_input):
        ctx = self.get_context(user_id)
        codes = re.findall(r'\b[A-Z]{2,4}\s?\d{3,5}\b', text_input.upper())
        clean_list = [re.sub(r'[^a-zA-Z0-9]', '', c).upper() for c in codes]
        if clean_list:
            ctx['passed_courses'].update(clean_list)
            # Add remedial by default
            ctx['passed_courses'].update({'ARB0099', 'ENGL0098', 'ENGL0099', 'MATH0099'})
        return clean_list

    def extract_track(self, text):
        norm = text.lower()
        if "cyber" in norm: return "Cybersecurity"
        if "data" in norm: return "Data Science"
        if "general" in norm: return "General"
        return None

    def set_last_entity(self, user_id, entity_type, entity_value):
        self.get_context(user_id)['last_entity'] = {'type': entity_type, 'value': entity_value}

    def get_last_entity(self, user_id):
        return self.get_context(user_id)['last_entity']

    def clear_flow(self, user_id):
        ctx = self.get_context(user_id)
        ctx['status'] = 'idle'
        ctx['track'] = None
        ctx['passed_courses'] = set()
        ctx['target_course'] = None


# --- 4. Advisor Logic ---
class AcademicAdvisor:
    def __init__(self, repo: CourseRepository):
        self.repo = repo
        self.attributes = self.repo.get_course_attributes()
        
    def _get_track_rules(self, track):
        if not self.attributes:
             self.attributes = self.repo.get_course_attributes()
        common = self.attributes.get('common_compulsory', set())
        track_data = self.attributes.get('tracks', {}).get(track, {'reqs': set(), 'electives': set()})
        return common, track_data['reqs'], track_data['electives']

    def generate_plan(self, track, passed_courses):
        all_data = self.repo.get_all_courses_dict()
        if not all_data: return "⚠️ Error: Could not access course catalog. Please try again later."

        common_compulsory, track_reqs, track_electives = self._get_track_rules(track)
        all_compulsory = common_compulsory.union(track_reqs)
        total_passed_hours = sum(all_data[c]['credit_hours'] for c in passed_courses if c in all_data)
        
        eligible = []
        for code, info in all_data.items():
            if code in passed_courses: continue
            prereqs = info.get('prereqs', set())
            real_missing = (prereqs - passed_courses) - {'ARB0099', 'ENGL0098', 'ENGL0099', 'MATH0099'}
            
            if not real_missing:
                priority = 100
                level_match = re.search(r'\d', code)
                level = int(level_match.group(0)) if level_match else 9
                if code in all_compulsory: priority += 50
                elif code in track_electives: priority += 20
                else: priority -= 50
                priority -= (level * 10)
                info['score'] = priority
                eligible.append(info)

        eligible.sort(key=lambda x: x['score'], reverse=True)
        
        schedule = []
        current_credits = 0
        MAX_CREDITS = 18
        
        for course in eligible:
            if course['course_code'] in ['CS391', 'CS491', 'CS492'] and total_passed_hours < 90: continue
            if current_credits + course['credit_hours'] <= MAX_CREDITS:
                schedule.append(course)
                current_credits += course['credit_hours']

        if not schedule:
            return f"🎓 **Semester Plan ({track})**\nYou have no eligible courses left or you've finished everything! 🎉"

        resp = f"🎓 **Semester Plan for {track}**\n(Completed Hours: ~{total_passed_hours})\n\n**Suggested Schedule ({current_credits} Cr):**\n"
        for c in schedule:
            is_req = c['course_code'] in track_reqs
            is_comp = c['course_code'] in common_compulsory
            tag = " ⭐(Track Req)" if is_req else (" (Core)" if is_comp else "")
            resp += f"- **{c['course_code']}**: {c['course_name']} ({c['credit_hours']} Cr){tag}\n"
            
        if current_credits < 12: resp += "\n⚠️ Credits are low. You might need to finish prerequisites first."
        return resp

    def check_eligibility(self, target_code, passed_courses):
        details = self.repo.get_course_details(target_code)
        if not details: return f"Course **{target_code}** not found in the catalog."
        prereqs_raw = self.repo.get_prerequisites(target_code)
        required_codes = {p[0] for p in prereqs_raw}
        missing = (required_codes - passed_courses) - {'ARB0099', 'ENGL0098', 'ENGL0099', 'MATH0099'}
        
        if not missing: 
            return f"✅ **Eligible!** You can take **{target_code}** ({details['course_name']})."
        else:
            resp = f"❌ **Not Eligible** for {target_code}. You are missing:\n"
            for m in missing: resp += f"- {m}\n"
            return resp

    def check_graduation(self, track, passed_courses):
        all_data = self.repo.get_all_courses_dict()
        common_compulsory, track_reqs, track_electives = self._get_track_rules(track)
        all_compulsory = common_compulsory.union(track_reqs)
        
        remaining_compulsory = {c for c in (all_compulsory - passed_courses) if c in all_data}
        passed_electives = [c for c in passed_courses if c in track_electives]
        total_hours = sum(all_data[c]['credit_hours'] for c in passed_courses if c in all_data)
        
        resp = f"🎓 **Graduation Status ({track})**\n📊 Total Hours: **{total_hours}** / ~132\n"
        
        if not remaining_compulsory: 
            resp += "✅ All Compulsory Courses Completed!\n"
        else:
            resp += f"⚠️ **Missing Compulsory ({len(remaining_compulsory)}):**\n"
            limit = 0
            for c in sorted(list(remaining_compulsory)):
                if limit < 8:
                    resp += f"- {c}\n"
                    limit += 1
            if len(remaining_compulsory) > 8: resp += f"...and {len(remaining_compulsory)-8} more.\n"
        
        resp += f"\nℹ️ Track Electives Passed: **{len(passed_electives)}**\n(You typically need 4-5 electives depending on your plan)."
        return resp