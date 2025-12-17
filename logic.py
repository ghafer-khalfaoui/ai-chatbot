import mysql.connector # type: ignore
import re
import difflib
import logging
import os
import json
import pickle
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# --- Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # <--- CHECK YOUR PASSWORD
    'database': 'chatbot_db'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LogicLayer")

# --- Constants ---
ELECTIVES_GENERAL = {"CS333", "CS357", "CS358", "CS359", "CS364", "CS365", "CS371", "CS430", "CS432", "CS439", "CS450", "CS457", "CS458", "CS460", "CS462", "CS481", "CS482", "CS484", "CS489", "CS4512", "CS4811", "CS4831", "CS4832", "CS4833"}
ELECTIVES_DATA_SCIENCE = {"CS358", "CS359", "CS364", "CS371", "CS432", "CS450", "CS456", "CS457", "CS462", "CS484", "CS4512", "CS4811", "CS4813", "CS4831", "CS4832", "CS4833"}
ELECTIVES_CYBERSECURITY = {"CS354", "CS357", "CS359", "CS370", "CS372", "CS373", "CS374", "CS458", "CS4511", "CS4711", "CS4712", "CS4715", "CS4812", "CS4831", "CS4832", "CS4833"}
ALL_ELECTIVES = ELECTIVES_GENERAL.union(ELECTIVES_DATA_SCIENCE).union(ELECTIVES_CYBERSECURITY)

# --- 1. AI/Model Layer ---
class AIModel:
    def __init__(self, model_dir="bert_intent_model"):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.max_len = 50
        self.intents = {}
        self.loaded = False
        self._load_resources(model_dir)

    def _load_resources(self, model_dir):
        try:
            if os.path.isdir(model_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
                
                with open(os.path.join(model_dir, 'label_encoder.pickle'), 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                with open(os.path.join(model_dir, 'max_len.json'), 'r') as f:
                    self.max_len = json.load(f)['max_len']
                
                with open('intents.json', 'r', encoding='utf-8') as f:
                    self.intents = json.load(f)
                    
                self.loaded = True
                logger.info("✅ BERT Model and resources loaded successfully.")
            else:
                logger.warning(f"⚠️ Model directory {model_dir} not found. Please run train_model.py first.")
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")

    def predict_intent(self, text):
        if not self.loaded: return None, 0.0
        try:
            inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding='max_length', max_length=self.max_len)
            logits = self.model(inputs).logits
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            pred_index = np.argmax(probs)
            confidence = probs[pred_index]
            tag = self.label_encoder.inverse_transform([pred_index])[0]
            return tag, confidence
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0

    def get_response_for_tag(self, tag):
        for intent in self.intents.get('intents', []):
            if intent['tag'] == tag:
                import random
                return random.choice(intent['responses'])
        return None

# --- 2. Database Layer ---
class CourseRepository:
    def get_connection(self):
        try:
            return mysql.connector.connect(**DB_CONFIG)
        except mysql.connector.Error as err:
            logger.error(f"DB Connection Error: {err}")
            return None

    def normalize_code(self, raw_code):
        # Extract CS116 from "about cs 116"
        match = re.search(r'\b(cs|ce|ee|ie|math|engl|arb|gerl|mils|ne)\s*(\d{3,5}|0099|0098|100)\b', raw_code.lower())
        if match:
            return f"{match.group(1).upper()}{match.group(2)}"
        return re.sub(r'[^a-zA-Z0-9]', '', raw_code).upper()

    def get_all_courses_dict(self):
        conn = self.get_connection()
        if not conn: return {}
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM courses")
            courses = {c['course_code']: c for c in cursor.fetchall()}
            
            # Inject German if missing
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
            conn.close()

    def get_course_details(self, code):
        clean = self.normalize_code(code)
        conn = self.get_connection()
        if not conn: return None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM courses WHERE course_code = %s", (clean,))
            return cursor.fetchone()
        finally:
            conn.close()
            
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
            conn.close()

    def fuzzy_find_instructor(self, user_text):
        norm = user_text.lower()
        match = re.search(r'(dr\.|prof\.|dr|prof)\s*([a-z\-\'\s]+)', norm)
        search_term = match.group(2).strip() if match else user_text
        
        conn = self.get_connection()
        if not conn: return None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM instructors WHERE name LIKE %s", (f"%{search_term}%",))
            results = cursor.fetchall()
            if results: return results[0] 
            
            cursor.execute("SELECT * FROM instructors")
            all_instr = cursor.fetchall()
            names = [i['name'] for i in all_instr]
            matches = difflib.get_close_matches(search_term, names, n=1, cutoff=0.4)
            if matches:
                for i in all_instr:
                    if i['name'] == matches[0]: return i
            return None
        finally:
            conn.close()

# --- 3. Context Manager ---
class ContextManager:
    def __init__(self):
        self.sessions = {}

    def get_context(self, user_id):
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'status': 'idle',
                'track': None,
                'passed_courses': set(),
                'target_course': None,
                'last_entity': {'type': None, 'value': None},
            }
        return self.sessions[user_id]

    def update_passed_courses(self, user_id, text_input):
        ctx = self.get_context(user_id)
        codes = re.findall(r'\b[A-Z]{2,4}\s?\d{3,5}\b', text_input.upper())
        clean_list = [re.sub(r'[^a-zA-Z0-9]', '', c).upper() for c in codes]
        if clean_list:
            ctx['passed_courses'].update(clean_list)
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
        self.TRACKS = {
            "General": {"CS330","CS332","CS419","CS477"},
            "Data Science": {"CS330","CE377","CS460","EE570"},
            "Cybersecurity": {"CE354","CS470","CS4713","CS4714"}
        }
        self.COMMON_COMPULSORY = {"CS201","CE201","CS222","CS223","CS263","CS264","CS323","CS342","CE352","CS355","CS356","CE357","CE3570","CS416","CS451","MATH101","MATH102","CS116","CS1160","CS117","CS1170","CE212","CE2120","EE317","IE0121","CS391","CS491","CS492","ARB100","ENGL1001","ENGL1002","MILS100","NE101"}

    def generate_plan(self, track, passed_courses):
        all_data = self.repo.get_all_courses_dict()
        if not all_data: return "Database error."

        track_reqs = self.TRACKS.get(track, set())
        all_compulsory = self.COMMON_COMPULSORY.union(track_reqs)
        track_electives = ELECTIVES_CYBERSECURITY if track == "Cybersecurity" else (ELECTIVES_DATA_SCIENCE if track == "Data Science" else ELECTIVES_GENERAL)

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
                
                if code.startswith(('CS', 'CE', 'MATH', 'EE', 'IE')): priority = level * 10
                elif code in ALL_ELECTIVES: priority = 60 + level
                else: priority = 80 + level
                
                if code in all_compulsory: priority -= 50
                if code in track_electives: priority -= 2
                
                info['score'] = priority
                eligible.append(info)

        eligible.sort(key=lambda x: x['score'])
        
        schedule = []
        current_credits = 0
        MAX_CREDITS = 18
        
        for course in eligible:
            if course['course_code'] in ['CS391', 'CS491', 'CS492'] and total_passed_hours < 90: continue
            if current_credits + course['credit_hours'] <= MAX_CREDITS:
                schedule.append(course)
                current_credits += course['credit_hours']

        resp = f"🎓 **Semester Plan for {track}**\n(Completed Hours: ~{total_passed_hours})\n\n**Suggested Schedule ({current_credits} Cr):**\n"
        for c in schedule:
            tag = " (Track Req)" if c['course_code'] in track_reqs else ""
            resp += f"- **{c['course_code']}**: {c['course_name']} ({c['credit_hours']} Cr){tag}\n"
        if current_credits < 12: resp += "\n⚠️ Credits are low. Finish prerequisites first."
        return resp

    def check_eligibility(self, target_code, passed_courses):
        details = self.repo.get_course_details(target_code)
        if not details: return f"Course **{target_code}** not found."
        
        prereqs_raw = self.repo.get_prerequisites(target_code)
        required_codes = {p[0] for p in prereqs_raw}
        missing = (required_codes - passed_courses) - {'ARB0099', 'ENGL0098', 'ENGL0099', 'MATH0099'}
        
        if not missing: return f"✅ **Eligible!** You can take **{target_code}** ({details['course_name']})."
        else:
            resp = f"❌ **Not Eligible** for {target_code}. You need to pass:\n"
            for m in missing: resp += f"- {m}\n"
            return resp

    def check_graduation(self, track, passed_courses):
        all_data = self.repo.get_all_courses_dict()
        track_reqs = self.TRACKS.get(track, set())
        all_compulsory = self.COMMON_COMPULSORY.union(track_reqs)
        
        remaining = {c for c in (all_compulsory - passed_courses) if c in all_data}
        total_hours = sum(all_data[c]['credit_hours'] for c in passed_courses if c in all_data)
        
        resp = f"🎓 **Graduation Status ({track})**\n📊 Total Hours: **{total_hours}** / 145\n"
        if not remaining: resp += "✅ All Compulsory Courses Completed!\n"
        else:
            resp += f"⚠️ **Missing Compulsory Courses ({len(remaining)}):**\n"
            limit = 0
            for c in sorted(list(remaining)):
                if limit < 10:
                    resp += f"- {c}: {all_data[c]['course_name']}\n"
                    limit += 1
            if len(remaining) > 10: resp += f"...and {len(remaining)-10} more.\n"
        
        passed_electives = [c for c in passed_courses if c in ALL_ELECTIVES]
        resp += f"\nℹ️ Electives Passed: {len(passed_electives)} (Target ~4)"
        return resp