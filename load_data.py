import json
import mysql.connector
import sys

# --- Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', 
    'password': '', 
    'database': 'chatbot_db'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Connection Error: {err}")
        return None

def load_data():
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        print("📂 Opening gju_data.json...")
        with open('gju_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Disable FK checks to allow truncation
        cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
        
        # 1. Instructors
        print("Processing Instructors...")
        cursor.execute("TRUNCATE TABLE instructors")
        for prof in data['instructors']:
            cursor.execute("""
                INSERT INTO instructors (name, title, office_location, phone, email, status)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (prof['name'], prof['title'], prof['office_location'], prof['phone'], prof['email'], prof['status']))

        # 2. Courses
        print("Processing Courses...")
        cursor.execute("TRUNCATE TABLE courses")
        for course in data['courses']:
            clean_code = course['code'].replace(" ", "").upper() # Ensure consistency
            cursor.execute("""
                INSERT INTO courses (course_code, course_name, credit_hours, description)
                VALUES (%s, %s, %s, %s)
            """, (clean_code, course['name'], course['hours'], course['desc']))

        # 3. Prerequisites
        print("Processing Prerequisites...")
        cursor.execute("TRUNCATE TABLE prerequisites")
        for prereq in data['prerequisites']:
            c_code = prereq['course'].replace(" ", "").upper()
            p_code = prereq['prereq'].replace(" ", "").upper()
            cursor.execute("""
                INSERT INTO prerequisites (course_code, prerequisite_code)
                VALUES (%s, %s)
            """, (c_code, p_code))

        cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
        conn.commit()
        print("✅ Data loaded successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    load_data()