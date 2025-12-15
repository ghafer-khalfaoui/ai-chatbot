import json
import mysql.connector

# --- Database Configuration ---
db_config = {
    'host': 'localhost',
    'user': 'root', 
    'password': '', 
    'database': 'chatbot_db'
}

# --- Function to connect to the database ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        print("✅ Database connection successful.")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Error connecting to MySQL: {err}")
        return None

# --- Function to load data ---
def load_data():
    conn = get_db_connection()
    if not conn:
        print("Aborting: Could not connect to database.")
        return

    try:
        cursor = conn.cursor()
        
        print("Opening gju_data.json...")
        with open('gju_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # --- 1. Load Instructors ---
        print("\nLoading instructors...")
        cursor.execute("TRUNCATE TABLE instructors")
        
        inst_query = """
        INSERT INTO instructors (name, title, office_location, phone, email, status)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        for prof in data['instructors']:
            cursor.execute(inst_query, (
                prof['name'], prof['title'], prof['office_location'], 
                prof['phone'], prof['email'], prof['status']
            ))
        print(f"Successfully loaded {len(data['instructors'])} instructors.")

        # --- 2. Load Courses ---
        print("\nLoading courses...")
        cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
        cursor.execute("TRUNCATE TABLE courses")
        cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
        
        course_query = """
        INSERT INTO courses (course_code, course_name, credit_hours, description)
        VALUES (%s, %s, %s, %s)
        """
        for course in data['courses']:
            # --- THIS IS THE FIX ---
            # Remove spaces from the course code
            clean_code = course['code'].replace(" ", "")
            cursor.execute(course_query, (
                clean_code, course['name'], course['hours'], course['desc']
            ))
        print(f"Successfully loaded {len(data['courses'])} courses.")

        # --- 3. Load Prerequisites ---
        print("\nLoading prerequisites...")
        cursor.execute("TRUNCATE TABLE prerequisites")
        
        prereq_query = """
        INSERT INTO prerequisites (course_code, prerequisite_code)
        VALUES (%s, %s)
        """
        for prereq in data['prerequisites']:
            # --- THIS IS THE FIX ---
            # Remove spaces from both codes
            clean_course_code = prereq['course'].replace(" ", "")
            clean_prereq_code = prereq['prereq'].replace(" ", "")
            cursor.execute(prereq_query, (
                clean_course_code, clean_prereq_code
            ))
        print(f"Successfully loaded {len(data['prerequisites'])} prerequisites.")

        conn.commit()
        print("\n✅ All data has been successfully loaded (with no spaces)!")

    except mysql.connector.Error as err:
        print(f"❌ A database error occurred: {err}")
        conn.rollback()
    except FileNotFoundError:
        print("❌ ERROR: 'gju_data.json' file not found.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    load_data()