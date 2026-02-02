
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os
import datetime

# Global variable to hold the db reference
db = None

def init_firebase():
    """
    Initializes Firebase Admin SDK.
    Expects 'FIREBASE_CREDENTIALS_PATH' env var pointing to the service account JSON,
    or relies on Google Application Default Credentials if running in a cloud environment that supports it.
    """
    global db
    if db is not None:
        return db

    try:
        # Check if already initialized
        if not firebase_admin._apps:
            cred_path = os.environ.get('FIREBASE_CREDENTIALS_PATH')
            
            # Auto-detect if file exists in root and env var is missing
            if not cred_path:
                possible_keys = ['firebase_key.json', 'serviceAccountKey.json']
                base_dir = os.getcwd() # or modify to find project root dynamically if needed
                for key_file in possible_keys:
                    full_path = os.path.join(base_dir, key_file)
                    if os.path.exists(full_path):
                        cred_path = full_path
                        # Also attempt to find if we are in a subdir (e.g. running from flask_app)
                        break
                    # Check parent dir just in case
                    parent_path = os.path.join(os.path.dirname(base_dir), key_file)
                    if os.path.exists(parent_path):
                         cred_path = parent_path
                         break

            if cred_path and os.path.exists(cred_path):
                print(f"Initializing Firebase with credentials at: {cred_path}")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            else:
                print("Initializing Firebase with Application Default Credentials (or no explicit creds found)")
                firebase_admin.initialize_app()
        
        db = firestore.client()
        print("Firebase initialized successfully.")
        return db
    except Exception as e:
        print(f"Failed to initialize Firebase: {e}")
        return None

def save_experiment_result(data):
    """
    Saves the experiment result to the 'experiments' collection in Firestore.
    """
    global db
    if db is None:
        db = init_firebase()
    
    if db is None:
        print("Skipping Firebase save: Database not initialized.")
        return False

    try:
        # Add a timestamp
        doc_data = data.copy()
        doc_data['timestamp'] = datetime.datetime.utcnow()
        
        # Remove heavy plot data if present to save space/bandwidth, 
        # or keep it if the user wants full history. 
        # For now, let's remove the serialized plots as they can be large strings.
        if 'plots' in doc_data:
            del doc_data['plots']

        db.collection('experiments').add(doc_data)
        print("Experiment saved to Firebase.")
        return True
    except Exception as e:
        print(f"Error saving to Firebase: {e}")
        return False
