import os
import json
import random
import string
from asyncio import current_task
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import asyncio
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB imports
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Telegram Bot imports
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Environment variables
BOT_TOKEN = os.getenv('BOT_TOKEN')
MONGO_URI = os.getenv('MONGO_URI')


# Download required NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


download_nltk_data()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)


class MedicalChatBot:
    def __init__(self):
        # MongoDB connection setup
        try:
            # Use environment variable for MongoDB connection
            self.mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client['testDb']
            self.dates_collection = self.db['dates']
            self.doctors_collection = self.db['doctors']
            self.patients_collection = self.db['patients']
            self.confirmations_collection = self.db['confirmations']

            # Test the connection
            self.mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise Exception("MongoDB connection failed")

        # In-memory storage for testing
        self.patients_data = {}

        # Booked slots storage - format: {date_time_key: [slot1, slot2, ...]}
        # date_time_key format: "2025-09-29_16:38 PM"
        self.booked_slots = {}

        # Blood groups
        self.blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Symptom-Disease mapping for basic NLP analysis
        self.symptom_disease_map = {
            'fever': ['Viral Infection', 'Bacterial Infection', 'Flu'],
            'headache': ['Migraine', 'Tension Headache', 'Sinusitis'],
            'cough': ['Common Cold', 'Bronchitis', 'Pneumonia'],
            'blocked': ['Common Cold', 'Allergic Rhinitis', 'Sinusitis'],
            'nose': ['Common Cold', 'Allergic Rhinitis', 'Sinusitis'],
            'sore': ['Viral Pharyngitis', 'Strep Throat', 'Common Cold'],
            'throat': ['Viral Pharyngitis', 'Strep Throat', 'Common Cold'],
            'body': ['Flu', 'Viral Infection', 'Muscle Strain'],
            'pain': ['Flu', 'Viral Infection', 'Muscle Strain'],
            'nausea': ['Food Poisoning', 'Gastroenteritis', 'Migraine'],
            'vomiting': ['Food Poisoning', 'Gastroenteritis', 'Viral Infection'],
            'diarrhea': ['Food Poisoning', 'Gastroenteritis', 'IBS'],
            'fatigue': ['Viral Infection', 'Anemia', 'Chronic Fatigue'],
            'chest': ['Acid Reflux', 'Muscle Strain', 'Anxiety'],
            'shortness': ['Asthma', 'Anxiety', 'Respiratory Infection'],
            'breath': ['Asthma', 'Anxiety', 'Respiratory Infection'],
            'cold': ['Common Cold', 'Viral Infection'],
            'runny': ['Common Cold', 'Allergic Rhinitis'],
            'sneezing': ['Common Cold', 'Allergic Rhinitis'],
            'weakness': ['Viral Infection', 'Anemia', 'Dehydration']
        }

        # User states
        self.user_states = {}

    def generate_unique_code(self):
        """Generate 8-digit unique code and ensure it's unique in database"""
        while True:
            code = ''.join(random.choices(string.digits, k=8))
            # Check if code already exists in patients collection
            existing_patient = self.patients_collection.find_one({"uniqueCode": code})
            if not existing_patient:
                return code

    async def get_booking_details_by_code(self, unique_code):
        """Get complete booking details by unique code"""
        try:
            # Find patient by unique code
            patient = self.patients_collection.find_one({"uniqueCode": unique_code})
            if not patient:
                return None

            # Find confirmation for this patient
            confirmation = self.confirmations_collection.find_one({"patient": patient["_id"]})
            if not confirmation:
                return None

            # Get doctor details if doctor ID exists
            doctor_name = confirmation.get("doctorName", "N/A")
            doctor_specialty = "N/A"

            if confirmation.get("doctor"):
                doctor = self.doctors_collection.find_one({"_id": confirmation["doctor"]})
                if doctor:
                    doctor_specialty = doctor.get("specialty", "N/A")

            # Format the booking details
            booking_details = {
                "uniqueCode": unique_code,
                "patient": {
                    "name": patient.get("name", ""),
                    "age": patient.get("age", ""),
                    "gender": patient.get("gender", ""),
                    "blood": patient.get("blood", ""),
                    "contact": patient.get("contact", ""),
                },
                "doctor": {
                    "name": doctor_name,
                    "specialty": doctor_specialty,
                },
                "appointment": {
                    "date": confirmation.get("date", ""),
                    "status": confirmation.get("status", ""),
                    "createdAt": confirmation.get("createdAt", ""),
                },
                "confirmationId": str(confirmation.get("_id", "")),
                "patientId": str(patient.get("_id", ""))
            }

            return booking_details

        except Exception as e:
            logger.error(f"Error fetching booking details: {e}")
            return None

    # MongoDB Data Functions
    async def save_patient_to_db(self, patient_data, unique_code):
        """Save patient information to MongoDB patients collection with unique code"""
        try:
            patient_document = {
                "name": patient_data.get('name', ''),
                "age": int(patient_data.get('age', 0)),
                "gender": patient_data.get('gender', ''),
                "blood": patient_data.get('blood_group', ''),
                "contact": patient_data.get('contact', ''),
                "uniqueCode": unique_code,
                "symptoms": patient_data.get('symptoms', []),
                "matchedSymptoms": patient_data.get('matched_symptoms', []),
                "possibleDiseases": patient_data.get('possible_diseases', []),
                "createdAt": datetime.now(),
                "updatedAt": datetime.now()
            }

            result = self.patients_collection.insert_one(patient_document)
            logging.info(f"Patient saved successfully with ID: {result.inserted_id}")
            return result.inserted_id

        except Exception as e:
            logging.error(f"Error saving patient: {e}")
            return None

    async def save_confirmation_to_db(self, patient_id, confirmation_data, unique_code):
        """Save appointment confirmation to MongoDB confirmations collection with unique code"""
        try:
            confirmation_document = {
                "patient": patient_id,
                "doctor": confirmation_data.get('doctor_id'),
                "doctorName": confirmation_data.get('doctor_name', ''),
                "date": confirmation_data.get('appointment_date'),
                # "time": confirmation_data.get('appointment_time', ''),
                "slot": confirmation_data.get('selected_slot', ''),
                "status": confirmation_data.get('status', 'confirmed'),
                "uniqueCode": unique_code,
                "createdAt": datetime.now(),
                "updatedAt": datetime.now()
            }

            result = self.confirmations_collection.insert_one(confirmation_document)
            logging.info(f"Confirmation saved successfully with ID: {result.inserted_id}")
            return result.inserted_id

        except Exception as e:
            logging.error(f"Error saving confirmation: {e}")
            return None

    async def complete_booking_process(self, user_id, slot_data):
        """Complete booking process by saving both patient and confirmation data with unique code"""
        try:
            patient_data = self.user_states[user_id]['patient_data']

            # Generate unique code
            unique_code = self.generate_unique_code()

            # Step 1: Save patient information with unique code
            patient_id = await self.save_patient_to_db(patient_data, unique_code)
            if not patient_id:
                return False, "Failed to save patient information"

            # Step 2: Prepare confirmation data
            confirmation_data = {
                'doctor_id': patient_data.get('selected_doctor', {}).get('_id'),
                'doctor_name': patient_data.get('selected_doctor', {}).get('name', ''),
                'appointment_date': patient_data.get('selected_date'),
                'appointment_time': patient_data.get('selected_time', ''),
                'selected_slot': patient_data.get('selected_slot', ''),
                'status': 'confirmed'
            }

            # Step 3: Save confirmation with unique code
            confirmation_id = await self.save_confirmation_to_db(patient_id, confirmation_data, unique_code)
            if not confirmation_id:
                return False, "Failed to save confirmation"

            # Step 4: Store in memory for immediate access (optional)
            self.patients_data[unique_code] = {
                'name': patient_data['name'],
                'age': patient_data['age'],
                'gender': patient_data['gender'],
                'blood_group': patient_data['blood_group'],
                'contact': patient_data['contact'],
                'symptoms': patient_data['symptoms'],
                'matched_symptoms': patient_data.get('matched_symptoms', []),
                'possible_diseases': patient_data.get('possible_diseases', []),
                'doctor_name': confirmation_data['doctor_name'],
                'doctor_specialty': patient_data.get('selected_doctor', {}).get('specialty', ''),
                'selected_date': confirmation_data['appointment_date'],
                'selected_date_display': patient_data.get('selected_date_display', ''),
                'selected_time': confirmation_data['appointment_time'],
                'selected_slot': confirmation_data['selected_slot'],
            }

            return True, {
                "patient_id": patient_id,
                "confirmation_id": confirmation_id,
                "unique_code": unique_code
            }

        except Exception as e:
            logging.error(f"Error in complete booking process: {e}")
            return False, f"Booking failed: {str(e)}"

    # Existing methods continue here...
    def get_available_doctors(self):
        """Get list of available doctors from MongoDB"""
        try:
            doctors_cursor = self.doctors_collection.find({"isDeleted": False})
            doctors = []
            for doc in doctors_cursor:
                doctor_info = {
                    '_id': str(doc.get('_id')),
                    'name': doc.get('name', ''),
                    'firstName': doc.get('firstName', ''),
                    'lastName': doc.get('lastName', ''),
                    'specialty': doc.get('specialty', ''),
                    'qualification': doc.get('qualification', ''),
                    'availability': doc.get('availability', '')
                }
                doctors.append(doctor_info)
            logger.info(f"Found {len(doctors)} available doctors")
            return doctors
        except Exception as e:
            logger.error(f"Error fetching doctors: {e}")
            return []

    def parse_time_slot(self, time_str):
        """Parse time slot string to datetime object for comparison"""
        try:
            # Handle formats like "2:00 PM", "14:00", "2 PM"
            time_str = time_str.strip().upper()
            # Try parsing with AM/PM
            for fmt in ["%I:%M %p", "%I %p", "%H:%M"]:
                try:
                    return datetime.strptime(time_str, fmt).time()
                except ValueError:
                    continue
            return None
        except Exception as e:
            logger.error(f"Error parsing time slot {time_str}: {e}")
            return None

    def parse_availability_range(self, availability_str):
        """Parse availability string like '2:00 PM - 5:00 PM' into start and end times"""
        try:
            if not availability_str or '-' not in availability_str:
                return None, None
            parts = availability_str.split('-')
            if len(parts) != 2:
                return None, None
            start_time = self.parse_time_slot(parts[0].strip())
            end_time = self.parse_time_slot(parts[1].strip())
            return start_time, end_time
        except Exception as e:
            logger.error(f"Error parsing availability range {availability_str}: {e}")
            return None, None

    def is_slot_in_availability(self, slot_time_str, availability_str):
        """Check if a slot time falls within doctor's availability"""
        try:
            slot_time = self.parse_time_slot(slot_time_str)
            if not slot_time:
                return False
            start_time, end_time = self.parse_availability_range(availability_str)
            if not start_time or not end_time:
                return True  # If can't parse availability, show all slots
            return start_time <= slot_time <= end_time
        except Exception as e:
            logger.error(f"Error checking slot availability: {e}")
            return False

    def filter_slots_by_doctor_availability(self, slots, doctor_availability):
        """Filter slots based on doctor's availability"""
        if not doctor_availability:
            return slots
        filtered_slots = [slot for slot in slots if self.is_slot_in_availability(slot, doctor_availability)]
        return filtered_slots

    def is_slot_passed(self, date_str, time_str):
        """Check if a slot has already passed for today"""
        try:
            slot_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            current_date = datetime.now().date()

            # If slot is in the future, it hasn't passed
            if slot_date > current_date:
                return False

            # If slot is in the past, it has passed
            if slot_date < current_date:
                return True

            # If slot is today, check the time
            slot_time = self.parse_time_slot(time_str)
            if not slot_time:
                return False

            current_time = datetime.now().time()
            return slot_time < current_time
        except Exception as e:
            logger.error(f"Error checking if slot passed: {e}")
            return False

    def parse_date_from_db(self, date_str):
        """Parse date string from MongoDB format to datetime object"""
        try:
            # Handle different date formats that might be in the database
            formats_to_try = [
                "%m/%d/%Y",  # 09/29/2025
                "%d/%m/%Y",  # 29/09/2025
                "%Y-%m-%d",  # 2025-09-29
                "%Y/%m/%d"  # 2025/09/29
            ]

            for date_format in formats_to_try:
                try:
                    return datetime.strptime(date_str, date_format)
                except ValueError:
                    continue

            # If no format works, try to parse manually
            parts = date_str.replace('-', '/').split('/')
            if len(parts) == 3:
                # Try different orders
                try:
                    month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                    return datetime(year, month, day)
                except ValueError:
                    try:
                        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                        return datetime(year, month, day)
                    except ValueError:
                        pass

            logger.warning(f"Could not parse date: {date_str}")
            return None
        except Exception as e:
            logger.error(f"Error parsing date {date_str}: {e}")
            return None

    def create_date_time_key(self, date_str, time_str):
        """Create a unique key for date-time combination"""
        try:
            # Convert date to standard format
            date_obj = self.parse_date_from_db(date_str)
            if date_obj:
                formatted_date = date_obj.strftime("%Y-%m-%d")
                return f"{formatted_date}_{time_str}"
            return f"{date_str}_{time_str}"
        except Exception as e:
            logger.error(f"Error creating date-time key: {e}")
            return f"{date_str}_{time_str}"

    def get_next_7_upcoming_dates(self, doctor_availability=None):
        """Get the next 7 upcoming dates from MongoDB with available slots"""
        try:
            # Get current date
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Fetch all documents from MongoDB
            dates_cursor = self.dates_collection.find({})

            # Parse and filter dates
            date_time_slots = []

            for doc in dates_cursor:
                date_str = doc.get('date', '')
                time_str = doc.get('time', '')
                slots = doc.get('slots', [])

                if not date_str or not time_str or not slots:
                    continue

                # Parse the date
                parsed_date = self.parse_date_from_db(date_str)
                if not parsed_date:
                    continue

                # Skip past dates
                if parsed_date < current_date:
                    continue

                # Create date-time key for booked slots lookup
                date_time_key = self.create_date_time_key(date_str, time_str)

                # Extract time values from slots array (each slot is an object with 'time' field)
                available_slots = []
                booked_for_this_slot = self.booked_slots.get(date_time_key, [])

                for slot in slots:
                    # Handle both object format and string format
                    if isinstance(slot, dict):
                        slot_time = slot.get('time', '')
                    else:
                        slot_time = str(slot)

                    if slot_time and slot_time not in booked_for_this_slot:
                        # Check if slot has passed (for today only)
                        formatted_date = parsed_date.strftime("%Y-%m-%d")
                        if not self.is_slot_passed(formatted_date, slot_time):
                            # Filter by doctor availability if provided
                            if doctor_availability:
                                if self.is_slot_in_availability(slot_time, doctor_availability):
                                    available_slots.append(slot_time)
                            else:
                                available_slots.append(slot_time)

                # Only add if there are available slots
                if available_slots:
                    date_time_slots.append({
                        'date': parsed_date.strftime("%Y-%m-%d"),
                        'date_obj': parsed_date,
                        'time': time_str,
                        'available_slots': available_slots,
                        'date_time_key': date_time_key
                    })

            # Sort by date and time
            date_time_slots.sort(key=lambda x: (x['date_obj'], x['time']))

            # Group by date and get next 7 dates
            dates_map = {}
            for slot_info in date_time_slots:
                date_key = slot_info['date']
                if date_key not in dates_map:
                    dates_map[date_key] = {
                        'date': date_key,
                        'date_obj': slot_info['date_obj'],
                        'display_name': f"{slot_info['date_obj'].strftime('%A')}, {slot_info['date_obj'].strftime('%B %d, %Y')}",
                        'time_slots': [],
                        'total_available_slots': 0
                    }

                dates_map[date_key]['time_slots'].append({
                    'time': slot_info['time'],
                    'available_slots': slot_info['available_slots'],
                    'date_time_key': slot_info['date_time_key']
                })
                dates_map[date_key]['total_available_slots'] += len(slot_info['available_slots'])

            # Get first 7 dates with available slots
            available_dates = []
            for date_key in sorted(dates_map.keys(), key=lambda x: dates_map[x]['date_obj']):
                if dates_map[date_key]['total_available_slots'] > 0:
                    available_dates.append(dates_map[date_key])
                    if len(available_dates) >= 7:
                        break

            return available_dates
        except Exception as e:
            logger.error(f"Error fetching upcoming dates: {e}")
            return []

    def get_available_dates(self, doctor_availability=None):
        """Get list of available dates (wrapper for backward compatibility)"""
        return self.get_next_7_upcoming_dates(doctor_availability)

    def get_available_slots_for_date(self, date_str, doctor_availability=None):
        """Get all available time slots for a specific date"""
        try:
            available_dates = self.get_next_7_upcoming_dates(doctor_availability)
            for date_info in available_dates:
                if date_info['date'] == date_str:
                    # Return all time slots with their available slots
                    return date_info['time_slots']
            return []
        except Exception as e:
            logger.error(f"Error fetching slots for date {date_str}: {e}")
            return []

    def get_slots_for_time_slot(self, date_time_key, doctor_availability=None):
        """Get available slots for a specific date-time combination"""
        try:
            # Parse the date_time_key to get date and time
            parts = date_time_key.split('_', 1)
            if len(parts) != 2:
                return []

            date_part, time_part = parts

            # Convert date back to original format for MongoDB query
            date_obj = datetime.strptime(date_part, "%Y-%m-%d")
            db_date_format = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"

            # Find the document in MongoDB
            doc = self.dates_collection.find_one({"date": db_date_format, "time": time_part})
            if not doc:
                return []

            all_slots = doc.get('slots', [])
            booked_slots = self.booked_slots.get(date_time_key, [])

            # Extract time values from slots (handle object format)
            available_times = []
            for slot in all_slots:
                # Handle both object format and string format
                if isinstance(slot, dict):
                    slot_time = slot.get('time', '')
                else:
                    slot_time = str(slot)

                if slot_time and slot_time not in booked_slots:
                    # Check if slot has passed
                    if not self.is_slot_passed(date_part, slot_time):
                        # Filter by doctor availability if provided
                        if doctor_availability:
                            if self.is_slot_in_availability(slot_time, doctor_availability):
                                available_times.append(slot_time)
                        else:
                            available_times.append(slot_time)

            return available_times
        except Exception as e:
            logger.error(f"Error getting slots for {date_time_key}: {e}")
            return []

    def book_slot(self, date_time_key, slot):
        """Book a slot by adding it to booked_slots"""
        try:
            if date_time_key not in self.booked_slots:
                self.booked_slots[date_time_key] = []

            if slot not in self.booked_slots[date_time_key]:
                self.booked_slots[date_time_key].append(slot)
                logger.info(f"Successfully booked slot {slot} for {date_time_key}")
                return True
            else:
                logger.warning(f"Slot {slot} for {date_time_key} already booked")
                return False
        except Exception as e:
            logger.error(f"Error booking slot {slot} for {date_time_key}: {e}")
            return False

    def get_date_display_name(self, date_str):
        """Get display name for a date"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            day_name = date_obj.strftime("%A")
            return f"{day_name}, {date_obj.strftime('%B %d, %Y')}"
        except ValueError:
            return date_str

    def preprocess_symptoms(self, text):
        """Preprocess symptoms text using NLP"""
        # Convert to lowercase
        text = text.lower()

        # Tokenize using updated NLTK
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            # Fallback to simple split
            tokens = text.split()

        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and token.isalpha():
                try:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
                except Exception as e:
                    logger.warning(f"Lemmatization error: {e}")
                    processed_tokens.append(token)

        return processed_tokens

    def analyze_symptoms(self, symptoms_list):
        """Analyze symptoms and predict possible diseases"""
        all_symptoms_text = " ".join(symptoms_list)
        processed_symptoms = self.preprocess_symptoms(all_symptoms_text)

        # Find matching symptoms
        matched_symptoms = []
        possible_diseases = set()

        for token in processed_symptoms:
            if token in self.symptom_disease_map:
                matched_symptoms.append(token)
                possible_diseases.update(self.symptom_disease_map[token])

        # Special handling for common combinations
        symptoms_lower = all_symptoms_text.lower()
        if 'blocked nose' in symptoms_lower or 'stuffy nose' in symptoms_lower:
            matched_symptoms.append('blocked_nose')
            possible_diseases.update(['Common Cold', 'Allergic Rhinitis', 'Sinusitis'])

        if 'sore throat' in symptoms_lower:
            matched_symptoms.append('sore_throat')
            possible_diseases.update(['Viral Pharyngitis', 'Strep Throat', 'Common Cold'])

        if 'body pain' in symptoms_lower or 'body ache' in symptoms_lower:
            matched_symptoms.append('body_pain')
            possible_diseases.update(['Flu', 'Viral Infection', 'Muscle Strain'])

        return list(set(matched_symptoms)), list(possible_diseases)

    def create_prescription_pdf(self, patient_data, unique_code):
        """Create prescription PDF"""
        filename = f"prescription_{unique_code}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Header
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            textColor=colors.blue
        )
        story.append(Paragraph("Medical Chatbot - Pre-Analysis Test", header_style))
        story.append(Spacer(1, 20))

        # Patient Information
        patient_info_style = ParagraphStyle(
            'PatientInfo',
            parent=styles['Normal'],
            fontSize=12,
            leftIndent=20
        )

        story.append(Paragraph("Patient Information:", styles['Heading2']))
        story.append(Paragraph(f"Name: {patient_data['name']}", patient_info_style))
        story.append(Paragraph(f"Age: {patient_data['age']}", patient_info_style))
        story.append(Paragraph(f"Gender: {patient_data['gender']}", patient_info_style))
        story.append(Paragraph(f"Blood Group: {patient_data['blood_group']}", patient_info_style))
        story.append(Paragraph(f"Contact: {patient_data['contact']}", patient_info_style))
        story.append(Spacer(1, 20))

        # Symptoms
        story.append(Paragraph("Reported Symptoms:", styles['Heading2']))
        for symptom in patient_data['symptoms']:
            story.append(Paragraph(f"• {symptom}", patient_info_style))
        story.append(Spacer(1, 20))

        # Analysis
        story.append(Paragraph("Pre-Analysis:", styles['Heading2']))
        story.append(Paragraph(
            f"Processed Symptoms: {', '.join(patient_data['matched_symptoms']) if patient_data['matched_symptoms'] else 'General symptoms detected'}",
            patient_info_style))
        story.append(Paragraph(
            f"Expected Diseases: {', '.join(patient_data['possible_diseases']) if patient_data['possible_diseases'] else 'Please consult doctor for proper diagnosis'}",
            patient_info_style))
        story.append(Spacer(1, 20))

        # Doctor Information
        story.append(Paragraph("Doctor Details:", styles['Heading2']))
        story.append(Paragraph(f"Doctor: {patient_data.get('doctor_name', 'N/A')}", patient_info_style))
        story.append(Paragraph(f"Specialty: {patient_data.get('doctor_specialty', 'N/A')}", patient_info_style))
        story.append(Spacer(1, 20))

        # Appointment Details
        story.append(Paragraph("Appointment Details:", styles['Heading2']))
        story.append(Paragraph(f"Date: {patient_data['selected_date_display']}", patient_info_style))
        story.append(Paragraph(f"Time: {patient_data['selected_time']}", patient_info_style))
        story.append(Paragraph(f"Slot: {patient_data['selected_slot']}", patient_info_style))
        story.append(Paragraph(f"Unique Code: {unique_code}", patient_info_style))
        story.append(Spacer(1, 20))

        # Footer
        story.append(Paragraph(
            "Note: This is a pre-analysis based on reported symptoms. Please consult with the doctor for proper diagnosis and treatment.",
            styles['Normal']))

        doc.build(story)
        return filename


# Initialize bot
bot = MedicalChatBot()


# Bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler with inline keyboard"""
    user_id = update.effective_user.id
    bot.user_states[user_id] = {'state': 'main_menu'}

    # Create inline keyboard buttons
    keyboard = [
        [InlineKeyboardButton("📅 Check Booking Details", callback_data="menu_check_slots")],
        [InlineKeyboardButton("🩺 Book New Appointment", callback_data="menu_book_appointment")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "🏥 Welcome to Medical Chatbot! 🏥\n\n"
        "I can help you with:\n"
        "• Check your existing booking details with unique code\n"
        "• Book a new doctor's appointment\n\n"
        "Please select an option:",
        reply_markup=reply_markup
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all text messages"""
    user_id = update.effective_user.id
    text = update.message.text

    if user_id not in bot.user_states:
        bot.user_states[user_id] = {'state': 'main_menu'}

    state_data = bot.user_states[user_id]
    current_state = state_data.get('state', 'main_menu')

    # Handle text input based on current state
    if current_state == 'waiting_code':
        await handle_code_verification(update, context)
    elif current_state == 'waiting_name':
        await handle_name_input(update, context)
    elif current_state == 'waiting_age':
        await handle_age_input(update, context)
    elif current_state == 'waiting_gender':
        await handle_gender_input(update, context)
    elif current_state == 'waiting_contact':
        await handle_contact_input(update, context)
    elif current_state == 'waiting_symptoms':
        await handle_symptoms_input(update, context)
    elif current_state == 'waiting_more_symptoms':
        await handle_more_symptoms(update, context)
    elif current_state == 'waiting_download_code':
        await handle_download_code(update, context)
    else:
        # If user sends random text, show main menu again
        await show_main_menu(update, context)


async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the main menu with inline keyboard"""
    keyboard = [
        [InlineKeyboardButton("📅 Check Booking Details", callback_data="menu_check_slots")],
        [InlineKeyboardButton("🩺 Book New Appointment", callback_data="menu_book_appointment")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "🏥 Welcome to Medical Chatbot! 🏥\n\n"
        "Please select an option:",
        reply_markup=reply_markup
    )


async def handle_check_slots(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Handle check booking details option"""
    user_id = update.effective_user.id if update else query.from_user.id
    bot.user_states[user_id]['state'] = 'waiting_code'

    message_text = "🔑 Please enter your 8-digit unique code to access your booking details:"

    if query:
        await query.edit_message_text(message_text)
    else:
        await update.message.reply_text(message_text, reply_markup=ReplyKeyboardRemove())


async def handle_code_verification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle code verification - now checks MongoDB"""
    user_id = update.effective_user.id
    code = update.message.text.strip()

    if len(code) != 8 or not code.isdigit():
        await update.message.reply_text("❌ Invalid code format. Please enter an 8-digit code:")
        return

    # Check MongoDB for booking details
    booking_details = await bot.get_booking_details_by_code(code)

    if booking_details:
        details_text = f"📋 **Booking Details:**\n\n"
        details_text += f"🔑 **Unique Code:** {booking_details['uniqueCode']}\n\n"
        details_text += f"👤 **Patient Information:**\n"
        details_text += f"• Name: {booking_details['patient']['name']}\n"
        details_text += f"• Age: {booking_details['patient']['age']}\n"
        details_text += f"• Gender: {booking_details['patient']['gender']}\n"
        details_text += f"• Blood Group: {booking_details['patient']['blood']}\n"
        details_text += f"• Contact: {booking_details['patient']['contact']}\n\n"
        details_text += f"👨⚕️ **Doctor Information:**\n"
        details_text += f"• Doctor: {booking_details['doctor']['name']}\n"
        details_text += f"• Specialty: {booking_details['doctor']['specialty']}\n\n"
        details_text += f"📅 **Appointment Details:**\n"
        details_text += f"• Date: {booking_details['appointment']['date']}\n"
        details_text += f"• Status: {booking_details['appointment']['status'].title()}\n"
        details_text += f"• Booked On: {booking_details['appointment']['createdAt'].strftime('%B %d, %Y at %I:%M %p')}\n\n"
        # details_text += f"🆔 **IDs:**\n"
        # details_text += f"• Patient ID: {booking_details['patientId']}\n"
        # details_text += f"• Confirmation ID: {booking_details['confirmationId']}\n"

        keyboard = [
            # [InlineKeyboardButton("📄 Download Prescription PDF", callback_data=f"download_pdf_{code}")],
            [InlineKeyboardButton("🏠 Back to Menu", callback_data="back_to_menu")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(details_text, reply_markup=reply_markup, parse_mode='Markdown')
    else:
        await update.message.reply_text("❌ Code not found. Please check your code and try again.")
        # Reset state
        bot.user_states[user_id]['state'] = 'main_menu'


async def handle_book_appointment(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Handle book new appointment"""
    user_id = update.effective_user.id if update else query.from_user.id
    bot.user_states[user_id] = {'state': 'waiting_name', 'patient_data': {}}

    message_text = "👤 Please enter your full name:"

    if query:
        await query.edit_message_text(message_text)
    else:
        await update.message.reply_text(message_text, reply_markup=ReplyKeyboardRemove())


async def handle_name_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle name input"""
    user_id = update.effective_user.id
    name = update.message.text.strip()

    bot.user_states[user_id]['patient_data']['name'] = name
    bot.user_states[user_id]['state'] = 'waiting_blood_group'

    # Blood group keyboard
    blood_keyboard = []
    for i in range(0, len(bot.blood_groups), 2):
        row = [InlineKeyboardButton(bot.blood_groups[i], callback_data=f"blood_{bot.blood_groups[i]}")]
        if i + 1 < len(bot.blood_groups):
            row.append(InlineKeyboardButton(bot.blood_groups[i + 1], callback_data=f"blood_{bot.blood_groups[i + 1]}"))
        blood_keyboard.append(row)

    reply_markup = InlineKeyboardMarkup(blood_keyboard)

    await update.message.reply_text(
        f"Hello {name}! 🩸 Please select your blood group:",
        reply_markup=reply_markup
    )


async def handle_age_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle age input"""
    user_id = update.effective_user.id
    age = update.message.text.strip()

    if not age.isdigit() or int(age) < 1 or int(age) > 120:
        await update.message.reply_text("❌ Please enter a valid age (1-120):")
        return

    bot.user_states[user_id]['patient_data']['age'] = age
    bot.user_states[user_id]['state'] = 'waiting_gender'

    gender_keyboard = [
        [InlineKeyboardButton("👨 Male", callback_data="gender_Male")],
        [InlineKeyboardButton("👩 Female", callback_data="gender_Female")],
        [InlineKeyboardButton("⚧ Other", callback_data="gender_Other")]
    ]

    reply_markup = InlineKeyboardMarkup(gender_keyboard)
    await update.message.reply_text("⚧ Please select your gender:", reply_markup=reply_markup)


async def handle_gender_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle gender input (fallback for text input)"""
    user_id = update.effective_user.id
    gender = update.message.text.strip()

    bot.user_states[user_id]['patient_data']['gender'] = gender
    bot.user_states[user_id]['state'] = 'waiting_contact'

    await update.message.reply_text("📞 Please enter your contact number:")


async def handle_contact_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle contact input"""
    user_id = update.effective_user.id
    contact = update.message.text.strip()

    bot.user_states[user_id]['patient_data']['contact'] = contact
    bot.user_states[user_id]['patient_data']['symptoms'] = []
    bot.user_states[user_id]['state'] = 'waiting_symptoms'

    await update.message.reply_text(
        "🩺 Please describe your symptoms (e.g., fever, headache, blocked nose, cough):\n\n"
        "You can type multiple symptoms separated by commas."
    )


async def handle_symptoms_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle symptoms input"""
    user_id = update.effective_user.id
    symptoms_text = update.message.text.strip()

    # Split by comma and clean up
    symptoms = [s.strip() for s in symptoms_text.split(',') if s.strip()]
    bot.user_states[user_id]['patient_data']['symptoms'].extend(symptoms)
    bot.user_states[user_id]['state'] = 'waiting_more_symptoms'

    keyboard = [
        [InlineKeyboardButton("✅ No, Continue", callback_data="no_more_symptoms")],
        [InlineKeyboardButton("➕ Yes, Add More", callback_data="add_more_symptoms")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"✅ Recorded symptoms: {', '.join(symptoms)}\n\n"
        "Do you have any other symptoms?",
        reply_markup=reply_markup
    )


async def handle_more_symptoms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle additional symptoms"""
    user_id = update.effective_user.id
    symptoms_text = update.message.text.strip()

    # Split by comma and clean up
    symptoms = [s.strip() for s in symptoms_text.split(',') if s.strip()]
    bot.user_states[user_id]['patient_data']['symptoms'].extend(symptoms)

    keyboard = [
        [InlineKeyboardButton("✅ No, Continue", callback_data="no_more_symptoms")],
        [InlineKeyboardButton("➕ Yes, Add More", callback_data="add_more_symptoms")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"✅ Added symptoms: {', '.join(symptoms)}\n\n"
        "Any other symptoms?",
        reply_markup=reply_markup
    )


async def show_doctor_selection(update, context, user_id):
    """Show available doctors for selection"""
    doctors = bot.get_available_doctors()

    if not doctors:
        message_text = "❌ Sorry, no doctors are currently available. Please try again later."
        if hasattr(update, 'message'):
            await update.message.reply_text(message_text)
        else:
            await update.edit_message_text(message_text)
        return

    # Create doctor selection keyboard
    doctor_keyboard = []
    for i, doctor in enumerate(doctors):
        doctor_display = f"Dr. {doctor['name']}"
        if doctor['specialty']:
            doctor_display += f" - {doctor['specialty']}"
        if doctor['availability']:
            doctor_display += f" ({doctor['availability']})"

        doctor_keyboard.append([
            InlineKeyboardButton(
                doctor_display,
                callback_data=f"select_doctor_{i}"
            )
        ])

    reply_markup = InlineKeyboardMarkup(doctor_keyboard)
    message_text = "👨⚕️ **Available Doctors:**\n\nPlease select a doctor for your appointment:"

    if hasattr(update, 'message'):
        await update.message.reply_text(
            message_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    else:
        await update.edit_message_text(
            message_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )


async def analyze_and_show_dates(update, context, user_id, doctor_info=None):
    """Analyze symptoms and show available dates from MongoDB"""
    patient_data = bot.user_states[user_id]['patient_data']

    # Analyze symptoms
    matched_symptoms, possible_diseases = bot.analyze_symptoms(patient_data['symptoms'])
    patient_data['matched_symptoms'] = matched_symptoms
    patient_data['possible_diseases'] = possible_diseases

    # Show analysis
    analysis_text = "🧪 **Pre-Analysis Results:**\n\n"
    analysis_text += f"📝 **Your Symptoms:** {', '.join(patient_data['symptoms'])}\n\n"
    analysis_text += f"🔍 **Processed Symptoms:** {', '.join(matched_symptoms) if matched_symptoms else 'General symptoms detected'}\n\n"

    if doctor_info:
        analysis_text += f"👨⚕️ **Selected Doctor:** Dr. {doctor_info['name']}\n"
        analysis_text += f"🏥 **Specialty:** {doctor_info['specialty']}\n"
        analysis_text += f"🕐 **Availability:** {doctor_info['availability']}\n\n"

    # Get available dates from MongoDB filtered by doctor availability
    doctor_availability = doctor_info['availability'] if doctor_info else None
    available_dates = bot.get_next_7_upcoming_dates(doctor_availability)

    if available_dates:
        analysis_text += "📅 **Available Appointment Dates (Next 7 Days):**\n"

        date_keyboard = []
        for date_info in available_dates:
            date_keyboard.append([
                InlineKeyboardButton(
                    f"📅 {date_info['display_name']} ({date_info['total_available_slots']} slots)",
                    callback_data=f"select_date_{date_info['date']}"
                )
            ])

        # Add back button to select different doctor
        if doctor_info:
            date_keyboard.append([
                InlineKeyboardButton("🔙 Select Different Doctor", callback_data="back_to_doctor_selection")
            ])

        reply_markup = InlineKeyboardMarkup(date_keyboard)

        # Handle both regular message and callback query
        if hasattr(update, 'message'):
            await update.message.reply_text(
                analysis_text + "\nPlease select your preferred appointment date:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        else:
            await update.edit_message_text(
                analysis_text + "\nPlease select your preferred appointment date:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
    else:
        message_text = analysis_text + f"\n❌ Sorry, Dr. {doctor_info['name']} has no available slots in the selected time range. Please select a different doctor."

        # Add button to go back to doctor selection
        keyboard = [
            [InlineKeyboardButton("🔙 Select Different Doctor", callback_data="back_to_doctor_selection")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        if hasattr(update, 'message'):
            await update.message.reply_text(message_text, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.edit_message_text(message_text, reply_markup=reply_markup, parse_mode='Markdown')


async def show_time_slots_for_date(query, context, user_id, selected_date):
    """Show available time slots for selected date"""
    try:
        patient_data = bot.user_states[user_id]['patient_data']
        doctor_info = patient_data.get('selected_doctor')
        doctor_availability = doctor_info['availability'] if doctor_info else None
        time_slots = bot.get_available_slots_for_date(selected_date, doctor_availability)

        bot.user_states[user_id]['patient_data']['selected_date'] = selected_date
        bot.user_states[user_id]['patient_data']['selected_date_display'] = bot.get_date_display_name(selected_date)

        # Create time slots keyboard
        time_keyboard = []
        for i, time_slot in enumerate(time_slots):
            slot_count = len(time_slot['available_slots'])
            time_keyboard.append([
                InlineKeyboardButton(
                    # f"🕐 {time_slot['time']} ({slot_count} slots)",
                    f"🕐 {slot_count} slots",
                    callback_data=f"select_time_{selected_date}_{i}"
                )
            ])

        # Add back button
        time_keyboard.append([
            InlineKeyboardButton("🔙 Back to Date Selection", callback_data="back_to_dates")
        ])

        reply_markup = InlineKeyboardMarkup(time_keyboard)
        date_display = bot.get_date_display_name(selected_date)

        await query.edit_message_text(
            f"🕐 **Available Time Slots for {date_display}:**\n\nPlease select your preferred time slot:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    except Exception as e:
        logging.error(f"Error showing time slots: {e}")


async def show_slots_for_time(query, context, user_id, selected_date, time_index):
    """Show individual slots for the selected time"""
    patient_data = bot.user_states[user_id]['patient_data']
    doctor_info = patient_data.get('selected_doctor')
    doctor_availability = doctor_info['availability'] if doctor_info else None

    time_slots = bot.get_available_slots_for_date(selected_date, doctor_availability)

    if time_index >= len(time_slots):
        await query.edit_message_text("❌ Selected time slot is no longer available. Please try again.")
        return

    selected_time_slot = time_slots[time_index]
    available_slots = selected_time_slot['available_slots']

    if not available_slots:
        await query.edit_message_text(
            "❌ Sorry, no slots are available for the selected time. Please choose another time."
        )
        return

    # Store selected time
    bot.user_states[user_id]['patient_data']['selected_time'] = selected_time_slot['time']
    bot.user_states[user_id]['patient_data']['selected_date_time_key'] = selected_time_slot['date_time_key']

    # Create slots keyboard
    slot_keyboard = []
    for i, slot in enumerate(available_slots):
        slot_keyboard.append([
            InlineKeyboardButton(
                f"🎫 {slot}",
                callback_data=f"book_slot_{selected_date}_{time_index}_{i}"
            )
        ])

    # Add back button
    slot_keyboard.append([
        InlineKeyboardButton("🔙 Back to Time Selection", callback_data=f"select_date_{selected_date}")
    ])

    reply_markup = InlineKeyboardMarkup(slot_keyboard)
    date_display = bot.get_date_display_name(selected_date)

    await query.edit_message_text(
        f"🎫 **Available Slots for {date_display}:**\n\nPlease select your preferred slot:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries from inline keyboards"""
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    await query.answer()

    try:
        # Main menu callbacks
        if data == "menu_check_slots":
            await handle_check_slots(None, context, query)
        elif data == "menu_book_appointment":
            await handle_book_appointment(None, context, query)
        elif data.startswith("blood_"):
            blood_group = data.replace("blood_", "")
            bot.user_states[user_id]['patient_data']['blood_group'] = blood_group
            bot.user_states[user_id]['state'] = 'waiting_age'

            await query.edit_message_text(f"✅ Blood group selected: {blood_group}")
            await context.bot.send_message(user_id, "🎂 Please enter your age:")
        elif data.startswith("gender_"):
            gender = data.replace("gender_", "")
            bot.user_states[user_id]['patient_data']['gender'] = gender
            bot.user_states[user_id]['state'] = 'waiting_contact'

            await query.edit_message_text(f"✅ Gender selected: {gender}")
            await context.bot.send_message(user_id, "📞 Please enter your contact number:")
        elif data == "add_more_symptoms":
            bot.user_states[user_id]['state'] = 'waiting_more_symptoms'
            await query.edit_message_text("➕ Please type your additional symptoms:")
        elif data == "no_more_symptoms":
            await query.edit_message_text("✅ Symptom collection completed. Please select a doctor...")
            await show_doctor_selection(query, context, user_id)
        elif data.startswith("select_doctor_"):
            doctor_index = int(data.replace("select_doctor_", ""))
            doctors = bot.get_available_doctors()

            if doctor_index >= len(doctors):
                await query.edit_message_text("❌ Selected doctor is no longer available. Please try again.")
                return

            selected_doctor = doctors[doctor_index]
            bot.user_states[user_id]['patient_data']['selected_doctor'] = selected_doctor
            bot.user_states[user_id]['patient_data']['doctor_name'] = selected_doctor['name']
            bot.user_states[user_id]['patient_data']['doctor_specialty'] = selected_doctor['specialty']

            await query.edit_message_text(
                f"✅ Doctor selected: Dr. {selected_doctor['name']}\n\nAnalyzing and fetching available slots...")
            await analyze_and_show_dates(query, context, user_id, selected_doctor)
        elif data == "back_to_doctor_selection":
            await show_doctor_selection(query, context, user_id)
        elif data.startswith("select_date_"):
            selected_date = data.replace("select_date_", "")
            await show_time_slots_for_date(query, context, user_id, selected_date)
        elif data == "back_to_dates":
            patient_data = bot.user_states[user_id]['patient_data']
            doctor_info = patient_data.get('selected_doctor')
            await analyze_and_show_dates(query, context, user_id, doctor_info)
        elif data.startswith("select_time_"):
            parts = data.replace("select_time_", "").split("_")
            selected_date = parts[0]
            time_index = int(parts[1])
            await show_slots_for_time(query, context, user_id, selected_date, time_index)
        elif data.startswith("book_slot_"):
            # Extract appointment details
            parts = data.split("_")
            selected_date = parts[2]
            time_index = int(parts[3])
            slot_index = int(parts[4])

            # Get slot information
            patient_data = bot.user_states[user_id]['patient_data']
            doctor_info = patient_data.get('selected_doctor')
            doctor_availability = doctor_info['availability'] if doctor_info else None
            time_slots = bot.get_available_slots_for_date(selected_date, doctor_availability)

            if time_index < len(time_slots):
                selected_time_slot = time_slots[time_index]
                if slot_index < len(selected_time_slot['available_slots']):
                    selected_slot = selected_time_slot['available_slots'][slot_index]

                    # Store selected slot information
                    bot.user_states[user_id]['patient_data']['selected_slot'] = selected_slot

            # Complete booking process (save to MongoDB)
            success, result = await bot.complete_booking_process(user_id, {})

            if success:
                patient_data = bot.user_states[user_id]['patient_data']
                doctor_info = patient_data.get('selected_doctor')
                patient_id = result['patient_id']
                confirmation_id = result['confirmation_id']
                unique_code = result['unique_code']

                confirmation_message = f"""
✅ **Appointment Confirmed!**

🔑 **Your Unique Code:** `{unique_code}`
*Save this code to check your booking details anytime*

👤 **Patient:** {patient_data['name']}
📞 **Contact:** {patient_data['contact']}
🩺 **Doctor:** {doctor_info['name']}
📅 **Date:** {bot.get_date_display_name(selected_date)}
🎫 **Slot:** {patient_data.get('selected_slot', 'N/A')}


Thank you for booking with us! Use your unique code {unique_code} to check details anytime.

To Book an Appointment again please type "/start" and press enter.
"""
                # 🕐 ** Time: ** {patient_data.get('selected_time', 'N/A')}
                # 📋 **Confirmation ID:** {str(confirmation_id)}
                # 🆔 **Patient ID:** {str(patient_id)}

                keyboard = [
                    # [InlineKeyboardButton("📄 Download Prescription PDF", callback_data=f"download_pdf_{unique_code}")],
                    # [InlineKeyboardButton("🏠 Back to Menu", callback_data="back_to_menu")]
                ]

                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    confirmation_message,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )

                # Clear user state after successful booking
                if user_id in bot.user_states:
                    del bot.user_states[user_id]
            else:
                await query.edit_message_text(
                    f"❌ **Booking Failed**\n\n{result}\n\nPlease try again or contact support.",
                    parse_mode='Markdown'
                )
        elif data.startswith("download_pdf_"):
            unique_code = data.replace("download_pdf_", "")
            # Check if data exists in memory first, then MongoDB
            patient_data = None

            if unique_code in bot.patients_data:
                patient_data = bot.patients_data[unique_code]
            else:
                # Get from MongoDB
                booking_details = await bot.get_booking_details_by_code(unique_code)
                if booking_details:
                    # Convert to expected format
                    patient_data = {
                        'name': booking_details['patient']['name'],
                        'age': booking_details['patient']['age'],
                        'gender': booking_details['patient']['gender'],
                        'blood_group': booking_details['patient']['blood'],
                        'contact': booking_details['patient']['contact'],
                        'symptoms': [],  # Will need to be stored in patient document
                        'matched_symptoms': [],
                        'possible_diseases': [],
                        'doctor_name': booking_details['doctor']['name'],
                        'doctor_specialty': booking_details['doctor']['specialty'],
                        'selected_date': booking_details['appointment']['date'],
                        'selected_date_display': booking_details['appointment']['date'],
                        # 'selected_time': '',  # Will need to be stored in confirmation document
                        'selected_slot': '',
                    }

            if patient_data:
                try:
                    # Create PDF
                    filename = bot.create_prescription_pdf(patient_data, unique_code)

                    # Send PDF
                    with open(filename, 'rb') as pdf_file:
                        await context.bot.send_document(
                            chat_id=user_id,
                            document=pdf_file,
                            filename=f"prescription_{unique_code}.pdf",
                            caption="📄 Your Pre-Analysis Prescription has been generated!"
                        )

                    # Clean up file
                    os.remove(filename)
                    await query.answer("✅ Prescription downloaded successfully!", show_alert=True)

                except Exception as e:
                    logger.error(f"Error creating PDF: {e}")
                    await query.answer("❌ Error generating prescription. Please try again later.", show_alert=True)
            else:
                await query.answer("❌ Booking details not found.", show_alert=True)

        elif data == "back_to_menu":
            # Show main menu with inline keyboard
            keyboard = [
                [InlineKeyboardButton("📅 Check Booking Details", callback_data="menu_check_slots")],
                [InlineKeyboardButton("🩺 Book New Appointment", callback_data="menu_book_appointment")]
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "🏥 Welcome back to Medical Chatbot!\n\nPlease select an option:",
                reply_markup=reply_markup
            )

            # Reset user state
            bot.user_states[user_id]['state'] = 'main_menu'

    except Exception as e:
        logger.error(f"Error in callback query handler: {e}")
        await query.answer("❌ An error occurred. Please try again.", show_alert=True)


async def handle_download_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle prescription download code verification"""
    user_id = update.effective_user.id
    code = update.message.text.strip()

    if len(code) != 8 or not code.isdigit():
        await update.message.reply_text("❌ Invalid code format. Please enter an 8-digit code:")
        return

    # Check MongoDB for patient data
    booking_details = await bot.get_booking_details_by_code(code)

    if booking_details:
        # Convert to expected format for PDF generation
        patient_data = {
            'name': booking_details['patient']['name'],
            'age': booking_details['patient']['age'],
            'gender': booking_details['patient']['gender'],
            'blood_group': booking_details['patient']['blood'],
            'contact': booking_details['patient']['contact'],
            'symptoms': [],  # Would need to be stored in patient document
            'matched_symptoms': [],
            'possible_diseases': [],
            'doctor_name': booking_details['doctor']['name'],
            'doctor_specialty': booking_details['doctor']['specialty'],
            'selected_date': booking_details['appointment']['date'],
            'selected_date_display': booking_details['appointment']['date'],
            # 'selected_time': '',
            'selected_slot': '',
        }

        try:
            # Create PDF
            filename = bot.create_prescription_pdf(patient_data, code)

            # Send PDF
            with open(filename, 'rb') as pdf_file:
                await context.bot.send_document(
                    chat_id=user_id,
                    document=pdf_file,
                    filename=f"prescription_{code}.pdf",
                    caption="📄 Your Pre-Analysis Prescription has been generated!"
                )

            # Clean up file
            os.remove(filename)
            await update.message.reply_text("✅ Prescription downloaded successfully!")

        except Exception as e:
            logger.error(f"Error creating PDF: {e}")
            await update.message.reply_text("❌ Error generating prescription. Please try again later.")
    else:
        await update.message.reply_text("❌ Code not found. Please check your code and try again.")

    # Reset state
    bot.user_states[user_id]['state'] = 'main_menu'


def main():
    """Run the bot"""
    # Create application
    from telegram.ext import ApplicationBuilder

    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .build()
    )

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    # Run the bot
    print("🤖 Medical Chatbot is starting...")
    print("📅 Fetching upcoming appointment dates from MongoDB...")

    # Test MongoDB connection and show available dates
    try:
        upcoming_dates = bot.get_next_7_upcoming_dates()
        print(f"✅ Found {len(upcoming_dates)} upcoming dates with available slots")
        for date_info in upcoming_dates:
            print(f" 📅 {date_info['display_name']} - {date_info['total_available_slots']} total slots")

        # Test doctors collection
        doctors = bot.get_available_doctors()
        print(f"\n✅ Found {len(doctors)} available doctors")
        for doctor in doctors:
            print(f" 👨⚕️ Dr. {doctor['name']} - {doctor['specialty']} ({doctor['availability']})")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
