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


# Download required NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

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
        # In-memory storage for testing
        self.patients_data = {}
        self.available_slots = [
            "Today 10:00 AM", "Today 2:00 PM", "Today 4:00 PM",
            "Tomorrow 9:00 AM", "Tomorrow 11:00 AM", "Tomorrow 3:00 PM",
            "Day After Tomorrow 10:00 AM", "Day After Tomorrow 1:00 PM"
        ]
        self.booked_slots = []

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
        """Generate 8-digit unique code"""
        return ''.join(random.choices(string.digits, k=8))

    def preprocess_symptoms(self, text):
        """Preprocess symptoms text using NLP"""
        # Convert to lowercase
        text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and token.isalpha():
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)

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

        story.append(Paragraph("<b>Patient Information:</b>", styles['Heading2']))
        story.append(Paragraph(f"<b>Name:</b> {patient_data['name']}", patient_info_style))
        story.append(Paragraph(f"<b>Age:</b> {patient_data['age']}", patient_info_style))
        story.append(Paragraph(f"<b>Gender:</b> {patient_data['gender']}", patient_info_style))
        story.append(Paragraph(f"<b>Blood Group:</b> {patient_data['blood_group']}", patient_info_style))
        story.append(Paragraph(f"<b>Contact:</b> {patient_data['contact']}", patient_info_style))
        story.append(Spacer(1, 20))

        # Symptoms
        story.append(Paragraph("<b>Reported Symptoms:</b>", styles['Heading2']))
        for symptom in patient_data['symptoms']:
            story.append(Paragraph(f"• {symptom}", patient_info_style))
        story.append(Spacer(1, 20))

        # Analysis
        story.append(Paragraph("<b>Pre-Analysis:</b>", styles['Heading2']))
        story.append(Paragraph(
            f"<b>Processed Symptoms:</b> {', '.join(patient_data['matched_symptoms']) if patient_data['matched_symptoms'] else 'General symptoms detected'}",
            patient_info_style))
        story.append(Paragraph(
            f"<b>Expected Diseases:</b> {', '.join(patient_data['possible_diseases']) if patient_data['possible_diseases'] else 'Please consult doctor for proper diagnosis'}",
            patient_info_style))
        story.append(Spacer(1, 20))

        # Appointment Details
        story.append(Paragraph("<b>Appointment Details:</b>", styles['Heading2']))
        story.append(Paragraph(f"<b>Slot:</b> {patient_data['selected_slot']}", patient_info_style))
        story.append(Paragraph(f"<b>Unique Code:</b> {unique_code}", patient_info_style))
        story.append(Spacer(1, 20))

        # Footer
        story.append(Paragraph(
            "<b>Note:</b> This is a pre-analysis based on reported symptoms. Please consult with the doctor for proper diagnosis and treatment.",
            styles['Normal']))

        doc.build(story)
        return filename


# Initialize bot
bot = MedicalChatBot()


# Bot handlers -- Updated with inline keyboard
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler with inline keyboard"""
    user_id = update.effective_user.id
    bot.user_states[user_id] = {'state': 'main_menu'}

    # Create inline keyboard buttons
    keyboard = [
        [InlineKeyboardButton("📅 Check Slots & Pre-Analysis", callback_data="menu_check_slots")],
        [InlineKeyboardButton("🩺 Book New Appointment", callback_data="menu_book_appointment")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "🏥 Welcome to Medical Chatbot! 🏥\n\n"
        "I can help you with:\n"
        "• Check your existing slots and Pre-Analysis Test Prescription\n"
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
        [InlineKeyboardButton("📅 Check Slots & Pre-Analysis", callback_data="menu_check_slots")],
        [InlineKeyboardButton("🩺 Book New Appointment", callback_data="menu_book_appointment")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "🏥 Welcome to Medical Chatbot! 🏥\n\n"
        "Please select an option:",
        reply_markup=reply_markup
    )


async def handle_check_slots(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Handle check slots option"""
    user_id = update.effective_user.id if update else query.from_user.id
    bot.user_states[user_id]['state'] = 'waiting_code'

    message_text = "🔑 Please enter your 8-digit unique code to access your details:"

    if query:
        await query.edit_message_text(message_text)
    else:
        await update.message.reply_text(message_text, reply_markup=ReplyKeyboardRemove())


async def handle_code_verification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle code verification"""
    user_id = update.effective_user.id
    code = update.message.text.strip()

    if len(code) != 8 or not code.isdigit():
        await update.message.reply_text("❌ Invalid code format. Please enter an 8-digit code:")
        return

    if code in bot.patients_data:
        patient_data = bot.patients_data[code]

        details_text = f"📋 **Patient Details:**\n\n"
        details_text += f"👤 **Name:** {patient_data['name']}\n"
        details_text += f"🩸 **Blood Group:** {patient_data['blood_group']}\n"
        details_text += f"🎂 **Age:** {patient_data['age']}\n"
        details_text += f"⚧ **Gender:** {patient_data['gender']}\n"
        details_text += f"📞 **Contact:** {patient_data['contact']}\n\n"
        details_text += f"🩺 **Symptoms:** {', '.join(patient_data['symptoms'])}\n\n"
        details_text += f"📅 **Appointed Slot:** {patient_data['selected_slot']}\n"
        details_text += f"🔑 **Unique Code:** {code}\n\n"
        details_text += f"🧪 **Expected Diseases:** {', '.join(patient_data['possible_diseases']) if patient_data['possible_diseases'] else 'Consult doctor'}"

        keyboard = [
            # [InlineKeyboardButton("📋 Download Prescription", callback_data="download_prescription")],
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


async def analyze_and_show_slots(update, context, user_id):
    """Analyze symptoms and show available slots"""
    patient_data = bot.user_states[user_id]['patient_data']

    # Analyze symptoms
    matched_symptoms, possible_diseases = bot.analyze_symptoms(patient_data['symptoms'])
    patient_data['matched_symptoms'] = matched_symptoms
    patient_data['possible_diseases'] = possible_diseases

    # Show analysis
    analysis_text = "🧪 **Pre-Analysis Results:**\n\n"
    analysis_text += f"📝 **Your Symptoms:** {', '.join(patient_data['symptoms'])}\n\n"
    analysis_text += f"🔍 **Processed Symptoms:** {', '.join(matched_symptoms) if matched_symptoms else 'General symptoms detected'}\n\n"
    # analysis_text += f"🩺 **Expected Diseases:** {', '.join(possible_diseases) if possible_diseases else 'Please consult doctor for proper diagnosis'}\n\n"

    # Show available slots
    if bot.available_slots:
        analysis_text += "📅 **Available Appointment Slots:**\n"

        slot_keyboard = []
        for i, slot in enumerate(bot.available_slots):
            slot_keyboard.append([InlineKeyboardButton(f"📅 {slot}", callback_data=f"select_slot_{i}")])

        reply_markup = InlineKeyboardMarkup(slot_keyboard)

        # Handle both regular message and callback query
        if hasattr(update, 'message'):
            await update.message.reply_text(
                analysis_text + "\nPlease select your preferred appointment slot:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        else:
            await update.edit_message_text(
                analysis_text + "\nPlease select your preferred appointment slot:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
    else:
        message_text = analysis_text + "\n❌ Sorry, no slots are currently available. Please try again later."
        if hasattr(update, 'message'):
            await update.message.reply_text(message_text, parse_mode='Markdown')
        else:
            await update.edit_message_text(message_text, parse_mode='Markdown')


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
            await query.edit_message_text("✅ Symptom collection completed. Analyzing...")
            await analyze_and_show_slots(query, context, user_id)

        elif data.startswith("select_slot_"):
            slot_index = int(data.replace("select_slot_", ""))
            selected_slot = bot.available_slots[slot_index]
            bot.user_states[user_id]['patient_data']['selected_slot'] = selected_slot

            confirm_keyboard = [
                [InlineKeyboardButton("✅ Confirm Slot", callback_data=f"confirm_slot_{slot_index}")],
                [InlineKeyboardButton("🔄 Reselect Slot", callback_data="reselect_slot")]
            ]
            reply_markup = InlineKeyboardMarkup(confirm_keyboard)

            await query.edit_message_text(
                f"📅 **Selected Slot:** {selected_slot}\n\n"
                "Please confirm your selection:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        elif data.startswith("confirm_slot_"):
            slot_index = int(data.replace("confirm_slot_", ""))
            selected_slot = bot.available_slots[slot_index]

            # Move slot from available to booked
            bot.available_slots.remove(selected_slot)
            bot.booked_slots.append(selected_slot)

            # Generate unique code and save patient data
            unique_code = bot.generate_unique_code()
            bot.patients_data[unique_code] = bot.user_states[user_id]['patient_data'].copy()

            # Create copy button for the code
            # menu_keyboard = [
            #             #     [InlineKeyboardButton("📋 Copy Code",callback_data=f"copy_code_{unique_code}")],
            #             #     [InlineKeyboardButton("📄 Download Prescription", callback_data="download_prescription")],
            #             #     [InlineKeyboardButton("🏠 Back to Menu", callback_data="back_to_menu")]
            #             # ]
            #             # reply_markup = InlineKeyboardMarkup(menu_keyboard)

            success_message = f"✅ **Appointment Booked Successfully!**\n\n"
            success_message += f"📅 **Slot:** {selected_slot}\n"
            success_message += f"🔑 **Your Unique Code:** `{unique_code}`\n\n"
            success_message += "NOTE: Click on the unique code to copy the CODE! \n\n"
            success_message += "⚠️ **Important:** Do not forget your 8-digit unique code as it is important for your verification!\n\n"
            success_message += "You can now download your Pre-Analysis prescription."



            await query.edit_message_text(
                success_message,
                # reply_markup=reply_markup,
                parse_mode='Markdown'
            )

            # Reset user state
            bot.user_states[user_id] = {'state': 'main_menu'}

        elif data == "reselect_slot":
            # Show available slots again
            if bot.available_slots:
                slot_keyboard = []
                for i, slot in enumerate(bot.available_slots):
                    slot_keyboard.append([InlineKeyboardButton(f"📅 {slot}", callback_data=f"select_slot_{i}")])

                reply_markup = InlineKeyboardMarkup(slot_keyboard)
                await query.edit_message_text(
                    "📅 **Available Appointment Slots:**\n\nPlease select your preferred appointment slot:",
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )

        # # elif data.startswith("copy_code_"):
        # #     unique_code = data.replace("copy_code_", "")
        # #     await query.answer(f"Code copied: {unique_code}", show_alert=True)
        #
        # elif query.data == "copy_code_":
        #     unique_code = data.replace("copy_code_", "")
        #     await query.message.reply_text(f"✅ Code copied: <code>{unique_code}</code>", parse_mode="HTML")

        # elif data == "download_prescription":
        #     bot.user_states[user_id]['state'] = 'waiting_download_code'
        #     await query.edit_message_text("🔑 Please enter your 8-digit unique code to download the prescription:")


        elif data == "back_to_menu":
            # Show main menu with inline keyboard
            keyboard = [
                [InlineKeyboardButton("📅 Check Slots & Pre-Analysis", callback_data="menu_check_slots")],
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

    if code in bot.patients_data:
        patient_data = bot.patients_data[code]

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
    # Replace with your actual bot token
    # BOT_TOKEN = "8399702585:AAG0f6QqWPG3kBrenZA01KG6IcPDgqdLP14"
    BOT_TOKEN = "8399702585:AAG0f6QqWPG3kBrenZA01KG6IcPDgqdLP14"
    # Create application with timezone configuration
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
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()

# Requirements for this version:
# pip install python-telegram-bot==20.7
# pip install nltk==3.8.1
# pip install reportlab==4.0.7
# pip install pytz==2023.3

# If you still get timezone errors, try:
# pip install APScheduler==3.10.4

# Alternative requirements (if above doesn't work):
# pip install python-telegram-bot[job-queue]==20.7


